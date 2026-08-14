"""Protocol freezing and simulator collection for Stage 3 NCER-AA."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os

import h5py
import numpy as np

from . import config
from .env_adapter import LiberoTaskRuntime
from .pipeline import _compare_rollouts
from .stage2 import (
    INPUT_COMMIT as STAGE2_INPUT_COMMIT,
    LIBERO_COMMIT,
    LIBERO_TREE_SHA256,
    STAGE1_5_DISPOSITION,
    STAGE1_5_RESULT_COMMIT,
    STAGE1_DISPOSITION,
    STAGE1_PUBLISHED_COMMIT,
    _array_hash,
    _episode_hash,
)
from .stage3_config import (
    ACTION_BANK_SIZE,
    BASE_ACTION_ABS_LIMIT,
    CALIBRATION_EPISODES,
    CONFIRMATION_EPISODES,
    CONTINUOUS_DIM,
    DEVELOPMENT_EPISODES,
    DIRECTION_COUNT,
    DIRECTION_FAMILIES,
    DIRECTION_FAMILY_COUNTS,
    EXECUTION_AMENDMENT,
    CONFIRMATION_INTEGRITY_AMENDMENT,
    GLOBAL_SEED,
    HISTORICAL_EPISODES,
    MAX_CROSS_SPLIT_ABS_COSINE,
    MAX_DIRECTION_COMPONENT,
    PHASES,
    PRIMARY_K,
    RADII_PER_DIRECTION,
    RADIUS_INTERVAL,
    SCRATCH_ROOT,
    SIGNS,
    SPLIT_EPISODES,
    SUPPORT_SEEDS,
    TASKS,
    TRAIN_EPISODES,
    model_definitions,
    split_for_episode,
)
from .storage import atomic_json, atomic_npz, sha256_file, sha256_tree


# Stage 3 begins from the published Stage 2 report commit.
STAGE3_INPUT_COMMIT = "74c98979910a3831d0abeb8d13111a7c9294b067"
STAGE2_DISPOSITION = "ORACLE_ONLY_NO_DEPLOYABLE_MODEL"
STAGE2_ACTION_BANK_RELATIVE = "experiments/r13_p15_ncea/stage2/action_bank.npz"
STAGE2_ROOT_RELATIVE = "experiments/r13_p15_ncea/stage2"


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _stable_seed(*parts):
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _task(task_id):
    for task in TASKS:
        if task["task_id"] == task_id:
            return dict(task)
    raise KeyError(task_id)


def _demo_path(paths, task):
    return os.path.join(paths["dataset_root"], config.SUITE, task["task_name"] + "_demo.hdf5")


def _dct_basis(length=4):
    basis = np.zeros((length, length), dtype=np.float64)
    for frequency in range(length):
        scale = math.sqrt(1.0 / length) if frequency == 0 else math.sqrt(2.0 / length)
        for time in range(length):
            basis[frequency, time] = scale * math.cos(
                math.pi * (time + 0.5) * frequency / length
            )
    return basis


def _normalize_direction(value):
    value = np.asarray(value, dtype=np.float64).reshape(4, 6)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("zero direction")
    output = (value / norm).reshape(-1)
    if float(np.max(np.abs(output))) > MAX_DIRECTION_COMPONENT:
        raise ValueError("coordinate-concentrated direction")
    return output


def _candidate_direction(rng, family, family_index):
    dct = _dct_basis(4)
    if family == "smooth_dct":
        coefficients = rng.normal(size=4) / np.asarray([1.0, 1.5, 4.0, 8.0])
        temporal = coefficients.dot(dct)
        value = np.outer(temporal, rng.normal(size=6))
        value += 0.15 * np.outer(
            (rng.normal(size=4) / np.asarray([2.0, 4.0, 8.0, 16.0])).dot(dct),
            rng.normal(size=6),
        )
    elif family == "suffix_contact":
        start = 1 + (int(family_index) % 2)
        temporal = np.zeros(4, dtype=np.float64)
        temporal[start:] = np.linspace(0.35, 1.0, 4 - start)
        temporal *= float(rng.choice([-1.0, 1.0]))
        value = np.outer(temporal, rng.normal(size=6))
        value += 0.10 * np.outer(temporal ** 2, rng.normal(size=6))
    elif family == "low_rank_temporal_action":
        value = np.zeros((4, 6), dtype=np.float64)
        for _ in range(2):
            value += np.outer(rng.normal(size=4), rng.normal(size=6))
    else:
        raise KeyError(family)
    return _normalize_direction(value)


def generate_support_codebooks():
    """Generate three fixed split codebooks under the strict cosine bound."""
    accepted = []
    result = {}
    for split in ("calibration", "development", "confirmation"):
        rng = np.random.RandomState(SUPPORT_SEEDS[split])
        directions = []
        family_ids = []
        attempts = 0
        for family_id, family in enumerate(DIRECTION_FAMILIES):
            target = int(DIRECTION_FAMILY_COUNTS[family])
            family_index = 0
            while family_index < target:
                attempts += 1
                if attempts > 100000:
                    raise RuntimeError("BLOCKED_SUPPORT_SEPARATION")
                try:
                    candidate = _candidate_direction(rng, family, family_index)
                except ValueError:
                    continue
                if accepted:
                    similarity = float(np.max(np.abs(np.asarray(accepted).dot(candidate))))
                    if similarity > MAX_CROSS_SPLIT_ABS_COSINE + 1e-12:
                        continue
                # Exact antipodes/duplicates within a split are also excluded.
                if directions and float(np.max(np.abs(np.asarray(directions).dot(candidate)))) > 0.999999:
                    continue
                directions.append(candidate)
                family_ids.append(family_id)
                family_index += 1
        directions = np.asarray(directions, dtype=np.float64)
        if directions.shape != (DIRECTION_COUNT, CONTINUOUS_DIM):
            raise AssertionError(directions.shape)
        radii = np.zeros((DIRECTION_COUNT, RADII_PER_DIRECTION), dtype=np.float64)
        for direction_id in range(DIRECTION_COUNT):
            radius_rng = np.random.RandomState(
                _stable_seed(SUPPORT_SEEDS[split], "radii", direction_id)
            )
            radii[direction_id] = np.sort(
                radius_rng.uniform(RADIUS_INTERVAL[0], RADIUS_INTERVAL[1], size=2)
            )
        result[split] = {
            "directions": directions,
            "radii": radii,
            "family_id": np.asarray(family_ids, dtype=np.int8),
            "seed": int(SUPPORT_SEEDS[split]),
        }
        accepted.extend(directions.tolist())
    return result


def support_separation_evidence(codebooks, action_bank):
    direction_hashes = {}
    residual_hashes = {}
    bank_hashes = {_array_hash(row) for row in np.asarray(action_bank, dtype=np.float64)}
    for split, bank in codebooks.items():
        directions = np.asarray(bank["directions"], dtype=np.float64)
        direction_hashes[split] = {_array_hash(row) for row in directions}
        residuals = []
        for direction_id, direction in enumerate(directions):
            for radius in bank["radii"][direction_id]:
                for sign in SIGNS:
                    residuals.append(float(sign) * float(radius) * direction)
        residual_hashes[split] = {_array_hash(row) for row in residuals}
    exact_direction_overlap = {}
    exact_residual_overlap = {}
    maximum_cosine = {}
    names = ("calibration", "development", "confirmation")
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            key = left + "__" + right
            exact_direction_overlap[key] = len(direction_hashes[left] & direction_hashes[right])
            exact_residual_overlap[key] = len(residual_hashes[left] & residual_hashes[right])
            maximum_cosine[key] = float(
                np.max(
                    np.abs(
                        np.asarray(codebooks[left]["directions"]).dot(
                            np.asarray(codebooks[right]["directions"]).T
                        )
                    )
                )
            )
    target_bank_matches = {
        split: len(residual_hashes[split] & bank_hashes) for split in names
    }
    passed = (
        not any(exact_direction_overlap.values())
        and not any(exact_residual_overlap.values())
        and not any(target_bank_matches.values())
        and max(maximum_cosine.values()) <= MAX_CROSS_SPLIT_ABS_COSINE + 1e-12
    )
    return {
        "exact_direction_overlap": exact_direction_overlap,
        "exact_residual_overlap": exact_residual_overlap,
        "maximum_cross_split_absolute_cosine_similarity": maximum_cosine,
        "maximum_allowed": MAX_CROSS_SPLIT_ABS_COSINE,
        "target_residual_matches_action_bank": target_bank_matches,
        "passed": bool(passed),
    }


def _record_from_stage2_freeze(stage2_root):
    records = {}
    replays = {}
    for task in TASKS:
        path = os.path.join(stage2_root, "work", "freeze_tasks", task["task_id"] + ".json")
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not payload["replay"]["passed"]:
            raise RuntimeError("Stage 2 replay validation changed for " + task["task_id"])
        for row in payload["records"]:
            records[(row["task_id"], int(row["episode_id"]), row["phase"])] = row
        for row in payload["replay"]["tests"]:
            replays[(row["task_id"], int(row["episode_id"]), row["phase"])] = row
    return records, replays


def _confirmation_snapshot_records(paths, codebooks, replay_tolerance=1e-12):
    records = []
    replay_tests = []
    for task in TASKS:
        runtime = LiberoTaskRuntime(task, paths["libero_source"], paths["dataset_root"])
        try:
            for episode_id in CONFIRMATION_EPISODES:
                episode = runtime.load_episode(episode_id)
                indices, notes, scan = runtime.select_phase_indices(episode)
                note_by_phase = {row["phase"]: row for row in notes}
                for phase in PHASES:
                    index = int(indices[phase])
                    base_actions = np.asarray(
                        episode["actions"][index : index + config.CHUNK_HORIZON],
                        dtype=np.float64,
                    )
                    base_continuous = runtime.continuous_chunk(base_actions)
                    snapshot = runtime.snapshot_from_recorded_state(
                        episode["states"][index], episode["actions"][:index]
                    )
                    nominal_first = runtime.execute_chunk(snapshot, base_actions)
                    nominal_second = runtime.execute_chunk(snapshot, base_actions)
                    nominal_third = runtime.execute_chunk(snapshot, base_actions)
                    same, same_metrics = _compare_rollouts(
                        nominal_first, nominal_second, replay_tolerance
                    )
                    repeated, repeated_metrics = _compare_rollouts(
                        nominal_first, nominal_third, replay_tolerance
                    )
                    replay_tests.append(
                        {
                            "task_id": task["task_id"],
                            "episode_id": int(episode_id),
                            "phase": phase,
                            "same_action_twice": {
                                "passed": bool(same),
                                "metrics": same_metrics,
                            },
                            "same_action_third_execution": {
                                "passed": bool(repeated),
                                "metrics": repeated_metrics,
                            },
                            "target_or_candidate_branch_executed": False,
                            "passed": bool(same and repeated),
                        }
                    )
                    records.append(
                        {
                            "key": "%s__e%02d__%s"
                            % (task["task_id"], episode_id, phase),
                            "task_id": task["task_id"],
                            "task_name": task["task_name"],
                            "episode_id": int(episode_id),
                            "split": "confirmation",
                            "phase": phase,
                            "snapshot_index": index,
                            "snapshot_state_sha256": _array_hash(episode["states"][index]),
                            "base_action_sha256": _array_hash(base_actions),
                            "base_actions": base_actions.tolist(),
                            "base_continuous": base_continuous.tolist(),
                            "episode_length": int(len(episode["actions"])),
                            "selection_note": note_by_phase[phase],
                            "contact_fraction": float(np.mean(scan["contact"])),
                            "target_contact_fraction": float(np.mean(scan["target_contact"])),
                        }
                    )
        finally:
            runtime.close()
    failures = [row for row in replay_tests if not row["passed"]]
    return records, {
        "tolerance": float(replay_tolerance),
        "n_tests": len(replay_tests),
        "n_failed": len(failures),
        "passed": not failures,
        "failed_tests": failures,
        "tests": replay_tests,
    }


def _episode_inventory(paths):
    tasks = []
    for task in TASKS:
        demo_path = _demo_path(paths, task)
        with h5py.File(demo_path, "r") as handle:
            data = handle["data"]
            episodes = []
            for episode_id in range(50):
                group = data["demo_%d" % episode_id]
                episodes.append(
                    {
                        "episode_id": int(episode_id),
                        "split": next(
                            name
                            for name, values in SPLIT_EPISODES.items()
                            if episode_id in values
                        ),
                        "sha256": _episode_hash(group),
                        "length": int(len(group["actions"])),
                        "successful": bool(np.max(group["rewards"][...]) > 0),
                    }
                )
        if not all(row["successful"] for row in episodes):
            raise RuntimeError("unsuccessful frozen episode for " + task["task_id"])
        tasks.append(
            {
                "task_id": task["task_id"],
                "task_name": task["task_name"],
                "demo_path": demo_path,
                "demo_sha256": sha256_file(demo_path),
                "demo_bytes": int(os.path.getsize(demo_path)),
                "episodes": episodes,
            }
        )
    return tasks


def _save_codebooks(path, codebooks):
    atomic_npz(
        path,
        split_names=np.asarray(("calibration", "development", "confirmation")),
        seeds=np.asarray(
            [codebooks[name]["seed"] for name in ("calibration", "development", "confirmation")],
            dtype=np.uint32,
        ),
        directions=np.stack(
            [codebooks[name]["directions"] for name in ("calibration", "development", "confirmation")]
        ),
        radii=np.stack(
            [codebooks[name]["radii"] for name in ("calibration", "development", "confirmation")]
        ),
        direction_family_id=np.stack(
            [codebooks[name]["family_id"] for name in ("calibration", "development", "confirmation")]
        ),
        direction_family_names=np.asarray(DIRECTION_FAMILIES),
        signs=np.asarray(SIGNS, dtype=np.int8),
    )


def freeze_protocol(project_root, paths, output_root, scratch_root=SCRATCH_ROOT):
    """Freeze every pre-result Stage 3 choice and validation artifact."""
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(scratch_root, exist_ok=True)
    stage2_root = os.path.join(project_root, STAGE2_ROOT_RELATIVE)
    action_bank_path = os.path.join(project_root, STAGE2_ACTION_BANK_RELATIVE)
    with np.load(action_bank_path, allow_pickle=False) as handle:
        action_bank = np.asarray(handle["residuals"], dtype=np.float64)
        action_bank_hashes = handle["residual_sha256"].astype(str)
    if action_bank.shape != (ACTION_BANK_SIZE, CONTINUOUS_DIM):
        raise RuntimeError("Stage 2 action bank shape changed")

    codebooks = generate_support_codebooks()
    separation = support_separation_evidence(codebooks, action_bank)
    if not separation["passed"]:
        raise RuntimeError("BLOCKED_SUPPORT_SEPARATION")

    inventory = _episode_inventory(paths)
    historical_records, historical_replays = _record_from_stage2_freeze(stage2_root)
    confirmation_records, confirmation_replay = _confirmation_snapshot_records(paths, codebooks)
    if not confirmation_replay["passed"]:
        raise RuntimeError("confirmation snapshot replay failed")
    snapshot_records = []
    replay_summary = {
        "stage2_reused_tests": 0,
        "stage2_reused_failed": 0,
        "confirmation_new_tests": int(confirmation_replay["n_tests"]),
        "confirmation_new_failed": int(confirmation_replay["n_failed"]),
    }
    for key in sorted(historical_records):
        row = dict(historical_records[key])
        if int(row["episode_id"]) < 16:
            continue
        row["split"] = split_for_episode(row["episode_id"])
        replay = historical_replays[key]
        replay_summary["stage2_reused_tests"] += 1
        replay_summary["stage2_reused_failed"] += int(not replay["passed"])
        snapshot_records.append(row)
    snapshot_records.extend(confirmation_records)
    snapshot_records.sort(
        key=lambda row: (
            next(i for i, task in enumerate(TASKS) if task["task_id"] == row["task_id"]),
            int(row["episode_id"]),
            PHASES.index(row["phase"]),
        )
    )

    # Bind episode hashes into every snapshot record and validate every fixed
    # target and candidate action without executing a method result.
    episode_hash_map = {
        (task["task_id"], row["episode_id"]): row["sha256"]
        for task in inventory
        for row in task["episodes"]
    }
    invalid_target_actions = []
    invalid_bank_actions = []
    for row in snapshot_records:
        row["episode_sha256"] = episode_hash_map[(row["task_id"], int(row["episode_id"]))]
        base = np.asarray(row["base_continuous"], dtype=np.float64)
        if float(np.max(np.abs(base))) > BASE_ACTION_ABS_LIMIT + 1e-12:
            raise RuntimeError("snapshot nominal action exceeds frozen bound")
        split = row["split"]
        if split in codebooks:
            bank = codebooks[split]
            for direction_id in range(DIRECTION_COUNT):
                for radius in bank["radii"][direction_id]:
                    for sign in SIGNS:
                        residual = float(sign) * float(radius) * bank["directions"][direction_id]
                        if float(np.max(np.abs(base + residual))) > 1.0 + 1e-12:
                            invalid_target_actions.append(row["key"])
        validity = np.max(np.abs(base[None, :] + action_bank), axis=1) <= 1.0 + 1e-12
        if not bool(np.all(validity)):
            invalid_bank_actions.append(
                {"key": row["key"], "valid_bank_size": int(np.sum(validity))}
            )
    if invalid_target_actions or invalid_bank_actions:
        raise RuntimeError("frozen action validity failed")

    codebook_path = os.path.join(output_root, "support_codebooks.npz")
    _save_codebooks(codebook_path, codebooks)
    episode_split_path = os.path.join(output_root, "episode_split.json")
    atomic_json(
        episode_split_path,
        {
            "created_utc": utc_now(),
            "splits": {name: list(values) for name, values in SPLIT_EPISODES.items()},
            "all_episodes_successful": True,
            "tasks": inventory,
            "snapshots": snapshot_records,
            "snapshot_count": len(snapshot_records),
            "replay_validation": replay_summary,
            "confirmation_replay_validation": confirmation_replay,
            "support_separation": separation,
            "invalid_target_actions": invalid_target_actions,
            "invalid_bank_actions": invalid_bank_actions,
        },
    )
    action_binding_path = os.path.join(output_root, "action_bank_binding.json")
    atomic_json(
        action_binding_path,
        {
            "source": STAGE2_ACTION_BANK_RELATIVE,
            "sha256": sha256_file(action_bank_path),
            "shape": list(action_bank.shape),
            "residual_hashes_sha256": hashlib.sha256(
                "\n".join(action_bank_hashes.tolist()).encode("ascii")
            ).hexdigest(),
            "max_absolute_component": float(np.max(np.abs(action_bank))),
            "min_l2_norm": float(np.min(np.linalg.norm(action_bank, axis=1))),
            "max_l2_norm": float(np.max(np.linalg.norm(action_bank, axis=1))),
            "action_semantics": "H=4 flattened 6-D OSC_POSE residual; gripper copied from nominal",
            "all_544_stage3_snapshot_banks_fully_valid": len(invalid_bank_actions) == 0,
            "primary_k": PRIMARY_K,
            "k_sensitivity_locked_until_final_disposition": [32, 128],
        },
    )
    model_path = os.path.join(output_root, "model_definitions.json")
    atomic_json(model_path, model_definitions())
    incident_path = os.path.join(output_root, "PRE_RESULT_PROTOCOL_INCIDENT.json")
    if not os.path.isfile(incident_path):
        raise FileNotFoundError(incident_path)

    required_history = {
        "stage1_disposition": STAGE1_DISPOSITION,
        "stage1_published_commit": STAGE1_PUBLISHED_COMMIT,
        "stage1_5_disposition": STAGE1_5_DISPOSITION,
        "stage1_5_result_commit": STAGE1_5_RESULT_COMMIT,
        "stage2_disposition": STAGE2_DISPOSITION,
        "stage2_published_commit": STAGE3_INPUT_COMMIT,
        "stage2_input_commit": STAGE2_INPUT_COMMIT,
    }
    input_binding = {
        "created_utc": utc_now(),
        "stage3_input_commit": STAGE3_INPUT_COMMIT,
        "historical_evidence": required_history,
        "libero": {
            "upstream_commit": LIBERO_COMMIT,
            "expected_tree_sha256": LIBERO_TREE_SHA256,
            "observed_tree_sha256": sha256_tree(paths["libero_source"]),
            "source_path": paths["libero_source"],
            "dataset_root": paths["dataset_root"],
        },
        "controller": {
            "robot": "Panda",
            "mode": "OSC_POSE",
            "frequency_hz": 20,
            "horizon": 4,
            "settle_steps": 3,
            "gripper": "copied unchanged from nominal demonstration",
        },
        "stage2_reuse": {
            "root": STAGE2_ROOT_RELATIVE,
            "root_tree_sha256": sha256_tree(stage2_root),
            "action_bank_sha256": sha256_file(action_bank_path),
            "support_episode_range": [16, 31],
            "candidate_effects_reused_episode_range": [24, 31],
            "candidate_effects_missing_episode_range": [16, 23],
        },
        "pre_result_artifacts": {
            "episode_split.json": sha256_file(episode_split_path),
            "support_codebooks.npz": sha256_file(codebook_path),
            "action_bank_binding.json": sha256_file(action_binding_path),
            "model_definitions.json": sha256_file(model_path),
            "PRE_RESULT_PROTOCOL_INCIDENT.json": sha256_file(incident_path),
        },
        "support_separation": separation,
        "scratch_root": scratch_root,
        "execution_amendment": EXECUTION_AMENDMENT,
        "confirmation_integrity_amendment": CONFIRMATION_INTEGRITY_AMENDMENT,
        "pai_jobs_submitted_before_freeze": 0,
    }
    if input_binding["libero"]["observed_tree_sha256"] != LIBERO_TREE_SHA256:
        raise RuntimeError("LIBERO source tree changed")
    atomic_json(os.path.join(output_root, "INPUT_BINDING.json"), input_binding)
    return input_binding


def refresh_frozen_metadata(output_root):
    """Refresh code-defined metadata after a pre-result definition refinement.

    This never reads a branch consequence or method result.  It exists so an
    already completed simulator replay need not be repeated merely to bind a
    more explicit model-definition JSON before collection starts.
    """
    required = (
        "episode_split.json",
        "support_codebooks.npz",
        "action_bank_binding.json",
        "PRE_RESULT_PROTOCOL_INCIDENT.json",
        "INPUT_BINDING.json",
    )
    for name in required:
        if not os.path.isfile(os.path.join(output_root, name)):
            raise FileNotFoundError(os.path.join(output_root, name))
    forbidden_results = (
        "training_pairs.parquet",
        "predictor_metrics.csv",
        "retrieval_metrics.csv",
        "development_quantization.csv",
        "confirmation_quantization.csv",
    )
    visible = [name for name in forbidden_results if os.path.exists(os.path.join(output_root, name))]
    if visible:
        raise RuntimeError("cannot refresh pre-result definitions after method results: %s" % visible)
    model_path = os.path.join(output_root, "model_definitions.json")
    atomic_json(model_path, model_definitions())
    binding_path = os.path.join(output_root, "INPUT_BINDING.json")
    with open(binding_path, "r", encoding="utf-8") as handle:
        binding = json.load(handle)
    binding["pre_result_metadata_refresh_utc"] = utc_now()
    binding["pre_result_metadata_refresh_before_method_results"] = True
    binding["pre_result_artifacts"]["model_definitions.json"] = sha256_file(model_path)
    binding["confirmation_integrity_amendment"] = CONFIRMATION_INTEGRITY_AMENDMENT
    atomic_json(binding_path, binding)
    return {
        "input_binding": binding_path,
        "model_definitions": model_path,
        "model_definitions_sha256": sha256_file(model_path),
        "method_results_visible": visible,
    }
