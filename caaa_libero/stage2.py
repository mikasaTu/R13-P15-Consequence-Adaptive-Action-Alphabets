"""Stage 2 fresh-support nonlinear consequence-atlas audit.

Simulator-facing functions remain compatible with the frozen LIBERO Python 3.8
environment.  Predictor training and Parquet/Zarr export run in the analysis
environment.  No function in this module trains a policy.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess

import h5py
import numpy as np

from . import config
from .env_adapter import FEATURE_NAMES, LiberoTaskRuntime
from .pipeline import CONTACT_MODE_TO_ID, _compare_rollouts, utc_now
from .stage2_config import (
    ACTION_BANK_SIZE,
    ALL_FRESH_EPISODES,
    CALIBRATION_EPISODES,
    CONFIRMATION_EPISODES,
    CONTINUOUS_DIM,
    DEVELOPMENT_EPISODES,
    DIRECTION_COUNT,
    DIRECTION_FAMILIES,
    DIRECTION_FAMILY_COUNTS,
    GLOBAL_SEED,
    HISTORICAL_EPISODES,
    MIN_VALID_BANK,
    PHASES,
    RADII_PER_DIRECTION,
    RADIUS_INTERVAL,
    SIGNS,
    SPLIT_EPISODES,
    TASKS,
    TRAIN_EPISODES,
    consequence_metric_definition,
    method_definitions,
    split_for_episode,
)
from .storage import (
    atomic_json,
    atomic_npz,
    mark_complete,
    sha256_file,
    sha256_tree,
    validate_complete,
)


INPUT_COMMIT = "154d4a89e071d94208f5302955c55c13e3cff7f3"
INPUT_TREE = "4199a6280cfb8f5e43b04547291fa792d132b725"
STAGE1_FORMAL_COMMIT = "34995e8e7c3069b22785ad04536f0d429e75c0fc"
STAGE1_PUBLISHED_COMMIT = "434427af0f8adc844851c27cfc050b2c9c6752dc"
STAGE1_DISPOSITION = "REJECT_CORE_HYPOTHESIS"
STAGE1_TREE_SHA256 = "047aae35193339a460cd1dbac0e4495d7f9cff4a1cb2799c58b738e86e0e4c5c"
STAGE1_5_PREREG_COMMIT = "9a3ac1a4c774103fe618bd283909c2793ed581ec"
STAGE1_5_FROZEN_METHOD_COMMIT = "aa82d46c5e0828956aef15918c2aa7656844472f"
STAGE1_5_RESULT_COMMIT = "76433b6e58196ceeedc4ad005a1110ea8e343ae2"
STAGE1_5_DISPOSITION = "REJECT_P15_FAMILY"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
LIBERO_TREE_SHA256 = "e9197ca08fe4d7325f561fc40d7425167830253e0f0fceb1af2663b23292f71f"

STAGE1_REQUIRED = (
    "PREREGISTRATION.md",
    "environment_lock.json",
    "task_and_seed_split.json",
    "branch_replay_validation.json",
    "consequence_schema.json",
    "branch_rollouts.zarr",
    "jacobian_metrics.parquet",
    "alphabet_codebooks",
    "results_by_task.csv",
    "results_by_phase.csv",
    "bootstrap_results.json",
    "mechanism_controls.csv",
    "STAGE1_REPORT.md",
)
STAGE1_5_REQUIRED = (
    "PREREGISTRATION.md",
    "STAGE1_INPUT_BINDING.json",
    "retrospective_diagnostics.parquet",
    "error_decomposition.csv",
    "fresh_holdout_split.json",
    "fresh_branch_rollouts.zarr",
    "method_definitions.json",
    "quantization_results_by_task.csv",
    "quantization_results_by_phase.csv",
    "mechanism_controls.csv",
    "bootstrap_results.json",
    "STAGE1_5_REPORT.md",
)


def _run(command, cwd=None):
    try:
        return subprocess.check_output(
            command, cwd=cwd, stderr=subprocess.STDOUT, universal_newlines=True
        ).strip()
    except Exception as error:
        return "ERROR:%s" % (error,)


def _stable_seed(*parts):
    value = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little", signed=False)


def _array_hash(value):
    array = np.asarray(value)
    header = "%s|%s" % (array.dtype.str, ",".join(str(x) for x in array.shape))
    digest = hashlib.sha256(header.encode("ascii") + b"\0")
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _path_hash(path):
    return sha256_tree(path) if os.path.isdir(path) else sha256_file(path)


def _task(task_id):
    for task in TASKS:
        if task["task_id"] == task_id:
            return dict(task)
    raise KeyError(task_id)


def _demo_path(paths, task):
    return os.path.join(
        paths["dataset_root"], config.SUITE, task["task_name"] + "_demo.hdf5"
    )


def _episode_hash(group):
    digest = hashlib.sha256()
    for name in ("actions", "states", "rewards", "dones"):
        value = np.asarray(group[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(np.ascontiguousarray(value).tobytes())
    for name in ("model_file", "num_samples"):
        value = group.attrs.get(name)
        if isinstance(value, bytes):
            encoded = value
        else:
            encoded = str(value).encode("utf-8")
        digest.update(name.encode("ascii") + b"\0" + encoded + b"\0")
    return digest.hexdigest()


def _dct_basis(length=4):
    basis = np.zeros((length, length), dtype=np.float64)
    for k in range(length):
        scale = math.sqrt(1.0 / length) if k == 0 else math.sqrt(2.0 / length)
        for t in range(length):
            basis[k, t] = scale * math.cos(math.pi * (t + 0.5) * k / length)
    return basis


def _normalize_direction(value, family):
    value = np.asarray(value, dtype=np.float64).reshape(4, 6)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("zero direction for %s" % family)
    value = value / norm
    # The snapshot selector guarantees |a| <= .895.  This bound guarantees
    # every radius <= .12 stays strictly inside normalized action bounds.
    if float(np.max(np.abs(value))) > 0.82:
        raise ValueError("coordinate-concentrated %s direction" % family)
    return value.reshape(-1)


def generate_direction_bank(task_id, episode_id, phase, split):
    seed = _stable_seed(GLOBAL_SEED, "directions", task_id, episode_id, phase, split)
    rng = np.random.RandomState(seed)
    dct = _dct_basis(4)
    directions = []
    families = []

    while len([x for x in families if x == "smooth_dct"]) < DIRECTION_FAMILY_COUNTS["smooth_dct"]:
        coefficients = rng.normal(size=4) / np.asarray([1.0, 1.5, 3.0, 6.0])
        temporal = coefficients.dot(dct)
        action = rng.normal(size=6)
        secondary = (rng.normal(size=4) / np.asarray([2.0, 3.0, 6.0, 12.0])).dot(dct)
        value = np.outer(temporal, action) + 0.20 * np.outer(secondary, rng.normal(size=6))
        try:
            directions.append(_normalize_direction(value, "smooth_dct"))
            families.append("smooth_dct")
        except ValueError:
            continue

    suffix_count = 0
    while suffix_count < DIRECTION_FAMILY_COUNTS["suffix_contact"]:
        start = 1 + (suffix_count % 2)
        temporal = np.zeros(4, dtype=np.float64)
        temporal[start:] = np.linspace(0.45, 1.0, 4 - start)
        temporal *= float(rng.choice([-1.0, 1.0]))
        action = rng.normal(size=6)
        action /= max(float(np.linalg.norm(action)), 1e-12)
        value = np.outer(temporal, action)
        try:
            directions.append(_normalize_direction(value, "suffix_contact"))
            families.append("suffix_contact")
            suffix_count += 1
        except ValueError:
            continue

    low_rank_count = 0
    while low_rank_count < DIRECTION_FAMILY_COUNTS["low_rank_temporal_action"]:
        value = np.zeros((4, 6), dtype=np.float64)
        for _ in range(2):
            value += np.outer(rng.normal(size=4), rng.normal(size=6))
        try:
            directions.append(_normalize_direction(value, "low_rank_temporal_action"))
            families.append("low_rank_temporal_action")
            low_rank_count += 1
        except ValueError:
            continue

    directions = np.stack(directions).astype(np.float64)
    if directions.shape != (DIRECTION_COUNT, CONTINUOUS_DIM):
        raise AssertionError(directions.shape)
    radii = np.zeros((DIRECTION_COUNT, RADII_PER_DIRECTION), dtype=np.float64)
    for direction_id in range(DIRECTION_COUNT):
        radius_rng = np.random.RandomState(
            _stable_seed(seed, "radii", direction_id, families[direction_id])
        )
        radii[direction_id] = np.sort(
            radius_rng.uniform(RADIUS_INTERVAL[0], RADIUS_INTERVAL[1], size=2)
        )
    return {
        "seed": int(seed),
        "directions": directions,
        "families": tuple(families),
        "radii": radii,
    }


def _fresh_record_key(task_id, episode_id, phase):
    return "%s__e%02d__%s" % (task_id, int(episode_id), phase)


def freeze_task(paths, output_root, task_id, replay_tolerance=1e-12):
    """Freeze demo hashes, phase snapshots, and replay evidence for one task."""
    task = _task(task_id)
    demo_path = _demo_path(paths, task)
    if not os.path.isfile(demo_path):
        raise FileNotFoundError(demo_path)
    with h5py.File(demo_path, "r") as handle:
        data = handle["data"]
        num_demos = int(data.attrs["num_demos"])
        episode_meta = {}
        for episode_id in ALL_FRESH_EPISODES:
            key = "demo_%d" % episode_id
            if key not in data:
                raise RuntimeError("BLOCKED_INSUFFICIENT_FRESH_DEMOS:%s:%d" % (task_id, episode_id))
            group = data[key]
            success = bool(np.max(group["rewards"][...]) > 0)
            if not success:
                raise RuntimeError("BLOCKED_INSUFFICIENT_FRESH_DEMOS:%s:%d" % (task_id, episode_id))
            episode_meta[int(episode_id)] = {
                "episode_id": int(episode_id),
                "sha256": _episode_hash(group),
                "length": int(len(group["actions"])),
                "successful": True,
            }
    demo_sha256 = sha256_file(demo_path)

    runtime = LiberoTaskRuntime(task, paths["libero_source"], paths["dataset_root"])
    records = []
    replay_tests = []
    try:
        for episode_id in ALL_FRESH_EPISODES:
            episode = runtime.load_episode(episode_id)
            indices, notes, scan = runtime.select_phase_indices(episode)
            note_by_phase = {row["phase"]: row for row in notes}
            split = split_for_episode(episode_id)
            for phase in PHASES:
                index = int(indices[phase])
                base_actions = np.asarray(
                    episode["actions"][index : index + config.CHUNK_HORIZON], dtype=np.float64
                )
                base_continuous = runtime.continuous_chunk(base_actions)
                snapshot = runtime.snapshot_from_recorded_state(
                    episode["states"][index], episode["actions"][:index]
                )
                bank = generate_direction_bank(task_id, episode_id, phase, split)
                radius = float(bank["radii"][0, 0])
                delta = radius * bank["directions"][0]
                pert_continuous = base_continuous + delta
                if float(np.max(np.abs(pert_continuous))) > 1.0 + 1e-12:
                    raise RuntimeError("frozen support crosses action bound")
                pert_actions = runtime.replace_continuous_chunk(base_actions, pert_continuous)
                a_first = runtime.execute_chunk(snapshot, base_actions)
                a_second = runtime.execute_chunk(snapshot, base_actions)
                b_first = runtime.execute_chunk(snapshot, pert_actions)
                a_after_b = runtime.execute_chunk(snapshot, base_actions)
                b_after_a = runtime.execute_chunk(snapshot, pert_actions)
                same_a, same_a_metrics = _compare_rollouts(a_first, a_second, replay_tolerance)
                order_a, order_a_metrics = _compare_rollouts(a_first, a_after_b, replay_tolerance)
                order_b, order_b_metrics = _compare_rollouts(b_first, b_after_a, replay_tolerance)
                replay_tests.append(
                    {
                        "task_id": task_id,
                        "episode_id": int(episode_id),
                        "phase": phase,
                        "snapshot_index": index,
                        "same_action_twice": {"passed": bool(same_a), "metrics": same_a_metrics},
                        "a_after_b": {"passed": bool(order_a), "metrics": order_a_metrics},
                        "b_after_a": {"passed": bool(order_b), "metrics": order_b_metrics},
                        "passed": bool(same_a and order_a and order_b),
                    }
                )
                records.append(
                    {
                        "key": _fresh_record_key(task_id, episode_id, phase),
                        "task_id": task_id,
                        "task_name": task["task_name"],
                        "episode_id": int(episode_id),
                        "split": split,
                        "phase": phase,
                        "snapshot_index": index,
                        "snapshot_state_sha256": _array_hash(episode["states"][index]),
                        "base_action_sha256": _array_hash(base_actions),
                        "base_actions": base_actions.tolist(),
                        "base_continuous": base_continuous.tolist(),
                        "episode_length": int(len(episode["actions"])),
                        "episode_sha256": episode_meta[int(episode_id)]["sha256"],
                        "recorded_success": True,
                        "selection_note": note_by_phase[phase],
                        "contact_fraction": float(np.mean(scan["contact"])),
                        "target_contact_fraction": float(np.mean(scan["target_contact"])),
                    }
                )
    finally:
        runtime.close()

    failures = [row for row in replay_tests if not row["passed"]]
    result = {
        "created_utc": utc_now(),
        "task": task,
        "demo": {
            "path": demo_path,
            "bytes": int(os.path.getsize(demo_path)),
            "sha256": demo_sha256,
            "num_demos": int(num_demos),
            "successful_fresh_ids": list(ALL_FRESH_EPISODES),
            "episodes": [episode_meta[i] for i in ALL_FRESH_EPISODES],
        },
        "records": records,
        "replay": {
            "tolerance": float(replay_tolerance),
            "tests": replay_tests,
            "n_tests": len(replay_tests),
            "n_failed": len(failures),
            "failed_tests": failures,
            "passed": not failures,
        },
    }
    path = os.path.join(output_root, "work", "freeze_tasks", task_id + ".json")
    atomic_json(path, result)
    return result


def _load_freeze_tasks(output_root):
    rows = []
    for task in TASKS:
        path = os.path.join(output_root, "work", "freeze_tasks", task["task_id"] + ".json")
        with open(path, "r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    return rows


def _generate_perturbation_artifact(records, output_root):
    task_ids, episode_ids, phases, splits, seeds = [], [], [], [], []
    directions, radii, family_ids = [], [], []
    family_to_id = {name: index for index, name in enumerate(DIRECTION_FAMILIES)}
    for record in sorted(records, key=lambda row: (row["task_id"], row["episode_id"], PHASES.index(row["phase"]))):
        bank = generate_direction_bank(
            record["task_id"], record["episode_id"], record["phase"], record["split"]
        )
        task_ids.append(record["task_id"])
        episode_ids.append(record["episode_id"])
        phases.append(record["phase"])
        splits.append(record["split"])
        seeds.append(bank["seed"])
        directions.append(bank["directions"])
        radii.append(bank["radii"])
        family_ids.append([family_to_id[name] for name in bank["families"]])
    path = os.path.join(output_root, "perturbation_banks.npz")
    atomic_npz(
        path,
        task_id=np.asarray(task_ids),
        episode_id=np.asarray(episode_ids, dtype=np.int16),
        phase=np.asarray(phases),
        split=np.asarray(splits),
        seed=np.asarray(seeds, dtype=np.uint32),
        directions=np.asarray(directions, dtype=np.float64),
        radii=np.asarray(radii, dtype=np.float64),
        direction_family_id=np.asarray(family_ids, dtype=np.int8),
        direction_family_names=np.asarray(DIRECTION_FAMILIES),
        signs=np.asarray(SIGNS, dtype=np.int8),
    )
    return path


def _support_overlap_checks(perturbation_path):
    with np.load(perturbation_path, allow_pickle=False) as data:
        split = data["split"].astype(str)
        directions = np.asarray(data["directions"], dtype=np.float64)
        radii = np.asarray(data["radii"], dtype=np.float64)
    direction_hashes = {}
    residual_hashes = {}
    flattened = {}
    for split_name in SPLIT_EPISODES:
        keep = np.flatnonzero(split == split_name)
        dirs = directions[keep].reshape(-1, CONTINUOUS_DIM)
        rads = radii[keep].reshape(-1, RADII_PER_DIRECTION)
        direction_hashes[split_name] = {_array_hash(row) for row in dirs}
        residual = []
        for row, row_radii in zip(dirs, rads):
            for radius in row_radii:
                for sign in SIGNS:
                    residual.append(float(sign) * float(radius) * row)
        residual = np.asarray(residual, dtype=np.float64)
        residual_hashes[split_name] = {_array_hash(row) for row in residual}
        flattened[split_name] = dirs
    exact_direction_overlap = {}
    exact_residual_overlap = {}
    max_cosine = {}
    names = list(SPLIT_EPISODES)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            label = left + "__" + right
            exact_direction_overlap[label] = len(direction_hashes[left] & direction_hashes[right])
            exact_residual_overlap[label] = len(residual_hashes[left] & residual_hashes[right])
            a, b = flattened[left], flattened[right]
            maximum = 0.0
            for start in range(0, len(a), 256):
                maximum = max(maximum, float(np.max(np.abs(a[start : start + 256].dot(b.T)))))
            max_cosine[label] = maximum
    return {
        "exact_direction_overlap": exact_direction_overlap,
        "exact_residual_overlap": exact_residual_overlap,
        "maximum_cross_split_absolute_cosine_similarity": max_cosine,
        "passed": not any(exact_direction_overlap.values()) and not any(exact_residual_overlap.values()),
    }


def _fps_indices(values, k, seed):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < int(k):
        raise ValueError("insufficient FPS candidates")
    first = int(seed) % len(values)
    chosen = [first]
    minimum = np.sum((values - values[first]) ** 2, axis=1)
    minimum[first] = -1.0
    while len(chosen) < int(k):
        index = int(np.argmax(minimum))
        chosen.append(index)
        distance = np.sum((values - values[index]) ** 2, axis=1)
        minimum = np.minimum(minimum, distance)
        minimum[np.asarray(chosen, dtype=np.int64)] = -1.0
    return np.asarray(chosen, dtype=np.int64)


def _build_action_bank(records, perturbation_path, output_root):
    with np.load(perturbation_path, allow_pickle=False) as data:
        tasks = data["task_id"].astype(str)
        episodes = data["episode_id"].astype(int)
        phases = data["phase"].astype(str)
        splits = data["split"].astype(str)
        directions = np.asarray(data["directions"], dtype=np.float64)
        radii = np.asarray(data["radii"], dtype=np.float64)
        family_ids = np.asarray(data["direction_family_id"], dtype=np.int8)
    train_bases = np.asarray(
        [row["base_continuous"] for row in records if row["split"] == "train"], dtype=np.float64
    )
    candidates = []
    for state_index in np.flatnonzero(splits == "train"):
        for direction_id in range(DIRECTION_COUNT):
            for radius_id in range(RADII_PER_DIRECTION):
                for sign in SIGNS:
                    residual = float(sign) * radii[state_index, direction_id, radius_id] * directions[state_index, direction_id]
                    globally_valid = bool(np.max(np.abs(train_bases + residual[None, :])) <= 1.0 + 1e-12)
                    if not globally_valid:
                        continue
                    candidates.append(
                        {
                            "residual": residual,
                            "task_id": tasks[state_index],
                            "episode_id": int(episodes[state_index]),
                            "phase": phases[state_index],
                            "family_id": int(family_ids[state_index, direction_id]),
                            "direction_id": int(direction_id),
                            "radius_id": int(radius_id),
                            "radius": float(radii[state_index, direction_id, radius_id]),
                            "sign": int(sign),
                        }
                    )
    strata = []
    for task in TASKS:
        for phase in PHASES:
            for family_id in range(len(DIRECTION_FAMILIES)):
                strata.append((task["task_id"], phase, family_id))
    base_quota = ACTION_BANK_SIZE // len(strata)
    extra = ACTION_BANK_SIZE % len(strata)
    selected = []
    selected_hashes = set()
    for stratum_index, stratum in enumerate(strata):
        quota = base_quota + (1 if stratum_index < extra else 0)
        pool = [
            row
            for row in candidates
            if (row["task_id"], row["phase"], row["family_id"]) == stratum
            and _array_hash(row["residual"]) not in selected_hashes
        ]
        # Stable hash order makes ties and source metadata deterministic.
        pool.sort(key=lambda row: _array_hash(row["residual"]))
        values = np.asarray([row["residual"] for row in pool], dtype=np.float64)
        indices = _fps_indices(values, quota, _stable_seed(GLOBAL_SEED, "bank", *stratum))
        for index in indices:
            row = pool[int(index)]
            selected.append(row)
            selected_hashes.add(_array_hash(row["residual"]))
    if len(selected) != ACTION_BANK_SIZE or len(selected_hashes) != ACTION_BANK_SIZE:
        raise RuntimeError("failed to build unique balanced M=256 bank")
    residuals = np.asarray([row["residual"] for row in selected], dtype=np.float64)
    path = os.path.join(output_root, "action_bank.npz")
    atomic_npz(
        path,
        residuals=residuals,
        residual_sha256=np.asarray([_array_hash(row) for row in residuals]),
        source_task_id=np.asarray([row["task_id"] for row in selected]),
        source_episode_id=np.asarray([row["episode_id"] for row in selected], dtype=np.int16),
        source_phase=np.asarray([row["phase"] for row in selected]),
        source_family_id=np.asarray([row["family_id"] for row in selected], dtype=np.int8),
        source_direction_id=np.asarray([row["direction_id"] for row in selected], dtype=np.int8),
        source_radius_id=np.asarray([row["radius_id"] for row in selected], dtype=np.int8),
        source_radius=np.asarray([row["radius"] for row in selected], dtype=np.float64),
        source_sign=np.asarray([row["sign"] for row in selected], dtype=np.int8),
    )
    return path


def _bank_validity(records, action_bank_path, perturbation_path):
    with np.load(action_bank_path, allow_pickle=False) as data:
        bank = np.asarray(data["residuals"], dtype=np.float64)
        bank_hashes = set(data["residual_sha256"].astype(str).tolist())
    with np.load(perturbation_path, allow_pickle=False) as data:
        tasks = data["task_id"].astype(str)
        episodes = data["episode_id"].astype(int)
        phases = data["phase"].astype(str)
        splits = data["split"].astype(str)
        directions = np.asarray(data["directions"], dtype=np.float64)
        radii = np.asarray(data["radii"], dtype=np.float64)
    record_map = {row["key"]: row for row in records}
    rows = []
    target_exact_matches = []
    for index in range(len(tasks)):
        key = _fresh_record_key(tasks[index], episodes[index], phases[index])
        base = np.asarray(record_map[key]["base_continuous"], dtype=np.float64)
        valid = np.max(np.abs(base[None, :] + bank), axis=1) <= 1.0 + 1e-12
        rows.append(
            {
                "key": key,
                "split": splits[index],
                "valid_bank_size": int(np.sum(valid)),
                "passed": int(np.sum(valid)) >= MIN_VALID_BANK,
            }
        )
        if splits[index] in ("development", "confirmation"):
            for direction_id in range(DIRECTION_COUNT):
                for radius in radii[index, direction_id]:
                    for sign in SIGNS:
                        residual_hash = _array_hash(float(sign) * float(radius) * directions[index, direction_id])
                        if residual_hash in bank_hashes:
                            target_exact_matches.append(
                                {"key": key, "direction_id": direction_id, "radius": float(radius), "sign": int(sign)}
                            )
    return {
        "minimum_valid_bank_size": min(row["valid_bank_size"] for row in rows),
        "maximum_valid_bank_size": max(row["valid_bank_size"] for row in rows),
        "states_below_128": [row for row in rows if not row["passed"]],
        "target_residual_exact_action_bank_matches": target_exact_matches,
        "rows": rows,
        "passed": not any(not row["passed"] for row in rows) and not target_exact_matches,
    }


def _required_hashes(root, names):
    output = {}
    for name in names:
        path = os.path.join(root, name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        output[name] = _path_hash(path)
    return output


def _input_binding(project_root, output_root, paths):
    stage1_root = os.path.join(project_root, "experiments", "r13_p15_caaa_v2", "stage1")
    stage1_5_root = os.path.join(project_root, "experiments", "r13_p15_caaa_v2", "stage1_5")
    observed_stage1_tree = sha256_tree(stage1_root)
    observed_stage1_5_tree = sha256_tree(stage1_5_root)
    git_identity = subprocess.call(
        [
            "git",
            "-C",
            project_root,
            "diff",
            "--quiet",
            INPUT_COMMIT,
            "--",
            "experiments/r13_p15_caaa_v2/stage1",
            "experiments/r13_p15_caaa_v2/stage1_5",
        ]
    ) == 0
    observed_libero_tree = sha256_tree(paths["libero_source"])
    binding = {
        "created_utc": utc_now(),
        "repository_input": {
            "commit": INPUT_COMMIT,
            "tree": INPUT_TREE,
            "observed_head_before_stage2_result": _run(["git", "rev-parse", "HEAD"], cwd=project_root),
            "branch": "r13-p15-stage2-nonlinear-consequence-atlas",
        },
        "historical_evidence": {
            "git_paths_byte_identical_to_input_commit": bool(git_identity),
            "stage1": {
                "formal_commit": STAGE1_FORMAL_COMMIT,
                "published_commit": STAGE1_PUBLISHED_COMMIT,
                "disposition": STAGE1_DISPOSITION,
                "expected_full_tree_sha256": STAGE1_TREE_SHA256,
                "observed_full_tree_sha256": observed_stage1_tree,
                "required_artifact_hashes": _required_hashes(stage1_root, STAGE1_REQUIRED),
                "release_verifier_sha256": sha256_file(
                    os.path.join(project_root, "provenance", "release_verification.json")
                ),
            },
            "stage1_5": {
                "preregistration_commit": STAGE1_5_PREREG_COMMIT,
                "frozen_method_commit": STAGE1_5_FROZEN_METHOD_COMMIT,
                "result_commit": STAGE1_5_RESULT_COMMIT,
                "disposition": STAGE1_5_DISPOSITION,
                "observed_full_tree_sha256": observed_stage1_5_tree,
                "required_artifact_hashes": _required_hashes(stage1_5_root, STAGE1_5_REQUIRED),
                "release_verifier_sha256": sha256_file(
                    os.path.join(project_root, "provenance", "stage1_5_release_verification.json")
                ),
            },
        },
        "simulator": {
            "libero_upstream_commit": LIBERO_COMMIT,
            "libero_source_path": paths["libero_source"],
            "expected_libero_tree_sha256": LIBERO_TREE_SHA256,
            "observed_libero_tree_sha256": observed_libero_tree,
            "git_metadata_status": "UNAVAILABLE_MIGRATED_SUBMODULE_GITDIR",
            "environment_lock_path": "experiments/r13_p15_caaa_v2/stage1/environment_lock.json",
            "environment_lock_sha256": sha256_file(
                os.path.join(stage1_root, "environment_lock.json")
            ),
            "controller": {
                "robot": "Panda",
                "mode": config.CONTROL_MODE,
                "frequency_hz": config.CONTROL_FREQUENCY_HZ,
                "action_dim": config.ACTION_DIM,
                "continuous_chunk_dim": config.CHUNK_CONTINUOUS_DIM,
                "horizon": config.CHUNK_HORIZON,
                "settle_steps": config.SETTLE_STEPS,
                "gripper": "unchanged nominal demonstration command",
            },
        },
        "hard_scope": {
            "historical_episode_ids_excluded": list(HISTORICAL_EPISODES),
            "cpu_simulation": True,
            "predictor_gpu_max": 1,
            "pai_jobs_submitted": 0,
            "policy_training": False,
            "k_primary_only": 64,
        },
    }
    if observed_stage1_tree != STAGE1_TREE_SHA256:
        raise RuntimeError("Stage 1 tree changed")
    if observed_libero_tree != LIBERO_TREE_SHA256:
        raise RuntimeError("LIBERO source tree changed")
    if not git_identity:
        raise RuntimeError("old evidence path changed")
    atomic_json(os.path.join(output_root, "INPUT_BINDING.json"), binding)
    return binding


def finalize_freeze(project_root, paths, output_root):
    task_results = _load_freeze_tasks(output_root)
    if any(not row["replay"]["passed"] for row in task_results):
        raise RuntimeError("BLOCKED_NONDETERMINISTIC_BRANCHING")
    records = [record for row in task_results for record in row["records"]]
    if len(records) != len(TASKS) * len(ALL_FRESH_EPISODES) * len(PHASES):
        raise RuntimeError("incomplete fresh record inventory")
    inventory = {
        "created_utc": utc_now(),
        "selection_rule": "preferred IDs 16-39; smallest ascending unused successful replacement only if invalid",
        "replacement_used": False,
        "historical_ids_excluded": list(HISTORICAL_EPISODES),
        "tasks": [
            {
                "task": row["task"],
                "demo": row["demo"],
                "replay": {
                    "n_tests": row["replay"]["n_tests"],
                    "n_failed": row["replay"]["n_failed"],
                    "passed": row["replay"]["passed"],
                },
            }
            for row in task_results
        ],
        "all_selected_demos_successful": True,
        "episode_count_per_task": len(ALL_FRESH_EPISODES),
    }
    atomic_json(os.path.join(output_root, "fresh_episode_inventory.json"), inventory)
    development = {
        "created_utc": utc_now(),
        "episode_split": {
            "train": list(TRAIN_EPISODES),
            "calibration": list(CALIBRATION_EPISODES),
            "development": list(DEVELOPMENT_EPISODES),
        },
        "records": [row for row in records if row["split"] != "confirmation"],
    }
    confirmation = {
        "created_utc": utc_now(),
        "locked_until_development_gate": True,
        "episode_split": {"confirmation": list(CONFIRMATION_EPISODES)},
        "records": [row for row in records if row["split"] == "confirmation"],
        "result_accessed": False,
    }
    atomic_json(os.path.join(output_root, "development_split.json"), development)
    atomic_json(os.path.join(output_root, "confirmation_split.json"), confirmation)
    perturbation_path = _generate_perturbation_artifact(records, output_root)
    overlap = _support_overlap_checks(perturbation_path)
    if not overlap["passed"]:
        raise RuntimeError("split-specific support overlap")
    action_bank_path = _build_action_bank(records, perturbation_path, output_root)
    bank_validity = _bank_validity(records, action_bank_path, perturbation_path)
    if not bank_validity["passed"]:
        raise RuntimeError("action bank validity gate failed")
    metrics = consequence_metric_definition()
    metrics["methods"] = method_definitions()
    atomic_json(os.path.join(output_root, "consequence_metrics.json"), metrics)
    atomic_json(os.path.join(output_root, "work", "method_definitions.json"), method_definitions())
    binding = _input_binding(project_root, output_root, paths)
    replay_failures = [
        test
        for row in task_results
        for test in row["replay"]["failed_tests"]
    ]
    validation = {
        "created_utc": utc_now(),
        "all_fresh_demos_successful": True,
        "snapshot_count": len(records),
        "replay_test_count": sum(row["replay"]["n_tests"] for row in task_results),
        "replay_failure_count": len(replay_failures),
        "replay_failures": replay_failures,
        "support_overlap": overlap,
        "action_bank_validity": bank_validity,
        "input_binding_sha256": sha256_file(os.path.join(output_root, "INPUT_BINDING.json")),
        "passed": not replay_failures and overlap["passed"] and bank_validity["passed"],
    }
    atomic_json(os.path.join(output_root, "work", "freeze_validation.json"), validation)
    return {
        "inventory": inventory,
        "development": development,
        "confirmation": confirmation,
        "binding": binding,
        "validation": validation,
    }


def _load_all_records(output_root):
    records = []
    for name in ("development_split.json", "confirmation_split.json"):
        with open(os.path.join(output_root, name), "r", encoding="utf-8") as handle:
            records.extend(json.load(handle)["records"])
    return records


def _load_perturbation_map(output_root):
    path = os.path.join(output_root, "perturbation_banks.npz")
    output = {}
    with np.load(path, allow_pickle=False) as data:
        for index in range(len(data["task_id"])):
            key = _fresh_record_key(
                str(data["task_id"][index]), int(data["episode_id"][index]), str(data["phase"][index])
            )
            output[key] = {
                "directions": np.asarray(data["directions"][index], dtype=np.float64),
                "radii": np.asarray(data["radii"][index], dtype=np.float64),
                "family_id": np.asarray(data["direction_family_id"][index], dtype=np.int8),
            }
    return output


def _pack_rollouts(rollouts):
    return {
        "initial": np.stack([row["initial"]["vector"] for row in rollouts]),
        "immediate": np.stack([row["immediate"]["vector"] for row in rollouts]),
        "settled": np.stack([row["settled"]["vector"] for row in rollouts]),
        "mask": np.stack([row["settled"]["mask"] for row in rollouts]),
        "initial_success": np.asarray([row["initial"]["success"] for row in rollouts], dtype=np.uint8),
        "settled_success": np.asarray([row["settled"]["success"] for row in rollouts], dtype=np.uint8),
        "initial_progress": np.asarray([row["initial"]["progress"] for row in rollouts], dtype=np.float64),
        "settled_progress": np.asarray([row["settled"]["progress"] for row in rollouts], dtype=np.float64),
        "contact_mode": np.asarray([CONTACT_MODE_TO_ID[row["contact_mode"]] for row in rollouts], dtype=np.int8),
        "contact_sequence": np.asarray([row["contact_sequence"] for row in rollouts], dtype=np.uint8),
        "final_state": np.stack([row["final_state"] for row in rollouts]),
    }


def _support_shard(output_root, split, task_id, episode_id, phase):
    return os.path.join(
        output_root,
        "work",
        "support_shards",
        split,
        task_id,
        "%s__e%02d__%s.npz" % (task_id, int(episode_id), phase),
    )


def _candidate_shard(output_root, split, task_id, episode_id, phase):
    return os.path.join(
        output_root,
        "work",
        "candidate_shards",
        split,
        task_id,
        "%s__e%02d__%s.npz" % (task_id, int(episode_id), phase),
    )


def collect_support_task(paths, output_root, task_id, splits):
    requested = set(splits)
    if "confirmation" in requested:
        with open(os.path.join(output_root, "work", "development_gate.json"), "r", encoding="utf-8") as handle:
            gate = json.load(handle)
        if not gate.get("confirmation_unlocked"):
            raise RuntimeError("confirmation is locked")
    task = _task(task_id)
    records = [
        row
        for row in _load_all_records(output_root)
        if row["task_id"] == task_id and row["split"] in requested
    ]
    perturbations = _load_perturbation_map(output_root)
    runtime = LiberoTaskRuntime(task, paths["libero_source"], paths["dataset_root"])
    completed = []
    current_episode = None
    episode = None
    try:
        for record in sorted(records, key=lambda row: (row["episode_id"], PHASES.index(row["phase"]))):
            shard = _support_shard(
                output_root, record["split"], task_id, record["episode_id"], record["phase"]
            )
            valid, evidence = validate_complete(shard)
            if valid:
                completed.append({"path": shard, "status": "resumed", "evidence": evidence})
                continue
            if current_episode != record["episode_id"]:
                episode = runtime.load_episode(record["episode_id"])
                runtime.initialize_episode_model(episode)
                current_episode = record["episode_id"]
            index = int(record["snapshot_index"])
            base_actions = np.asarray(episode["actions"][index : index + config.CHUNK_HORIZON], dtype=np.float64)
            base_continuous = runtime.continuous_chunk(base_actions)
            snapshot = runtime.snapshot_from_recorded_state(
                episode["states"][index], episode["actions"][:index]
            )
            bank = perturbations[record["key"]]
            rollouts = [runtime.execute_chunk(snapshot, base_actions)]
            residuals = [np.zeros(CONTINUOUS_DIM, dtype=np.float64)]
            full_actions = [base_actions]
            direction_ids = [-1]
            family_ids = [-1]
            radius_ids = [-1]
            radii = [0.0]
            signs = [0]
            for direction_id in range(DIRECTION_COUNT):
                for radius_id in range(RADII_PER_DIRECTION):
                    radius = float(bank["radii"][direction_id, radius_id])
                    for sign in SIGNS:
                        residual = float(sign) * radius * bank["directions"][direction_id]
                        continuous = base_continuous + residual
                        if np.max(np.abs(continuous)) > 1.0 + 1e-12:
                            raise RuntimeError("invalid frozen target action")
                        actions = runtime.replace_continuous_chunk(base_actions, continuous)
                        rollouts.append(runtime.execute_chunk(snapshot, actions))
                        residuals.append(residual)
                        full_actions.append(actions)
                        direction_ids.append(direction_id)
                        family_ids.append(int(bank["family_id"][direction_id]))
                        radius_ids.append(radius_id)
                        radii.append(radius)
                        signs.append(int(sign))
            packed = _pack_rollouts(rollouts)
            atomic_npz(
                shard,
                task_id=np.asarray(task_id),
                episode_id=np.asarray(record["episode_id"], dtype=np.int16),
                split=np.asarray(record["split"]),
                phase=np.asarray(record["phase"]),
                snapshot_index=np.asarray(index, dtype=np.int32),
                base_actions=base_actions,
                residual_action=np.asarray(residuals, dtype=np.float64),
                action_full=np.asarray(full_actions, dtype=np.float64),
                direction_id=np.asarray(direction_ids, dtype=np.int8),
                direction_family_id=np.asarray(family_ids, dtype=np.int8),
                radius_id=np.asarray(radius_ids, dtype=np.int8),
                radius=np.asarray(radii, dtype=np.float64),
                sign=np.asarray(signs, dtype=np.int8),
                **packed
            )
            marker = mark_complete(
                shard,
                {
                    "kind": "stage2_split_specific_support",
                    "task_id": task_id,
                    "episode_id": int(record["episode_id"]),
                    "split": record["split"],
                    "phase": record["phase"],
                    "branches": len(rollouts),
                    "created_utc": utc_now(),
                },
            )
            completed.append({"path": shard, "status": "created", "marker": marker})
            print(
                "STAGE2_SUPPORT_COMPLETE task=%s episode=%d phase=%s branches=%d"
                % (task_id, record["episode_id"], record["phase"], len(rollouts)),
                flush=True,
            )
    finally:
        runtime.close()
    manifest = os.path.join(output_root, "work", "collection", "support_%s.json" % task_id)
    atomic_json(manifest, {"created_utc": utc_now(), "task_id": task_id, "splits": sorted(requested), "shards": completed})
    return completed


def collect_candidates_task(paths, output_root, task_id, splits):
    requested = set(splits)
    if not requested.issubset({"calibration", "development", "confirmation"}):
        raise ValueError("candidate bank effects are only collected for calibration/development/confirmation")
    if "confirmation" in requested:
        with open(os.path.join(output_root, "work", "development_gate.json"), "r", encoding="utf-8") as handle:
            gate = json.load(handle)
        if not gate.get("confirmation_unlocked"):
            raise RuntimeError("confirmation is locked")
    with np.load(os.path.join(output_root, "action_bank.npz"), allow_pickle=False) as data:
        action_bank = np.asarray(data["residuals"], dtype=np.float64)
    task = _task(task_id)
    records = [
        row
        for row in _load_all_records(output_root)
        if row["task_id"] == task_id and row["split"] in requested
    ]
    runtime = LiberoTaskRuntime(task, paths["libero_source"], paths["dataset_root"])
    completed = []
    current_episode = None
    episode = None
    try:
        for record in sorted(records, key=lambda row: (row["episode_id"], PHASES.index(row["phase"]))):
            shard = _candidate_shard(
                output_root, record["split"], task_id, record["episode_id"], record["phase"]
            )
            valid, evidence = validate_complete(shard)
            if valid:
                completed.append({"path": shard, "status": "resumed", "evidence": evidence})
                continue
            if current_episode != record["episode_id"]:
                episode = runtime.load_episode(record["episode_id"])
                runtime.initialize_episode_model(episode)
                current_episode = record["episode_id"]
            index = int(record["snapshot_index"])
            base_actions = np.asarray(episode["actions"][index : index + config.CHUNK_HORIZON], dtype=np.float64)
            base_continuous = runtime.continuous_chunk(base_actions)
            snapshot = runtime.snapshot_from_recorded_state(
                episode["states"][index], episode["actions"][:index]
            )
            valid_indices = np.flatnonzero(
                np.max(np.abs(base_continuous[None, :] + action_bank), axis=1) <= 1.0 + 1e-12
            )
            if len(valid_indices) < MIN_VALID_BANK:
                raise RuntimeError("valid action bank below 128")
            rollouts = [runtime.execute_chunk(snapshot, base_actions)]
            full_actions = [base_actions]
            for bank_index in valid_indices:
                actions = runtime.replace_continuous_chunk(
                    base_actions, base_continuous + action_bank[int(bank_index)]
                )
                rollouts.append(runtime.execute_chunk(snapshot, actions))
                full_actions.append(actions)
            packed = _pack_rollouts(rollouts)
            atomic_npz(
                shard,
                task_id=np.asarray(task_id),
                episode_id=np.asarray(record["episode_id"], dtype=np.int16),
                split=np.asarray(record["split"]),
                phase=np.asarray(record["phase"]),
                snapshot_index=np.asarray(index, dtype=np.int32),
                base_actions=base_actions,
                bank_index=np.concatenate((np.asarray([-1], dtype=np.int16), valid_indices.astype(np.int16))),
                residual_action=np.concatenate(
                    (np.zeros((1, CONTINUOUS_DIM), dtype=np.float64), action_bank[valid_indices]), axis=0
                ),
                action_full=np.asarray(full_actions, dtype=np.float64),
                **packed
            )
            marker = mark_complete(
                shard,
                {
                    "kind": "stage2_common_action_bank_effects",
                    "task_id": task_id,
                    "episode_id": int(record["episode_id"]),
                    "split": record["split"],
                    "phase": record["phase"],
                    "branches": len(rollouts),
                    "valid_bank_size": int(len(valid_indices)),
                    "created_utc": utc_now(),
                },
            )
            completed.append({"path": shard, "status": "created", "marker": marker})
            print(
                "STAGE2_CANDIDATES_COMPLETE task=%s episode=%d phase=%s branches=%d"
                % (task_id, record["episode_id"], record["phase"], len(rollouts)),
                flush=True,
            )
    finally:
        runtime.close()
    manifest = os.path.join(output_root, "work", "collection", "candidates_%s.json" % task_id)
    atomic_json(manifest, {"created_utc": utc_now(), "task_id": task_id, "splits": sorted(requested), "shards": completed})
    return completed


def export_rollouts_zarr(output_root, confirmation=False):
    import glob
    import zarr

    selected_splits = {"confirmation"} if confirmation else {"train", "calibration", "development"}
    destination = os.path.join(
        output_root, "confirmation_rollouts.zarr" if confirmation else "development_rollouts.zarr"
    )
    root = zarr.open_group(destination, mode="w")
    root.attrs.update(
        {
            "schema_version": "r13-p15-ncea-stage2-rollouts-v1",
            "splits": sorted(selected_splits),
            "created_utc": utc_now(),
            "feature_names": list(FEATURE_NAMES),
        }
    )
    counts = {"support_states": 0, "candidate_states": 0, "branches": 0}
    for kind, directory in (("support", "support_shards"), ("candidate_bank", "candidate_shards")):
        pattern = os.path.join(output_root, "work", directory, "*", "*", "*.npz")
        for path in sorted(glob.glob(pattern)):
            with np.load(path, allow_pickle=False) as data:
                split = str(data["split"].item())
                if split not in selected_splits:
                    continue
                task_id = str(data["task_id"].item())
                episode_id = int(data["episode_id"].item())
                phase = str(data["phase"].item())
                group = root.require_group(
                    "%s/%s/%s/episode_%02d/%s" % (split, kind, task_id, episode_id, phase)
                )
                group.attrs.update(
                    {
                        "source_npz": os.path.relpath(path, output_root),
                        "source_sha256": sha256_file(path),
                    }
                )
                for name in data.files:
                    value = np.asarray(data[name])
                    if value.shape == () and value.dtype.kind in "US":
                        group.attrs[name] = str(value.item())
                    elif value.shape == ():
                        group.attrs[name] = value.item()
                    else:
                        chunks = (min(len(value), 64),) + value.shape[1:] if value.ndim else None
                        group.create_dataset(name, data=value, chunks=chunks, overwrite=True)
                counts["branches"] += int(len(data["settled"]))
                counts["support_states" if kind == "support" else "candidate_states"] += 1
    root.attrs.update(counts)
    return {"path": destination, "sha256": sha256_tree(destination), **counts}
