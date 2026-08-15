"""Freeze all Stage 4 choices that must precede result inspection.

This command is intentionally read-only with respect to historical evidence.
It hashes the Stage 1--3 artifacts in their original checkout, selects exact
unused train timesteps directly from the official HDF5 demonstrations, and
writes only preregistration artifacts.  It never imports or starts LIBERO's
simulator and it never computes a Stage 4 method score.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict

import h5py
import numpy as np

from . import config
from .stage2 import _array_hash, _episode_hash
from .stage3 import _candidate_direction, _stable_seed
from .stage4_config import (
    ACTION_BANK_SIZE,
    BASE_ACTION_ABS_LIMIT,
    EXPECTED_HISTORICAL_HASHES,
    HISTORICAL_EVIDENCE,
    HISTORICAL_EXPLORATORY_EPISODES,
    HISTORICAL_REPOSITORY_ROOT,
    HISTORICAL_STAGE3_RELATIVE,
    HISTORICAL_STAGE_TREES,
    HORIZON,
    OUTPUT_RELATIVE,
    PHASES,
    STAGE2_ACTION_BANK_RELATIVE,
    STAGE4_INPUT_COMMIT,
    STAGE4_INPUT_TREE,
    SUPPORT_DIRECTION_COUNT,
    SUPPORT_DIRECTION_FAMILIES,
    SUPPORT_DIRECTIONS_PER_FAMILY,
    SUPPORT_MAX_COMPONENT,
    SUPPORT_RADII,
    SUPPORT_SIGNS,
    SUPPORT_TARGET_COUNT,
    TASKS,
    TRAIN_EPISODES,
    TRAIN_STATE_COUNT,
    TRAIN_STATE_SELECTION_SEED,
    TRAIN_STATES_PER_EPISODE_PHASE,
    TRAIN_SUPPORT_SEED,
    method_definitions,
)
from .storage import atomic_json, atomic_npz, sha256_file, sha256_tree


HISTORICAL_FILE_PATHS = {
    "action_bank.npz": STAGE2_ACTION_BANK_RELATIVE,
    "model_scalers.npz": HISTORICAL_STAGE3_RELATIVE + "/model_scalers.npz",
    "C3_NC_BIENCODER_member_0.pt": HISTORICAL_STAGE3_RELATIVE
    + "/models/C3_NC_BIENCODER_member_0.pt",
    "C3_NC_BIENCODER_member_1.pt": HISTORICAL_STAGE3_RELATIVE
    + "/models/C3_NC_BIENCODER_member_1.pt",
    "C3_NC_BIENCODER_member_2.pt": HISTORICAL_STAGE3_RELATIVE
    + "/models/C3_NC_BIENCODER_member_2.pt",
    "C4_NC_PAIR_RANKER_member_0.pt": HISTORICAL_STAGE3_RELATIVE
    + "/models/C4_NC_PAIR_RANKER_member_0.pt",
    "C4_NC_PAIR_RANKER_member_1.pt": HISTORICAL_STAGE3_RELATIVE
    + "/models/C4_NC_PAIR_RANKER_member_1.pt",
    "C4_NC_PAIR_RANKER_member_2.pt": HISTORICAL_STAGE3_RELATIVE
    + "/models/C4_NC_PAIR_RANKER_member_2.pt",
    "development_quantization.csv": HISTORICAL_STAGE3_RELATIVE
    + "/development_quantization.csv",
    "confirmation_quantization.csv": HISTORICAL_STAGE3_RELATIVE
    + "/confirmation_quantization.csv",
    "retrieval_metrics.csv": HISTORICAL_STAGE3_RELATIVE + "/retrieval_metrics.csv",
    "development_gate.json": HISTORICAL_STAGE3_RELATIVE + "/development_gate.json",
    "bootstrap_results.json": HISTORICAL_STAGE3_RELATIVE + "/bootstrap_results.json",
    "STAGE3_REPORT.md": HISTORICAL_STAGE3_RELATIVE + "/STAGE3_REPORT.md",
}


def _git(repository, *arguments):
    return subprocess.check_output(
        ["git", "-C", repository, *arguments], stderr=subprocess.STDOUT, text=True
    ).strip()


def _canonical_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _historical_binding():
    head = _git(HISTORICAL_REPOSITORY_ROOT, "rev-parse", "HEAD")
    tree = _git(HISTORICAL_REPOSITORY_ROOT, "rev-parse", "HEAD^{tree}")
    if head != STAGE4_INPUT_COMMIT or tree != STAGE4_INPUT_TREE:
        raise RuntimeError(
            "historical checkout moved: %s/%s expected %s/%s"
            % (head, tree, STAGE4_INPUT_COMMIT, STAGE4_INPUT_TREE)
        )

    observed_trees = {}
    stage_paths = {
        "stage1": "experiments/r13_p15_caaa_v2/stage1",
        "stage1_5": "experiments/r13_p15_caaa_v2/stage1_5",
        "stage2": "experiments/r13_p15_ncea/stage2",
        "stage3": HISTORICAL_STAGE3_RELATIVE,
    }
    for stage, relative in stage_paths.items():
        observed = _git(HISTORICAL_REPOSITORY_ROOT, "rev-parse", "HEAD:" + relative)
        expected = HISTORICAL_STAGE_TREES[stage]
        if observed != expected:
            raise RuntimeError("historical tree mismatch for %s" % stage)
        observed_trees[stage] = {
            "path": relative,
            "git_tree_object": observed,
            "expected_git_tree_object": expected,
            "matched": True,
        }

    files = {}
    for name, relative in HISTORICAL_FILE_PATHS.items():
        path = os.path.join(HISTORICAL_REPOSITORY_ROOT, relative)
        observed = sha256_file(path)
        expected = EXPECTED_HISTORICAL_HASHES[name]
        if observed != expected:
            raise RuntimeError("historical file mismatch for %s" % name)
        files[name] = {
            "relative_path": relative,
            "bytes": int(os.path.getsize(path)),
            "sha256": observed,
            "expected_sha256": expected,
            "matched": True,
        }

    c3 = [
        files["C3_NC_BIENCODER_member_%d.pt" % index]["sha256"]
        for index in range(3)
    ]
    c4 = [
        files["C4_NC_PAIR_RANKER_member_%d.pt" % index]["sha256"]
        for index in range(3)
    ]
    c5_manifest = {
        "kind": "frozen_stage3_composite_no_standalone_checkpoint",
        "ordered_c3_ensemble_sha256": c3,
        "ordered_c4_ensemble_sha256": c4,
        "composition": (
            "C3 predicted-embedding deterministic K=64 atlas followed by "
            "C4 symmetric pair-ranker reranking"
        ),
        "candidate_bank_sha256": files["action_bank.npz"]["sha256"],
        "scaler_sha256": files["model_scalers.npz"]["sha256"],
    }
    c5_manifest["manifest_sha256"] = _canonical_sha256(c5_manifest)

    stage3_environment = _load_json(
        os.path.join(
            HISTORICAL_REPOSITORY_ROOT,
            HISTORICAL_STAGE3_RELATIVE,
            "execution_environment.json",
        )
    )
    libero_source = config.LIBERO_SOURCE_DEFAULT
    # The frozen LIBERO checkout is a detached worktree whose .git file points
    # to a parent checkout that is no longer mounted.  Its source bytes remain
    # independently verifiable, while the upstream commit is bound by the
    # already-published Stage 3 environment record.
    try:
        observed_libero_commit = _git(libero_source, "rev-parse", "HEAD")
        commit_source = "live_git_metadata"
    except subprocess.CalledProcessError:
        observed_libero_commit = stage3_environment["libero_commit"]
        commit_source = "published_stage3_environment_broken_worktree_pointer"
    observed_libero_tree = sha256_tree(libero_source)
    if observed_libero_commit != config.UPSTREAM_LIBERO_COMMIT:
        raise RuntimeError("LIBERO commit mismatch")
    expected_libero_tree = stage3_environment["libero_tree_sha256"]
    if observed_libero_tree != expected_libero_tree:
        raise RuntimeError("LIBERO source tree mismatch")

    return {
        "binding_kind": "stage4_pre_result_read_only_historical_binding",
        "stage4_input": {
            "commit": head,
            "tree": tree,
            "branch_requested": "r13-p15-stage4-c3-context-trust-region",
        },
        "historical_evidence": HISTORICAL_EVIDENCE,
        "historical_stage_trees": observed_trees,
        "historical_files": files,
        "stage3_c5_composite": c5_manifest,
        "libero": {
            "source_path": libero_source,
            "commit": observed_libero_commit,
            "commit_source": commit_source,
            "source_tree_sha256": observed_libero_tree,
        },
        "stage3_environment": stage3_environment,
        "historical_files_modified": False,
        "all_hashes_match": True,
        "stage4_method_result_inspected_before_freeze": False,
        "pai_jobs_submitted": 0,
        "policy_training_performed": False,
    }


def generate_training_support_bank():
    """Return 24 directions and all 96 radius/sign target residuals."""
    rng = np.random.RandomState(TRAIN_SUPPORT_SEED)
    directions = []
    family_ids = []
    family_local_ids = []
    for family_id, family in enumerate(SUPPORT_DIRECTION_FAMILIES):
        family_index = 0
        attempts = 0
        while family_index < SUPPORT_DIRECTIONS_PER_FAMILY:
            attempts += 1
            if attempts > 100000:
                raise RuntimeError("unable to generate Stage 4 support bank")
            try:
                candidate = _candidate_direction(rng, family, family_index)
            except ValueError:
                continue
            if directions and float(
                np.max(np.abs(np.asarray(directions).dot(candidate)))
            ) > 0.999999:
                continue
            if float(np.max(np.abs(candidate))) > SUPPORT_MAX_COMPONENT + 1e-12:
                continue
            directions.append(candidate)
            family_ids.append(family_id)
            family_local_ids.append(family_index)
            family_index += 1

    directions = np.asarray(directions, dtype=np.float64)
    if directions.shape != (SUPPORT_DIRECTION_COUNT, 24):
        raise AssertionError(directions.shape)
    residuals = []
    residual_direction_ids = []
    residual_family_ids = []
    residual_radius_ids = []
    residual_signs = []
    for direction_id, direction in enumerate(directions):
        for radius_id, radius in enumerate(SUPPORT_RADII):
            for sign in SUPPORT_SIGNS:
                residuals.append(float(sign) * float(radius) * direction)
                residual_direction_ids.append(direction_id)
                residual_family_ids.append(family_ids[direction_id])
                residual_radius_ids.append(radius_id)
                residual_signs.append(sign)
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.shape != (SUPPORT_TARGET_COUNT, 24):
        raise AssertionError(residuals.shape)
    if Counter(residual_family_ids) != {
        family_id: SUPPORT_TARGET_COUNT // len(SUPPORT_DIRECTION_FAMILIES)
        for family_id in range(len(SUPPORT_DIRECTION_FAMILIES))
    }:
        raise AssertionError("support family imbalance")
    return {
        "directions": directions,
        "direction_family_id": np.asarray(family_ids, dtype=np.int8),
        "direction_family_local_id": np.asarray(family_local_ids, dtype=np.int8),
        "direction_family_names": np.asarray(SUPPORT_DIRECTION_FAMILIES),
        "residuals": residuals,
        "residual_direction_id": np.asarray(residual_direction_ids, dtype=np.int8),
        "residual_family_id": np.asarray(residual_family_ids, dtype=np.int8),
        "residual_radius_id": np.asarray(residual_radius_ids, dtype=np.int8),
        "residual_radius": np.asarray(
            [SUPPORT_RADII[index] for index in residual_radius_ids], dtype=np.float64
        ),
        "residual_sign": np.asarray(residual_signs, dtype=np.int8),
        "seed": np.asarray(TRAIN_SUPPORT_SEED, dtype=np.int64),
    }


def _demo_path(task):
    return os.path.join(
        config.DATASET_ROOT_DEFAULT,
        config.SUITE,
        task["task_name"] + "_demo.hdf5",
    )


def _phase_windows(anchors, maximum_index):
    ordered = sorted((int(index), phase) for phase, index in anchors.items())
    result = {}
    for position, (anchor, phase) in enumerate(ordered):
        low = 0 if position == 0 else (ordered[position - 1][0] + anchor) // 2 + 1
        high = (
            maximum_index
            if position == len(ordered) - 1
            else (anchor + ordered[position + 1][0]) // 2
        )
        result[phase] = (max(0, low), min(maximum_index, high))
    return result


def _valid_action_chunk(actions, index, action_bank, support_residuals):
    chunk = np.asarray(actions[index : index + HORIZON, :6], dtype=np.float64).reshape(-1)
    if chunk.shape != (24,):
        return False, None
    if float(np.max(np.abs(chunk))) > BASE_ACTION_ABS_LIMIT + 1e-12:
        return False, chunk
    if float(np.max(np.abs(chunk[None, :] + action_bank))) > 1.0 + 1e-12:
        return False, chunk
    if float(np.max(np.abs(chunk[None, :] + support_residuals))) > 1.0 + 1e-12:
        return False, chunk
    return True, chunk


def _load_stage3_snapshot_metadata():
    path = os.path.join(
        HISTORICAL_REPOSITORY_ROOT, HISTORICAL_STAGE3_RELATIVE, "episode_split.json"
    )
    payload = _load_json(path)
    return payload, path


def select_training_states(action_bank, support_bank):
    """Freeze exactly 768 unique, executable, non-Stage-3 train snapshots."""
    stage3, split_path = _load_stage3_snapshot_metadata()
    all_stage3 = {
        (row["task_id"], int(row["episode_id"]), int(row["snapshot_index"]))
        for row in stage3["snapshots"]
    }
    anchors = {
        (row["task_id"], int(row["episode_id"]), row["phase"]): row
        for row in stage3["snapshots"]
        if int(row["episode_id"]) in TRAIN_EPISODES
    }
    episode_hashes = {
        (row["task_id"], int(row["episode_id"])): row["episode_sha256"]
        for row in stage3["snapshots"]
    }
    stage2_inventory_path = os.path.join(
        HISTORICAL_REPOSITORY_ROOT,
        "experiments/r13_p15_ncea/stage2/fresh_episode_inventory.json",
    )
    stage2_inventory = _load_json(stage2_inventory_path)
    demo_hashes = {
        row["task"]["task_id"]: row["demo"]["sha256"]
        for row in stage2_inventory["tasks"]
    }

    records = []
    fallback_count = 0
    residuals = support_bank["residuals"]
    for task in TASKS:
        task_id = task["task_id"]
        path = _demo_path(task)
        with h5py.File(path, "r") as handle:
            data = handle["data"]
            for episode_id in TRAIN_EPISODES:
                group = data["demo_%d" % episode_id]
                actions = np.asarray(group["actions"], dtype=np.float64)
                states = np.asarray(group["states"], dtype=np.float64)
                observed_episode_hash = _episode_hash(group)
                expected_episode_hash = episode_hashes[(task_id, episode_id)]
                if observed_episode_hash != expected_episode_hash:
                    raise RuntimeError(
                        "episode hash mismatch for %s e%d" % (task_id, episode_id)
                    )
                anchor_map = {
                    phase: int(anchors[(task_id, episode_id, phase)]["snapshot_index"])
                    for phase in PHASES
                }
                maximum_index = len(actions) - HORIZON
                windows = _phase_windows(anchor_map, maximum_index)
                selected_in_episode = set()
                for phase in PHASES:
                    anchor = anchor_map[phase]
                    low, high = windows[phase]
                    candidates = []
                    for index in range(maximum_index + 1):
                        if index in selected_in_episode:
                            continue
                        if (task_id, episode_id, index) in all_stage3:
                            continue
                        valid, continuous = _valid_action_chunk(
                            actions, index, action_bank, residuals
                        )
                        if not valid:
                            continue
                        in_window = low <= index <= high
                        tie = _stable_seed(
                            TRAIN_STATE_SELECTION_SEED,
                            task_id,
                            episode_id,
                            phase,
                            index,
                        )
                        candidates.append(
                            (not in_window, abs(index - anchor), tie, index, continuous)
                        )
                    candidates.sort(key=lambda value: value[:4])
                    chosen = candidates[:TRAIN_STATES_PER_EPISODE_PHASE]
                    if len(chosen) != TRAIN_STATES_PER_EPISODE_PHASE:
                        raise RuntimeError(
                            "insufficient unused states for %s e%d %s"
                            % (task_id, episode_id, phase)
                        )
                    for local_id, (outside, _, _, index, continuous) in enumerate(chosen):
                        selected_in_episode.add(index)
                        fallback_count += int(outside)
                        full_actions = np.asarray(
                            actions[index : index + HORIZON], dtype=np.float64
                        )
                        state = np.asarray(states[index], dtype=np.float64)
                        records.append(
                            {
                                "key": "%s__e%02d__%s__u%d"
                                % (task_id, episode_id, phase, local_id),
                                "task_id": task_id,
                                "task_name": task["task_name"],
                                "episode_id": int(episode_id),
                                "split": "train",
                                "phase": phase,
                                "phase_local_id": int(local_id),
                                "snapshot_index": int(index),
                                "phase_anchor_snapshot_index": int(anchor),
                                "phase_window": [int(low), int(high)],
                                "phase_window_fallback": bool(outside),
                                "stage3_snapshot_excluded": True,
                                "snapshot_state_sha256": _array_hash(state),
                                "snapshot_state_dimension": int(state.size),
                                "base_action_sha256": _array_hash(full_actions),
                                "base_continuous_sha256": _array_hash(continuous),
                                "base_actions": full_actions.tolist(),
                                "base_continuous": continuous.tolist(),
                                "episode_sha256": observed_episode_hash,
                                "demo_file_sha256": demo_hashes[task_id],
                                "episode_length": int(len(actions)),
                                "max_abs_base_continuous": float(
                                    np.max(np.abs(continuous))
                                ),
                                "candidate_bank_executable_without_clipping": True,
                                "support_bank_executable_without_clipping": True,
                            }
                        )

    if len(records) != TRAIN_STATE_COUNT:
        raise AssertionError((len(records), TRAIN_STATE_COUNT))
    keys = [row["key"] for row in records]
    state_locations = [
        (row["task_id"], row["episode_id"], row["snapshot_index"])
        for row in records
    ]
    if len(set(keys)) != len(keys) or len(set(state_locations)) != len(state_locations):
        raise AssertionError("training state duplication")
    if any(location in all_stage3 for location in state_locations):
        raise AssertionError("Stage 3 snapshot leaked into Stage 4 training")
    task_counts = Counter(row["task_id"] for row in records)
    phase_counts = Counter(row["phase"] for row in records)
    episode_counts = Counter((row["task_id"], row["episode_id"]) for row in records)
    if set(task_counts.values()) != {TRAIN_STATE_COUNT // len(TASKS)}:
        raise AssertionError(task_counts)
    if set(phase_counts.values()) != {TRAIN_STATE_COUNT // len(PHASES)}:
        raise AssertionError(phase_counts)
    if set(episode_counts.values()) != {
        len(PHASES) * TRAIN_STATES_PER_EPISODE_PHASE
    }:
        raise AssertionError(episode_counts)
    return {
        "schema_version": "stage4-training-state-manifest-v1",
        "selection_seed": TRAIN_STATE_SELECTION_SEED,
        "selection_rule": (
            "For every task/train-episode/Stage-3 phase anchor, choose the "
            "three nearest executable unused H=4 timesteps inside the "
            "deterministic midpoint phase window; stable hash breaks ties and "
            "out-of-window candidates are ordered only as a recorded fallback."
        ),
        "stage3_snapshot_manifest": {
            "relative_path": HISTORICAL_STAGE3_RELATIVE + "/episode_split.json",
            "sha256": sha256_file(split_path),
            "snapshot_count_excluded": len(all_stage3),
        },
        "action_bank_sha256": EXPECTED_HISTORICAL_HASHES["action_bank.npz"],
        "support_target_count_per_state": SUPPORT_TARGET_COUNT,
        "candidate_count_per_state": ACTION_BANK_SIZE,
        "state_count": len(records),
        "task_counts": dict(sorted(task_counts.items())),
        "phase_counts": dict(sorted(phase_counts.items())),
        "states_per_task_episode": len(PHASES) * TRAIN_STATES_PER_EPISODE_PHASE,
        "phase_window_fallback_count": int(fallback_count),
        "all_actions_executable_without_clipping": True,
        "all_stage3_snapshots_excluded": True,
        "no_confirmation_state_used": True,
        "records": records,
    }


def fresh_state_inventory():
    """Audit fresh-source availability without freezing or executing it."""
    stage3, _ = _load_stage3_snapshot_metadata()
    used = defaultdict(set)
    for row in stage3["snapshots"]:
        used[(row["task_id"], int(row["episode_id"]))].add(
            int(row["snapshot_index"])
        )
    tasks = []
    for task in TASKS:
        path = _demo_path(task)
        with h5py.File(path, "r") as handle:
            data = handle["data"]
            ids = sorted(
                int(name.split("_")[1])
                for name in data.keys()
                if name.startswith("demo_")
            )
            fresh_ids = [episode_id for episode_id in ids if episode_id >= 50]
            unused_by_episode = {}
            for episode_id in HISTORICAL_EXPLORATORY_EPISODES:
                group = data["demo_%d" % episode_id]
                maximum_index = len(group["actions"]) - HORIZON
                unused_by_episode[str(episode_id)] = int(
                    sum(
                        index not in used[(task["task_id"], episode_id)]
                        for index in range(maximum_index + 1)
                    )
                )
            tasks.append(
                {
                    "task_id": task["task_id"],
                    "demo_path": path,
                    "num_demos": len(ids),
                    "minimum_demo_id": min(ids),
                    "maximum_demo_id": max(ids),
                    "successful_ids_ge_50": fresh_ids,
                    "unused_h4_timesteps_in_episodes_40_49": unused_by_episode,
                    "unused_h4_timestep_count_episodes_40_49": int(
                        sum(unused_by_episode.values())
                    ),
                }
            )
    return {
        "schema_version": "stage4-fresh-state-inventory-v1",
        "audit_only_no_branch_execution": True,
        "preferred_source_order": [
            "unused successful demonstrations with IDs >= 50",
            "new successful trajectories from a frozen nominal generator and new seeds",
            "bounded physical perturbations of previously unused timesteps",
        ],
        "source_1_available": any(row["successful_ids_ge_50"] for row in tasks),
        "source_1_result": "UNAVAILABLE_OFFICIAL_FILES_END_AT_DEMO_49",
        "source_2_status": "NO_FROZEN_NOMINAL_GENERATOR_AVAILABLE_AT_PREREGISTRATION",
        "source_3_available_for_later_freeze": all(
            row["unused_h4_timestep_count_episodes_40_49"] > 0 for row in tasks
        ),
        "source_3_required_label": "FRESH_PERTURBED_STATE_CONFIRMATION",
        "source_3_is_new_episode_claim": False,
        "confirmation_split_frozen": False,
        "confirmation_execution_allowed": False,
        "freeze_condition": (
            "Only after all development methods, selected model/ensemble, "
            "atlas algorithm, K, L, metrics and thresholds are frozen."
        ),
        "tasks": tasks,
    }


def freeze(project_root):
    project_root = os.path.abspath(project_root)
    output_root = os.path.join(project_root, OUTPUT_RELATIVE)
    os.makedirs(output_root, exist_ok=True)

    forbidden_results = (
        "C3_FAILURE_DECOMPOSITION.csv",
        "CONTEXT_DEPENDENCE_AUDIT.csv",
        "C3_CONTEXT_INTERVENTIONS.csv",
        "MODEL_SELECTION.json",
        "DEVELOPMENT_RETRIEVAL.csv",
        "DEVELOPMENT_REALIZED.csv",
        "DEVELOPMENT_GATE.json",
        "FRESH_CONFIRMATION_SPLIT.json",
        "CONFIRMATION_RETRIEVAL.csv",
        "CONFIRMATION_REALIZED.csv",
        "BOOTSTRAP_RESULTS.json",
        "STAGE4_REPORT.md",
    )
    present = [name for name in forbidden_results if os.path.exists(os.path.join(output_root, name))]
    if present:
        raise RuntimeError("refusing to refreeze after Stage 4 results: " + ",".join(present))

    binding = _historical_binding()
    support = generate_training_support_bank()
    action_bank_path = os.path.join(
        HISTORICAL_REPOSITORY_ROOT, STAGE2_ACTION_BANK_RELATIVE
    )
    with np.load(action_bank_path, allow_pickle=False) as data:
        action_bank = np.asarray(data["residuals"], dtype=np.float64)
    if action_bank.shape != (ACTION_BANK_SIZE, 24):
        raise RuntimeError("historical action bank shape changed")
    if sha256_file(action_bank_path) != EXPECTED_HISTORICAL_HASHES["action_bank.npz"]:
        raise RuntimeError("historical action bank hash changed")

    support_path = os.path.join(output_root, "training_support_bank.npz")
    atomic_npz(support_path, **support)
    support_summary = {
        "relative_path": "training_support_bank.npz",
        "sha256": sha256_file(support_path),
        "direction_array_sha256": _array_hash(support["directions"]),
        "residual_array_sha256": _array_hash(support["residuals"]),
        "direction_count": SUPPORT_DIRECTION_COUNT,
        "target_count": SUPPORT_TARGET_COUNT,
        "family_target_counts": {
            SUPPORT_DIRECTION_FAMILIES[index]: int(
                np.sum(support["residual_family_id"] == index)
            )
            for index in range(len(SUPPORT_DIRECTION_FAMILIES))
        },
        "radii": list(SUPPORT_RADII),
        "signs": list(SUPPORT_SIGNS),
        "seed": TRAIN_SUPPORT_SEED,
    }
    manifest = select_training_states(action_bank, support)
    manifest["support_bank"] = support_summary

    methods = method_definitions()
    methods["pre_result_data_selection"] = {
        "training_state_count": TRAIN_STATE_COUNT,
        "episodes": list(TRAIN_EPISODES),
        "states_per_task_episode_phase": TRAIN_STATES_PER_EPISODE_PHASE,
        "stage3_snapshots_excluded": True,
        "phase_balance": "exact",
        "support_family_balance": "1:1:1 exact",
        "same_frozen_m256_action_bank": True,
        "base_action_absolute_limit": BASE_ACTION_ABS_LIMIT,
    }
    methods["pre_result_artifact_contract"] = {
        "results_may_be_computed_only_after_git_commit": True,
        "fresh_confirmation_may_be_executed_only_after_FRESH_CONFIRMATION_SPLIT_commit": True,
        "sacrificial_replay_states_must_not_be_confirmation_states": True,
    }

    atomic_json(os.path.join(output_root, "HISTORICAL_BINDING.json"), binding)
    atomic_json(os.path.join(output_root, "METHOD_DEFINITIONS.json"), methods)
    atomic_json(os.path.join(output_root, "TRAINING_STATE_MANIFEST.json"), manifest)
    atomic_json(os.path.join(output_root, "FRESH_STATE_INVENTORY.json"), fresh_state_inventory())
    return {
        "output_root": output_root,
        "training_states": manifest["state_count"],
        "support_targets_per_state": SUPPORT_TARGET_COUNT,
        "candidate_actions_per_state": ACTION_BANK_SIZE,
        "historical_hashes_match": binding["all_hashes_match"],
        "stage4_results_written": False,
    }


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    args = parser.parse_args(argv)
    print(json.dumps(freeze(args.project_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
