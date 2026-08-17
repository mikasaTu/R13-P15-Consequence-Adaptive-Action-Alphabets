"""Fresh-policy trajectory branch collection for the Stage 5 firewall."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np

from . import config
from .env_adapter import LiberoTaskRuntime
from .stage2 import _pack_rollouts
from .stage3_data import normalized_context, raw_context, true_distance_matrix
from .stage5_config import OUTPUT_RELATIVE, SCRATCH_ROOT, TASKS
from .stage5_data import CACHE_SCHEMA
from .stage5_fresh import _load_trajectory, _reset_for_seed
from .storage import atomic_json, atomic_npz, mark_complete, sha256_file, validate_complete


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _task(task_id):
    return next(dict(value) for value in TASKS if value["task_id"] == task_id)


def _paths(scratch_root, state_key):
    base = os.path.join(scratch_root, "fresh_branches", state_key.split("__", 1)[0], state_key)
    return {
        "context": base + "__context.npz",
        "support": base + "__support.npz",
        "candidate": base + "__candidate.npz",
    }


def _require_committed_split(project_root, output_root):
    path = os.path.join(output_root, "FRESH_CONFIRMATION_SPLIT.json")
    split = _load_json(path)
    if not split.get("complete"):
        raise RuntimeError("BLOCKED_NO_FRESH_TRAJECTORIES")
    relative = os.path.relpath(path, project_root)
    subprocess.check_call(
        ["git", "-C", project_root, "ls-files", "--error-unmatch", relative],
        stdout=subprocess.DEVNULL,
    )
    committed = subprocess.check_output(
        ["git", "-C", project_root, "show", "HEAD:" + relative]
    )
    import hashlib

    if hashlib.sha256(committed).hexdigest() != sha256_file(path):
        raise RuntimeError("fresh split differs from committed HEAD")
    return split


def _context_from_trajectory(values, index, current, base_actions):
    vector = np.asarray(current["vector"], dtype=np.float64)
    mask = np.asarray(current["mask"], dtype=bool)
    previous_vectors = []
    previous_masks = []
    for offset in (1, 2):
        source = int(index) - offset
        if source >= 0:
            previous_vectors.append(np.asarray(values["observable_state"][source], dtype=np.float64))
            previous_masks.append(np.asarray(values["observable_mask"][source], dtype=bool))
        else:
            previous_vectors.append(vector.copy())
            previous_masks.append(np.zeros_like(mask))
    deltas = []
    delta_masks = []
    left_vectors = (vector, previous_vectors[0])
    left_masks = (mask, previous_masks[0])
    for slot in range(2):
        if int(index) - slot - 1 < 0:
            delta = np.zeros_like(vector)
            active = np.zeros_like(mask)
        else:
            delta = left_vectors[slot] - previous_vectors[slot]
            active = left_masks[slot] & previous_masks[slot]
        delta[41:44] = 0.0
        active[41:44] = False
        deltas.append(delta)
        delta_masks.append(active)
    previous_action = np.zeros((2, config.ACTION_DIM), dtype=np.float64)
    previous_action_mask = np.zeros(2, dtype=bool)
    for slot, source in enumerate((int(index) - 1, int(index) - 2)):
        if source >= 0:
            previous_action[slot] = values["executed_action"][source]
            previous_action_mask[slot] = True
    return {
        "observable_state": vector,
        "observable_mask": mask,
        "history_delta": np.asarray(deltas, dtype=np.float64),
        "history_delta_mask": np.asarray(delta_masks, dtype=bool),
        "previous_action": previous_action,
        "previous_action_mask": previous_action_mask,
        "current_contact": np.asarray(bool(current["contacts"]["relevant"])),
        "nominal_continuous": base_actions[:, :6].reshape(-1).astype(np.float64),
        "nominal_full": np.asarray(base_actions, dtype=np.float64),
    }


def _save_context(path, record, arrays):
    atomic_npz(
        path,
        task_id=np.asarray(record["task_id"]),
        episode_id=np.asarray(record["source_episode_id"], dtype=np.int64),
        split=np.asarray("fresh_confirmation"),
        phase=np.asarray(record["phase"]),
        snapshot_index=np.asarray(record["state_index"], dtype=np.int32),
        **arrays,
    )
    mark_complete(path, {"kind": "stage5_fresh_observable_context", "state_key": record["state_key"], "future_or_outcome_input_present": False})


def _save_rollouts(path, record, base_actions, residuals, rollouts, metadata):
    packed = _pack_rollouts(rollouts)
    action_full = []
    for residual in residuals:
        actions = np.asarray(base_actions, dtype=np.float64).copy()
        actions[:, :6] = actions[:, :6] + np.asarray(residual, dtype=np.float64).reshape(4, 6)
        action_full.append(actions)
    atomic_npz(
        path,
        task_id=np.asarray(record["task_id"]),
        episode_id=np.asarray(record["source_episode_id"], dtype=np.int64),
        split=np.asarray("fresh_confirmation"),
        phase=np.asarray(record["phase"]),
        snapshot_index=np.asarray(record["state_index"], dtype=np.int32),
        base_actions=np.asarray(base_actions, dtype=np.float64),
        residual_action=np.asarray(residuals, dtype=np.float64),
        action_full=np.asarray(action_full, dtype=np.float64),
        **metadata,
        **packed,
    )
    mark_complete(path, {"kind": "stage5_fresh_branch_shard", "state_key": record["state_key"], "branches": len(rollouts), "restore_identical_state": True, "clipping": False})


def collect_task(
    project_root,
    task_id,
    libero_source=config.LIBERO_SOURCE_DEFAULT,
    dataset_root=config.DATASET_ROOT_DEFAULT,
    output_root=None,
    scratch_root=SCRATCH_ROOT,
):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    split = _require_committed_split(project_root, output_root)
    records = [row for row in split["records"] if row["task_id"] == task_id]
    with np.load(os.path.join(output_root, "LOCAL_BANK.npz"), allow_pickle=False) as data:
        local = {name: np.asarray(data[name]).copy() for name in data.files}
    with np.load(os.path.join(output_root, "FRESH_TARGET_BANK.npz"), allow_pickle=False) as data:
        fresh = {name: np.asarray(data[name]).copy() for name in data.files}
    runtime = LiberoTaskRuntime(_task(task_id), libero_source, dataset_root)
    rows = []
    started = time.time()
    try:
        for state_number, record in enumerate(records):
            paths = _paths(scratch_root, record["state_key"])
            complete = all(validate_complete(path)[0] for path in paths.values())
            if not complete:
                values = _load_trajectory(record["trajectory_path"])
                seed = int(record["rollout_seed"])
                index = int(record["state_index"])
                _reset_for_seed(runtime, seed)
                snapshot = runtime.snapshot_from_recorded_state(
                    values["sim_state"][index],
                    action_history=values["executed_action"][:index],
                )
                base_actions = np.asarray(values["predicted_h4_chunk"][index], dtype=np.float64)
                base_continuous = base_actions[:, :6].reshape(-1)
                candidate_valid = np.all(np.abs(base_continuous[None] + local["residuals"]) <= 1.0, axis=1)
                target_valid = np.all(np.abs(base_continuous[None] + fresh["residuals"]) <= 1.0, axis=1)
                if not np.all(candidate_valid) or not np.all(target_valid):
                    raise RuntimeError("frozen fresh state requires clipping")
                nominal = runtime.execute_chunk(snapshot, base_actions, settle_steps=3)
                context = _context_from_trajectory(values, index, nominal["initial"], base_actions)
                _save_context(paths["context"], record, context)
                support_rollouts = [nominal]
                for residual in fresh["residuals"]:
                    actions = runtime.replace_continuous_chunk(base_actions, base_continuous + residual)
                    support_rollouts.append(runtime.execute_chunk(snapshot, actions, settle_steps=3))
                support_residuals = np.concatenate((np.zeros((1, 24)), fresh["residuals"]), axis=0)
                _save_rollouts(
                    paths["support"],
                    record,
                    base_actions,
                    support_residuals,
                    support_rollouts,
                    {
                        "direction_id": np.concatenate((np.asarray([-1], dtype=np.int8), fresh["residual_direction_id"].astype(np.int8))),
                        "direction_family_id": np.concatenate((np.asarray([-1], dtype=np.int8), fresh["residual_family_id"].astype(np.int8))),
                        "radius_id": np.concatenate((np.asarray([-1], dtype=np.int8), fresh["residual_radius_id"].astype(np.int8))),
                        "radius": np.concatenate((np.asarray([0.0]), fresh["residual_radius"])),
                        "sign": np.concatenate((np.asarray([0], dtype=np.int8), fresh["residual_sign"].astype(np.int8))),
                    },
                )
                candidate_rollouts = [nominal]
                for residual in local["residuals"]:
                    actions = runtime.replace_continuous_chunk(base_actions, base_continuous + residual)
                    candidate_rollouts.append(runtime.execute_chunk(snapshot, actions, settle_steps=3))
                candidate_residuals = np.concatenate((np.zeros((1, 24)), local["residuals"]), axis=0)
                _save_rollouts(
                    paths["candidate"],
                    record,
                    base_actions,
                    candidate_residuals,
                    candidate_rollouts,
                    {
                        "bank_index": np.arange(-1, len(local["residuals"]), dtype=np.int16),
                        "source_bank_index": np.concatenate((np.asarray([-1]), local["source_indices"])).astype(np.int16),
                    },
                )
            rows.append(
                {
                    "state_key": record["state_key"],
                    "task_id": task_id,
                    "context_path": paths["context"],
                    "context_sha256": sha256_file(paths["context"]),
                    "support_path": paths["support"],
                    "support_sha256": sha256_file(paths["support"]),
                    "candidate_path": paths["candidate"],
                    "candidate_sha256": sha256_file(paths["candidate"]),
                }
            )
            print("fresh-branches task=%s state=%d/%d elapsed=%.1fs" % (task_id, state_number + 1, len(records), time.time() - started), flush=True)
    finally:
        runtime.close()
    result = {"task_id": task_id, "states": len(rows), "rows": rows, "support_branches_per_state": 97, "candidate_branches_per_state": 129, "clipped": 0}
    atomic_json(os.path.join(output_root, "FRESH_BRANCH_%s.json" % task_id.upper()), result)
    return result


def _load_shard(path):
    valid, evidence = validate_complete(path)
    if not valid:
        raise RuntimeError("incomplete shard %s: %s" % (path, evidence))
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]).copy() for name in data.files}


def build_fresh_cache(project_root, output_root=None, scratch_root=SCRATCH_ROOT):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    split = _load_json(os.path.join(output_root, "FRESH_CONFIRMATION_SPLIT.json"))
    train_cache_path = os.path.join(scratch_root, "derived", "stage5_train_cache.npz")
    with np.load(train_cache_path, allow_pickle=False) as data:
        consequence_scale = np.asarray(data["consequence_scale"], dtype=np.float64)
        context_center = np.asarray(data["context_center"], dtype=np.float32)
        context_scale = np.asarray(data["context_scale"], dtype=np.float32)
    with np.load(os.path.join(output_root, "LOCAL_BANK.npz"), allow_pickle=False) as data:
        candidate_residual = np.asarray(data["residuals"], dtype=np.float32)
        candidate_source = np.asarray(data["source_indices"], dtype=np.int64)
    with np.load(os.path.join(output_root, "FRESH_TARGET_BANK.npz"), allow_pickle=False) as data:
        target_residual = np.asarray(data["residuals"], dtype=np.float32)
        target_family = np.asarray(data["residual_family_id"], dtype=np.int8)
    rows = []
    arrays = {name: [] for name in ("context", "nominal_action", "true_distance", "target_contact_mode", "candidate_contact_mode", "current_contact", "key", "task_id", "task_index", "episode_id", "phase", "phase_index", "snapshot_index")}
    for record in split["records"]:
        paths = _paths(scratch_root, record["state_key"])
        context = _load_shard(paths["context"])
        support = _load_shard(paths["support"])
        candidate = _load_shard(paths["candidate"])
        wrapped = {"meta": {"task_id": record["task_id"], "episode_id": int(record["source_episode_id"]), "phase": record["phase"], "key": record["state_key"]}, "context": context, "support": support, "candidate": candidate}
        arrays["context"].append(normalized_context(wrapped, context_center, context_scale))
        arrays["nominal_action"].append(np.asarray(context["nominal_continuous"], dtype=np.float32))
        arrays["true_distance"].append(true_distance_matrix(wrapped, consequence_scale))
        arrays["target_contact_mode"].append(np.asarray(support["contact_mode"][1:], dtype=np.int8))
        arrays["candidate_contact_mode"].append(np.asarray(candidate["contact_mode"][1:], dtype=np.int8))
        arrays["current_contact"].append(int(bool(context["current_contact"].item())))
        arrays["key"].append(record["state_key"])
        arrays["task_id"].append(record["task_id"])
        arrays["task_index"].append([task["task_id"] for task in TASKS].index(record["task_id"]))
        arrays["episode_id"].append(int(record["source_episode_id"]))
        arrays["phase"].append(record["phase"])
        arrays["phase_index"].append(list(config.PHASES).index(record["phase"]))
        arrays["snapshot_index"].append(int(record["state_index"]))
        rows.append({"state_key": record["state_key"], **{name + "_sha256": sha256_file(path) for name, path in paths.items()}})
    destination = os.path.join(scratch_root, "derived", "stage5_fresh_confirmation_cache.npz")
    atomic_npz(
        destination,
        schema_version=np.asarray(CACHE_SCHEMA),
        source_stage4_cache_sha256=np.asarray("FRESH_POLICY_TRAJECTORY_CONFIRMATION"),
        split=np.asarray("fresh_confirmation"),
        context=np.asarray(arrays["context"], dtype=np.float32),
        nominal_action=np.asarray(arrays["nominal_action"], dtype=np.float32),
        target_residual=target_residual,
        candidate_residual=candidate_residual,
        candidate_source_index=candidate_source,
        true_distance=np.asarray(arrays["true_distance"], dtype=np.float32),
        target_contact_mode=np.asarray(arrays["target_contact_mode"], dtype=np.int8),
        candidate_contact_mode=np.asarray(arrays["candidate_contact_mode"], dtype=np.int8),
        current_contact=np.asarray(arrays["current_contact"], dtype=np.int8),
        key=np.asarray(arrays["key"]),
        task_id=np.asarray(arrays["task_id"]),
        task_index=np.asarray(arrays["task_index"], dtype=np.int8),
        episode_id=np.asarray(arrays["episode_id"], dtype=np.int64),
        phase=np.asarray(arrays["phase"]),
        phase_index=np.asarray(arrays["phase_index"], dtype=np.int8),
        snapshot_index=np.asarray(arrays["snapshot_index"], dtype=np.int32),
        direction_family_id=target_family,
        consequence_scale=consequence_scale,
        context_center=context_center,
        context_scale=context_scale,
    )
    mark_complete(destination, {"kind": "stage5_fresh_confirmation_cache", "schema_version": CACHE_SCHEMA, "states": len(rows), "targets": 96, "candidates": 128})
    task_manifests = [_load_json(os.path.join(output_root, "FRESH_BRANCH_%s.json" % task["task_id"].upper())) for task in TASKS]
    manifest = {
        "evidence_label": "FRESH_POLICY_TRAJECTORY_CONFIRMATION",
        "cache_path": destination,
        "cache_sha256": sha256_file(destination),
        "states": len(rows),
        "targets_per_state": 96,
        "candidates_per_state": 128,
        "nominal_branches": len(rows),
        "target_branches": len(rows) * 96,
        "candidate_branches": len(rows) * 128,
        "total_short_rollouts": len(rows) * 225,
        "clipped": 0,
        "task_manifests": task_manifests,
        "rows": rows,
    }
    atomic_json(os.path.join(output_root, "FRESH_BRANCH_MANIFEST.json"), manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect-task")
    collect.add_argument("--task-id", required=True, choices=[task["task_id"] for task in TASKS])
    sub.add_parser("build-cache")
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--libero-source", default=config.LIBERO_SOURCE_DEFAULT)
    parser.add_argument("--dataset-root", default=config.DATASET_ROOT_DEFAULT)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--scratch-root", default=SCRATCH_ROOT)
    args = parser.parse_args(argv)
    if args.command == "collect-task":
        result = collect_task(args.project_root, args.task_id, args.libero_source, args.dataset_root, args.output_root, args.scratch_root)
    else:
        result = build_fresh_cache(args.project_root, args.output_root, args.scratch_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
