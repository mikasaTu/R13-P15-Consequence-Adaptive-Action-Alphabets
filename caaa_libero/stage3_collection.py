"""Resumable CPU branch collection for the Stage 3 NCER-AA audit.

Large simulator shards live in the frozen CPFS scratch root.  The repository
contains their completion/hash manifest and every derived required artifact.
No policy is trained by this module.
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import config
from .env_adapter import FEATURE_NAMES, LiberoTaskRuntime
from .stage2 import _pack_rollouts
from .stage2_config import split_for_episode as stage2_split_for_episode
from .stage3 import _task, utc_now
from .stage3_config import (
    ACTION_BANK_SIZE,
    CONTINUOUS_DIM,
    DIRECTION_COUNT,
    DIRECTION_FAMILIES,
    PHASES,
    RADII_PER_DIRECTION,
    SCRATCH_ROOT,
    SIGNS,
    TASKS,
)
from .storage import atomic_json, atomic_npz, mark_complete, validate_complete


def _load_records(output_root):
    with open(os.path.join(output_root, "episode_split.json"), "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload["snapshots"])


def _load_codebooks(output_root):
    with np.load(os.path.join(output_root, "support_codebooks.npz"), allow_pickle=False) as data:
        names = data["split_names"].astype(str)
        result = {}
        for index, name in enumerate(names):
            result[str(name)] = {
                "directions": np.asarray(data["directions"][index], dtype=np.float64),
                "radii": np.asarray(data["radii"][index], dtype=np.float64),
                "family_id": np.asarray(data["direction_family_id"][index], dtype=np.int8),
            }
    return result


def _load_action_bank(project_root):
    path = os.path.join(project_root, "experiments", "r13_p15_ncea", "stage2", "action_bank.npz")
    with np.load(path, allow_pickle=False) as data:
        bank = np.asarray(data["residuals"], dtype=np.float64)
    if bank.shape != (ACTION_BANK_SIZE, CONTINUOUS_DIM):
        raise RuntimeError("frozen action bank shape changed")
    return bank


def context_shard(scratch_root, task_id, episode_id, phase):
    return os.path.join(
        scratch_root,
        "context_shards",
        task_id,
        "%s__e%02d__%s.npz" % (task_id, int(episode_id), phase),
    )


def support_shard(scratch_root, split, task_id, episode_id, phase):
    return os.path.join(
        scratch_root,
        "support_shards",
        split,
        task_id,
        "%s__e%02d__%s.npz" % (task_id, int(episode_id), phase),
    )


def candidate_shard(scratch_root, split, task_id, episode_id, phase):
    return os.path.join(
        scratch_root,
        "candidate_shards",
        split,
        task_id,
        "%s__e%02d__%s.npz" % (task_id, int(episode_id), phase),
    )


def stage2_support_shard(project_root, task_id, episode_id, phase):
    split = stage2_split_for_episode(int(episode_id))
    return os.path.join(
        project_root,
        "experiments",
        "r13_p15_ncea",
        "stage2",
        "work",
        "support_shards",
        split,
        task_id,
        "%s__e%02d__%s.npz" % (task_id, int(episode_id), phase),
    )


def stage2_candidate_shard(project_root, task_id, episode_id, phase):
    split = stage2_split_for_episode(int(episode_id))
    return os.path.join(
        project_root,
        "experiments",
        "r13_p15_ncea",
        "stage2",
        "work",
        "candidate_shards",
        split,
        task_id,
        "%s__e%02d__%s.npz" % (task_id, int(episode_id), phase),
    )


def resolved_support_shard(project_root, scratch_root, split, task_id, episode_id, phase):
    if split == "train":
        return stage2_support_shard(project_root, task_id, episode_id, phase)
    return support_shard(scratch_root, split, task_id, episode_id, phase)


def resolved_candidate_shard(project_root, scratch_root, split, task_id, episode_id, phase):
    if split == "train" and int(episode_id) >= 24:
        return stage2_candidate_shard(project_root, task_id, episode_id, phase)
    return candidate_shard(scratch_root, split, task_id, episode_id, phase)


def _recorded_measure(runtime, episode, index):
    runtime.env.sim.set_state_from_flattened(episode["states"][int(index)])
    runtime.env.sim.forward()
    runtime.env._post_process()
    return runtime.measure()


def _context_arrays(runtime, episode, record, base_actions):
    index = int(record["snapshot_index"])
    measured = []
    for offset in (0, 1, 2):
        source = max(index - offset, 0)
        measured.append(_recorded_measure(runtime, episode, source))
    current = measured[0]
    deltas = []
    delta_masks = []
    for left, right in ((0, 1), (1, 2)):
        if index - right < 0:
            deltas.append(np.zeros(len(FEATURE_NAMES), dtype=np.float64))
            delta_masks.append(np.zeros(len(FEATURE_NAMES), dtype=bool))
        else:
            deltas.append(measured[left]["vector"] - measured[right]["vector"])
            delta_masks.append(measured[left]["mask"] & measured[right]["mask"])
    previous_actions = np.zeros((2, config.ACTION_DIM), dtype=np.float64)
    previous_action_mask = np.zeros(2, dtype=bool)
    for slot, source in enumerate((index - 1, index - 2)):
        if source >= 0:
            previous_actions[slot] = episode["actions"][source]
            previous_action_mask[slot] = True
    return {
        "observable_state": np.asarray(current["vector"], dtype=np.float64),
        "observable_mask": np.asarray(current["mask"], dtype=bool),
        "history_delta": np.asarray(deltas, dtype=np.float64),
        "history_delta_mask": np.asarray(delta_masks, dtype=bool),
        "previous_action": previous_actions,
        "previous_action_mask": previous_action_mask,
        "current_contact": np.asarray(current["contacts"]["relevant"], dtype=bool),
        "nominal_continuous": runtime.continuous_chunk(base_actions),
        "nominal_full": np.asarray(base_actions, dtype=np.float64),
    }


def _write_context(path, record, arrays):
    atomic_npz(
        path,
        task_id=np.asarray(record["task_id"]),
        episode_id=np.asarray(record["episode_id"], dtype=np.int16),
        split=np.asarray(record["split"]),
        phase=np.asarray(record["phase"]),
        snapshot_index=np.asarray(record["snapshot_index"], dtype=np.int32),
        **arrays
    )
    return mark_complete(
        path,
        {
            "kind": "stage3_observable_context",
            "task_id": record["task_id"],
            "episode_id": int(record["episode_id"]),
            "split": record["split"],
            "phase": record["phase"],
            "future_or_outcome_input_present": False,
            "created_utc": utc_now(),
        },
    )


def _write_support(path, record, base_actions, bank, nominal, runtime, snapshot):
    base_continuous = runtime.continuous_chunk(base_actions)
    rollouts = [nominal]
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
                if float(np.max(np.abs(continuous))) > 1.0 + 1e-12:
                    raise RuntimeError("frozen Stage 3 support requires clipping")
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
        path,
        task_id=np.asarray(record["task_id"]),
        episode_id=np.asarray(record["episode_id"], dtype=np.int16),
        split=np.asarray(record["split"]),
        phase=np.asarray(record["phase"]),
        snapshot_index=np.asarray(record["snapshot_index"], dtype=np.int32),
        base_actions=base_actions,
        residual_action=np.asarray(residuals, dtype=np.float64),
        action_full=np.asarray(full_actions, dtype=np.float64),
        direction_id=np.asarray(direction_ids, dtype=np.int8),
        direction_family_id=np.asarray(family_ids, dtype=np.int8),
        direction_family_names=np.asarray(DIRECTION_FAMILIES),
        radius_id=np.asarray(radius_ids, dtype=np.int8),
        radius=np.asarray(radii, dtype=np.float64),
        sign=np.asarray(signs, dtype=np.int8),
        **packed
    )
    return mark_complete(
        path,
        {
            "kind": "stage3_split_fixed_fresh_support",
            "task_id": record["task_id"],
            "episode_id": int(record["episode_id"]),
            "split": record["split"],
            "phase": record["phase"],
            "branches": len(rollouts),
            "created_utc": utc_now(),
        },
    )


def _write_candidates(path, record, base_actions, action_bank, nominal, runtime, snapshot):
    base_continuous = runtime.continuous_chunk(base_actions)
    valid = np.max(np.abs(base_continuous[None, :] + action_bank), axis=1) <= 1.0 + 1e-12
    if not bool(np.all(valid)):
        raise RuntimeError("frozen Stage 3 bank requires clipping")
    rollouts = [nominal]
    full_actions = [base_actions]
    for bank_index in range(ACTION_BANK_SIZE):
        actions = runtime.replace_continuous_chunk(
            base_actions, base_continuous + action_bank[bank_index]
        )
        rollouts.append(runtime.execute_chunk(snapshot, actions))
        full_actions.append(actions)
    packed = _pack_rollouts(rollouts)
    atomic_npz(
        path,
        task_id=np.asarray(record["task_id"]),
        episode_id=np.asarray(record["episode_id"], dtype=np.int16),
        split=np.asarray(record["split"]),
        phase=np.asarray(record["phase"]),
        snapshot_index=np.asarray(record["snapshot_index"], dtype=np.int32),
        base_actions=base_actions,
        bank_index=np.arange(-1, ACTION_BANK_SIZE, dtype=np.int16),
        residual_action=np.concatenate(
            (np.zeros((1, CONTINUOUS_DIM), dtype=np.float64), action_bank), axis=0
        ),
        action_full=np.asarray(full_actions, dtype=np.float64),
        **packed
    )
    return mark_complete(
        path,
        {
            "kind": "stage3_common_action_bank_effects",
            "task_id": record["task_id"],
            "episode_id": int(record["episode_id"]),
            "split": record["split"],
            "phase": record["phase"],
            "branches": len(rollouts),
            "valid_bank_size": ACTION_BANK_SIZE,
            "created_utc": utc_now(),
        },
    )


def _confirmation_collection_allowed(output_root):
    path = os.path.join(output_root, "development_gate.json")
    if not os.path.isfile(path):
        return False
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return bool(payload.get("method_settings_frozen_before_holdout"))


def collect_task(project_root, paths, output_root, task_id, splits, scratch_root=SCRATCH_ROOT):
    """Collect all required target/bank branches for one task and split set."""
    requested = set(str(value) for value in splits)
    allowed = {"train", "calibration", "development", "confirmation"}
    if not requested or not requested.issubset(allowed):
        raise ValueError("invalid Stage 3 split request")
    if "confirmation" in requested and not _confirmation_collection_allowed(output_root):
        raise RuntimeError("Stage 3 holdout remains locked until method settings are frozen")

    records = [
        row
        for row in _load_records(output_root)
        if row["task_id"] == task_id and row["split"] in requested
    ]
    records.sort(key=lambda row: (int(row["episode_id"]), PHASES.index(row["phase"])))
    codebooks = _load_codebooks(output_root)
    action_bank = _load_action_bank(project_root)
    runtime = LiberoTaskRuntime(_task(task_id), paths["libero_source"], paths["dataset_root"])
    completed = []
    current_episode_id = None
    episode = None
    try:
        for record in records:
            split = record["split"]
            episode_id = int(record["episode_id"])
            phase = record["phase"]
            context_path = context_shard(scratch_root, task_id, episode_id, phase)
            support_path = resolved_support_shard(
                project_root, scratch_root, split, task_id, episode_id, phase
            )
            candidate_path = resolved_candidate_shard(
                project_root, scratch_root, split, task_id, episode_id, phase
            )
            context_valid, context_evidence = validate_complete(context_path)
            support_valid, support_evidence = validate_complete(support_path)
            candidate_valid, candidate_evidence = validate_complete(candidate_path)
            # Train support is always reused.  Train candidates 24-31 are reused.
            need_support = split != "train" and not support_valid
            need_candidate = not candidate_valid
            need_context = not context_valid
            if not (need_context or need_support or need_candidate):
                completed.append(
                    {
                        "key": record["key"],
                        "status": "resumed",
                        "context": context_evidence,
                        "support": support_evidence,
                        "candidate": candidate_evidence,
                    }
                )
                continue
            if current_episode_id != episode_id:
                episode = runtime.load_episode(episode_id)
                runtime.initialize_episode_model(episode)
                current_episode_id = episode_id
            index = int(record["snapshot_index"])
            base_actions = np.asarray(
                episode["actions"][index : index + config.CHUNK_HORIZON], dtype=np.float64
            )
            if need_context:
                arrays = _context_arrays(runtime, episode, record, base_actions)
                context_marker = _write_context(context_path, record, arrays)
            else:
                context_marker = context_path + ".complete.json"
            snapshot = None
            nominal = None
            if need_support or need_candidate:
                snapshot = runtime.snapshot_from_recorded_state(
                    episode["states"][index], episode["actions"][:index]
                )
                nominal = runtime.execute_chunk(snapshot, base_actions)
            if need_support:
                support_marker = _write_support(
                    support_path,
                    record,
                    base_actions,
                    codebooks[split],
                    nominal,
                    runtime,
                    snapshot,
                )
            else:
                support_marker = support_path + ".complete.json"
            if need_candidate:
                candidate_marker = _write_candidates(
                    candidate_path,
                    record,
                    base_actions,
                    action_bank,
                    nominal,
                    runtime,
                    snapshot,
                )
            else:
                candidate_marker = candidate_path + ".complete.json"
            completed.append(
                {
                    "key": record["key"],
                    "status": "created",
                    "context_marker": context_marker,
                    "support_marker": support_marker,
                    "candidate_marker": candidate_marker,
                }
            )
            print(
                "STAGE3_COLLECTION_COMPLETE task=%s episode=%d phase=%s split=%s support=%s candidate=%s"
                % (task_id, episode_id, phase, split, need_support, need_candidate),
                flush=True,
            )
    finally:
        runtime.close()

    manifest_path = os.path.join(
        output_root,
        "work_manifests",
        "collection_%s_%s.json" % (task_id, "_".join(sorted(requested))),
    )
    atomic_json(
        manifest_path,
        {
            "created_utc": utc_now(),
            "task_id": task_id,
            "splits": sorted(requested),
            "scratch_root": scratch_root,
            "records": completed,
        },
    )
    return {"manifest": manifest_path, "records": len(completed)}


def verify_training_reuse(project_root, output_root, scratch_root=SCRATCH_ROOT):
    """Validate every reused and newly collected training shard marker."""
    failures = []
    rows = []
    for record in _load_records(output_root):
        if record["split"] != "train":
            continue
        task_id = record["task_id"]
        episode_id = int(record["episode_id"])
        phase = record["phase"]
        paths = {
            "support": resolved_support_shard(
                project_root, scratch_root, "train", task_id, episode_id, phase
            ),
            "candidate": resolved_candidate_shard(
                project_root, scratch_root, "train", task_id, episode_id, phase
            ),
            "context": context_shard(scratch_root, task_id, episode_id, phase),
        }
        evidence = {}
        for kind, path in paths.items():
            valid, detail = validate_complete(path)
            evidence[kind] = {"path": path, "valid": bool(valid), "detail": detail}
            if not valid:
                failures.append({"key": record["key"], "kind": kind, "detail": detail})
        rows.append({"key": record["key"], "artifacts": evidence})
    destination = os.path.join(output_root, "training_reuse_validation.json")
    atomic_json(
        destination,
        {
            "created_utc": utc_now(),
            "states": len(rows),
            "failed": len(failures),
            "passed": not failures,
            "failures": failures,
            "rows": rows,
        },
    )
    if failures:
        raise RuntimeError("Stage 3 training reuse validation failed")
    return {"path": destination, "states": len(rows), "passed": True}
