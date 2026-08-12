"""Execute decoded held-out action chunks from identical LIBERO snapshots."""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from . import config
from .env_adapter import LiberoTaskRuntime
from .pipeline import _pack_rollouts, utc_now
from .storage import atomic_json, atomic_npz, mark_complete, validate_complete


def _plan_paths(output_root, task_id=None):
    pattern = os.path.join(
        output_root,
        "work",
        "quantization_plans",
        task_id if task_id else "*",
        "*.npz",
    )
    return sorted(glob.glob(pattern))


def _quantized_path(output_root, task_id, plan_path):
    return os.path.join(
        output_root,
        "work",
        "quantized_shards",
        task_id,
        os.path.basename(plan_path),
    )


def _validate_resumed_shard_against_plan(plan_path, destination):
    """Require a resumed realized-rollout shard to match the current frozen plan."""
    copied_fields = (
        "task_id",
        "episode_id",
        "split",
        "phase",
        "snapshot_index",
        "methods",
        "k",
        "direction",
        "sign",
        "radius",
        "code_index",
        "decoded_actions",
        "original_actions",
        "original_immediate",
        "original_settled",
        "original_mask",
        "original_contact_mode",
        "original_settled_progress",
        "original_settled_success",
    )
    with np.load(plan_path, allow_pickle=False) as plan, np.load(destination, allow_pickle=False) as shard:
        missing = [name for name in copied_fields if name not in plan.files or name not in shard.files]
        if missing:
            raise RuntimeError("resumed quantized shard is missing plan fields: %s" % missing)
        mismatched = []
        for name in copied_fields:
            planned = np.asarray(plan[name])
            realized = np.asarray(shard[name])
            equal = (
                np.array_equal(planned, realized, equal_nan=True)
                if planned.dtype.kind in "fc" and realized.dtype.kind in "fc"
                else np.array_equal(planned, realized)
            )
            if not equal:
                mismatched.append(name)
    if mismatched:
        raise RuntimeError(
            "resumed quantized shard does not match current frozen plan %s: %s"
            % (plan_path, mismatched)
        )


def collect_quantized(paths, output_root, task_id=None, plan_limit=None):
    tasks = [task for task in config.TASKS if task_id is None or task["task_id"] == task_id]
    if not tasks:
        raise KeyError(task_id)
    completed = []
    for task in tasks:
        runtime = LiberoTaskRuntime(task, paths["libero_source"], paths["dataset_root"])
        try:
            plans = _plan_paths(output_root, task["task_id"])
            if plan_limit is not None:
                plans = plans[: int(plan_limit)]
            for plan_path in plans:
                valid, evidence = validate_complete(plan_path)
                if not valid:
                    raise RuntimeError("invalid plan %s: %s" % (plan_path, evidence))
                destination = _quantized_path(output_root, task["task_id"], plan_path)
                valid, evidence = validate_complete(destination)
                if valid:
                    _validate_resumed_shard_against_plan(plan_path, destination)
                    completed.append({"path": destination, "status": "resumed", "evidence": evidence})
                    continue
                with np.load(plan_path, allow_pickle=False) as plan:
                    episode_id = int(plan["episode_id"].item())
                    snapshot_index = int(plan["snapshot_index"].item())
                    decoded_actions = np.asarray(plan["decoded_actions"], dtype=np.float64)
                    copied = {name: np.asarray(plan[name]).copy() for name in plan.files if name != "decoded_actions"}
                episode = runtime.load_episode(episode_id)
                runtime.initialize_episode_model(episode)
                snapshot = runtime.snapshot_from_recorded_state(
                    episode["states"][snapshot_index], episode["actions"][:snapshot_index]
                )
                rollouts = []
                for row, action in enumerate(decoded_actions):
                    rollouts.append(runtime.execute_chunk(snapshot, action))
                    if (row + 1) % 64 == 0:
                        print(
                            "QUANTIZED_PROGRESS task=%s plan=%s rows=%d/%d"
                            % (task["task_id"], os.path.basename(plan_path), row + 1, len(decoded_actions)),
                            flush=True,
                        )
                packed = _pack_rollouts(rollouts)
                atomic_npz(destination, decoded_actions=decoded_actions, **copied, **packed)
                marker = mark_complete(
                    destination,
                    {
                        "task_id": task["task_id"],
                        "episode_id": episode_id,
                        "snapshot_index": snapshot_index,
                        "rows": len(decoded_actions),
                        "source_plan": plan_path,
                        "created_utc": utc_now(),
                        "pai_run_id": os.environ.get("PAI_CANARY_RUN_ID"),
                        "pai_nonce": os.environ.get("PAI_CANARY_NONCE"),
                    },
                )
                completed.append({"path": destination, "marker": marker, "status": "created"})
                print(
                    "QUANTIZED_SHARD_COMPLETE task=%s plan=%s rows=%d"
                    % (task["task_id"], os.path.basename(plan_path), len(decoded_actions)),
                    flush=True,
                )
        finally:
            runtime.close()
    manifest_path = os.path.join(
        output_root,
        "work",
        "quantized_collection_%s.json" % (task_id or "all"),
    )
    atomic_json(
        manifest_path,
        {
            "created_utc": utc_now(),
            "pai_run_id": os.environ.get("PAI_CANARY_RUN_ID"),
            "pai_nonce": os.environ.get("PAI_CANARY_NONCE"),
            "task_id": task_id,
            "shards": completed,
            "count": len(completed),
            "resume_validation": "payload_hash_and_exact_current_plan_arrays",
        },
    )
    return completed
