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
        {"created_utc": utc_now(), "task_id": task_id, "shards": completed, "count": len(completed)},
    )
    return completed

