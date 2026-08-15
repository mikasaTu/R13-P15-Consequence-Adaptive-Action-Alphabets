"""Resumable CPU simulator collection for expanded Stage 4 train states."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter

import numpy as np

from . import config
from .env_adapter import LiberoTaskRuntime
from .pipeline import _compare_rollouts, utc_now
from .stage2 import _array_hash
from .stage3_collection import _context_arrays, _write_candidates, _write_context, _write_support
from .stage4_config import (
    ACTION_BANK_SIZE,
    HISTORICAL_REPOSITORY_ROOT,
    HISTORICAL_STAGE3_RELATIVE,
    OUTPUT_RELATIVE,
    SCRATCH_ROOT,
    STAGE2_ACTION_BANK_RELATIVE,
    SUPPORT_DIRECTION_COUNT,
    SUPPORT_RADII,
    SUPPORT_TARGET_COUNT,
    TASKS,
    TRAIN_EPISODES,
)
from .storage import atomic_json, sha256_file, validate_complete


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _task(task_id):
    for task in TASKS:
        if task["task_id"] == task_id:
            return dict(task)
    raise KeyError(task_id)


def _stage4_output(project_root, output_root=None):
    return output_root or os.path.join(os.path.abspath(project_root), OUTPUT_RELATIVE)


def _paths_for_key(key, scratch_root=SCRATCH_ROOT):
    return {
        kind: os.path.join(scratch_root, kind + "_shards", "train", key + ".npz")
        for kind in ("context", "support", "candidate")
    }


def _load_banks(output_root):
    with np.load(os.path.join(output_root, "training_support_bank.npz"), allow_pickle=False) as data:
        support = {name: np.asarray(data[name]).copy() for name in data.files}
    support_for_writer = {
        "directions": np.asarray(support["directions"], dtype=np.float64),
        "radii": np.tile(
            np.asarray(SUPPORT_RADII, dtype=np.float64)[None, :],
            (SUPPORT_DIRECTION_COUNT, 1),
        ),
        "family_id": np.asarray(support["direction_family_id"], dtype=np.int8),
    }
    if support["residuals"].shape != (SUPPORT_TARGET_COUNT, 24):
        raise RuntimeError("Stage 4 support bank shape changed")
    action_path = os.path.join(HISTORICAL_REPOSITORY_ROOT, STAGE2_ACTION_BANK_RELATIVE)
    with np.load(action_path, allow_pickle=False) as data:
        action = np.asarray(data["residuals"], dtype=np.float64)
    if action.shape != (ACTION_BANK_SIZE, 24):
        raise RuntimeError("Stage 2 action bank shape changed")
    return support_for_writer, action


def _worker_unit(record):
    task_index = next(
        index for index, task in enumerate(TASKS) if task["task_id"] == record["task_id"]
    )
    return task_index * len(TRAIN_EPISODES) + (
        int(record["episode_id"]) - min(TRAIN_EPISODES)
    )


def validate_sacrificial_replay(
    project_root,
    output_root=None,
    tolerance=1e-12,
):
    """Validate A/B/A restore ordering only on calibration episode 32."""
    project_root = os.path.abspath(project_root)
    output_root = _stage4_output(project_root, output_root)
    paths = config.resolved_paths()
    stage3_split = _load_json(
        os.path.join(
            HISTORICAL_REPOSITORY_ROOT,
            HISTORICAL_STAGE3_RELATIVE,
            "episode_split.json",
        )
    )
    _, action_bank = _load_banks(output_root)
    tests = []
    for task_index, task in enumerate(TASKS):
        record = next(
            row
            for row in stage3_split["snapshots"]
            if row["task_id"] == task["task_id"]
            and int(row["episode_id"]) == 32
            and row["phase"] == "pre_contact"
        )
        runtime = LiberoTaskRuntime(task, paths["libero_source"], paths["dataset_root"])
        try:
            episode = runtime.load_episode(32)
            runtime.initialize_episode_model(episode)
            index = int(record["snapshot_index"])
            base = np.asarray(
                episode["actions"][index : index + config.CHUNK_HORIZON],
                dtype=np.float64,
            )
            base_continuous = runtime.continuous_chunk(base)
            bank_index = int((task_index * 53 + 17) % ACTION_BANK_SIZE)
            perturbed_continuous = base_continuous + action_bank[bank_index]
            if float(np.max(np.abs(perturbed_continuous))) > 1.0 + 1e-12:
                valid = np.flatnonzero(
                    np.max(np.abs(base_continuous[None, :] + action_bank), axis=1)
                    <= 1.0 + 1e-12
                )
                bank_index = int(valid[0])
                perturbed_continuous = base_continuous + action_bank[bank_index]
            perturbed = runtime.replace_continuous_chunk(base, perturbed_continuous)
            snapshot = runtime.snapshot_from_recorded_state(
                episode["states"][index], episode["actions"][:index]
            )
            nominal_left = runtime.execute_chunk(snapshot, base)
            perturb_left = runtime.execute_chunk(snapshot, perturbed)
            nominal_right = runtime.execute_chunk(snapshot, base)
            perturb_right = runtime.execute_chunk(snapshot, perturbed)
            nominal_passed, nominal_metrics = _compare_rollouts(
                nominal_left, nominal_right, tolerance
            )
            perturb_passed, perturb_metrics = _compare_rollouts(
                perturb_left, perturb_right, tolerance
            )
            tests.append(
                {
                    "task_id": task["task_id"],
                    "episode_id": 32,
                    "phase": "pre_contact",
                    "snapshot_index": index,
                    "candidate_bank_index": bank_index,
                    "order": "nominal,perturbed,nominal,perturbed",
                    "nominal": {"passed": nominal_passed, **nominal_metrics},
                    "perturbed": {"passed": perturb_passed, **perturb_metrics},
                    "passed": bool(nominal_passed and perturb_passed),
                }
            )
        finally:
            runtime.close()
    result = {
        "scope": "sacrificial calibration episode 32 only",
        "confirmation_state_executed": False,
        "fresh_confirmation_state_executed": False,
        "tolerance": tolerance,
        "tests": tests,
        "failed_tests": [row for row in tests if not row["passed"]],
        "passed": bool(all(row["passed"] for row in tests)),
    }
    destination = os.path.join(output_root, "snapshot_restore_validation.json")
    atomic_json(destination, result)
    if not result["passed"]:
        raise RuntimeError("sacrificial snapshot replay validation failed")
    return result


def collect_worker(
    project_root,
    output_root=None,
    worker_index=0,
    worker_count=1,
    limit=None,
    scratch_root=SCRATCH_ROOT,
):
    """Collect complete context/support/candidate shards for assigned episodes."""
    project_root = os.path.abspath(project_root)
    output_root = _stage4_output(project_root, output_root)
    replay_path = os.path.join(output_root, "snapshot_restore_validation.json")
    if not os.path.isfile(replay_path) or not _load_json(replay_path).get("passed"):
        raise RuntimeError("sacrificial replay validation has not passed")
    manifest_path = os.path.join(output_root, "TRAINING_STATE_MANIFEST.json")
    manifest = _load_json(manifest_path)
    support_bank, action_bank = _load_banks(output_root)
    paths = config.resolved_paths()
    records = [
        row
        for row in manifest["records"]
        if _worker_unit(row) % int(worker_count) == int(worker_index)
    ]
    records.sort(
        key=lambda row: (
            next(i for i, task in enumerate(TASKS) if task["task_id"] == row["task_id"]),
            int(row["episode_id"]),
            int(row["snapshot_index"]),
        )
    )
    if limit is not None:
        records = records[: int(limit)]
    completed = []
    current_task = None
    current_episode = None
    runtime = None
    episode = None
    try:
        for ordinal, record in enumerate(records):
            shard_paths = _paths_for_key(record["key"], scratch_root)
            validation = {
                kind: validate_complete(path) for kind, path in shard_paths.items()
            }
            if all(value[0] for value in validation.values()):
                completed.append(
                    {
                        "key": record["key"],
                        "status": "resumed",
                        "paths": shard_paths,
                    }
                )
                continue
            if current_task != record["task_id"]:
                if runtime is not None:
                    runtime.close()
                runtime = LiberoTaskRuntime(
                    _task(record["task_id"]),
                    paths["libero_source"],
                    paths["dataset_root"],
                )
                current_task = record["task_id"]
                current_episode = None
            if current_episode != int(record["episode_id"]):
                episode = runtime.load_episode(record["episode_id"])
                runtime.initialize_episode_model(episode)
                current_episode = int(record["episode_id"])
            index = int(record["snapshot_index"])
            state = np.asarray(episode["states"][index], dtype=np.float64)
            base = np.asarray(
                episode["actions"][index : index + config.CHUNK_HORIZON],
                dtype=np.float64,
            )
            if _array_hash(state) != record["snapshot_state_sha256"]:
                raise RuntimeError("frozen state hash mismatch for " + record["key"])
            if _array_hash(base) != record["base_action_sha256"]:
                raise RuntimeError("frozen base-action hash mismatch for " + record["key"])
            snapshot = runtime.snapshot_from_recorded_state(
                state, episode["actions"][:index]
            )
            nominal = runtime.execute_chunk(snapshot, base)
            if not validation["context"][0]:
                arrays = _context_arrays(
                    runtime,
                    episode,
                    record,
                    base,
                    nominal["initial"]["vector"],
                    nominal["initial"]["mask"],
                    nominal["initial"]["contacts"]["relevant"],
                )
                _write_context(shard_paths["context"], record, arrays)
            if not validation["support"][0]:
                _write_support(
                    shard_paths["support"],
                    record,
                    base,
                    support_bank,
                    nominal,
                    runtime,
                    snapshot,
                )
            if not validation["candidate"][0]:
                _write_candidates(
                    shard_paths["candidate"],
                    record,
                    base,
                    action_bank,
                    nominal,
                    runtime,
                    snapshot,
                )
            post = {
                kind: validate_complete(path) for kind, path in shard_paths.items()
            }
            if not all(value[0] for value in post.values()):
                raise RuntimeError("incomplete shard after write for " + record["key"])
            completed.append(
                {"key": record["key"], "status": "created", "paths": shard_paths}
            )
            print(
                "STAGE4_TRAIN_STATE_COMPLETE worker=%d/%d ordinal=%d/%d key=%s"
                % (
                    int(worker_index),
                    int(worker_count),
                    ordinal + 1,
                    len(records),
                    record["key"],
                ),
                flush=True,
            )
    finally:
        if runtime is not None:
            runtime.close()
    worker_manifest = os.path.join(
        scratch_root,
        "worker_manifests",
        "train_worker_%02d_of_%02d.json" % (int(worker_index), int(worker_count)),
    )
    atomic_json(
        worker_manifest,
        {
            "created_utc": utc_now(),
            "worker_index": int(worker_index),
            "worker_count": int(worker_count),
            "requested_records": len(records),
            "completed_records": len(completed),
            "training_state_manifest_sha256": sha256_file(manifest_path),
            "records": completed,
        },
    )
    return {
        "worker_manifest": worker_manifest,
        "records": len(completed),
        "worker_index": int(worker_index),
        "worker_count": int(worker_count),
    }


def verify_collection(project_root, output_root=None, scratch_root=SCRATCH_ROOT):
    project_root = os.path.abspath(project_root)
    output_root = _stage4_output(project_root, output_root)
    manifest_path = os.path.join(output_root, "TRAINING_STATE_MANIFEST.json")
    manifest = _load_json(manifest_path)
    files = []
    failures = []
    branch_counts = Counter()
    combined = hashlib.sha256()
    for record in manifest["records"]:
        for kind, path in sorted(_paths_for_key(record["key"], scratch_root).items()):
            valid, evidence = validate_complete(path)
            if not valid:
                failures.append(
                    {"key": record["key"], "kind": kind, "reason": evidence}
                )
                continue
            digest = sha256_file(path)
            relative = os.path.relpath(path, scratch_root).replace(os.sep, "/")
            combined.update(relative.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\0")
            with np.load(path, allow_pickle=False) as data:
                if kind == "context":
                    branches = 1
                else:
                    branches = int(len(data["residual_action"]))
                    expected = SUPPORT_TARGET_COUNT + 1 if kind == "support" else ACTION_BANK_SIZE + 1
                    if branches != expected:
                        failures.append(
                            {
                                "key": record["key"],
                                "kind": kind,
                                "reason": "branch_count_%d_expected_%d" % (branches, expected),
                            }
                        )
                branch_counts[kind] += branches
            files.append(
                {
                    "key": record["key"],
                    "kind": kind,
                    "relative_path": relative,
                    "bytes": int(os.path.getsize(path)),
                    "sha256": digest,
                }
            )
    result = {
        "schema_version": "stage4-expanded-training-collection-v1",
        "scratch_root": scratch_root,
        "training_state_manifest_sha256": sha256_file(manifest_path),
        "expected_states": manifest["state_count"],
        "complete_states": int(len(files) // 3),
        "file_count": len(files),
        "branch_counts": dict(sorted(branch_counts.items())),
        "combined_path_sha256_digest": combined.hexdigest(),
        "failures": failures,
        "passed": not failures and len(files) == 3 * int(manifest["state_count"]),
        "files": files,
    }
    destination = os.path.join(output_root, "expanded_training_collection.json")
    atomic_json(destination, result)
    if not result["passed"]:
        raise RuntimeError("expanded training collection incomplete")
    return result


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-replay", "collect", "verify"))
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--scratch-root", default=SCRATCH_ROOT)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    if args.command == "validate-replay":
        result = validate_sacrificial_replay(args.project_root, args.output_root)
    elif args.command == "collect":
        result = collect_worker(
            args.project_root,
            args.output_root,
            args.worker_index,
            args.worker_count,
            args.limit,
            args.scratch_root,
        )
    else:
        result = verify_collection(args.project_root, args.output_root, args.scratch_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
