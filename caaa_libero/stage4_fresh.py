"""Fresh-perturbed-state firewall, collection, and cache construction.

The split generator reads recorded HDF5 arrays only.  Simulator execution is
locked until the exact JSON split is present in the current Git HEAD.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter

import h5py
import numpy as np

from . import config
from .env_adapter import LiberoTaskRuntime
from .stage2 import _array_hash
from .stage3_collection import _context_arrays, _write_candidates, _write_context, _write_support
from .stage4_collection import _load_banks
from .stage4_config import (
    ACTION_BANK_SIZE,
    CONFIRMATION_SELECTION_SEED,
    HISTORICAL_EXPLORATORY_EPISODES,
    HISTORICAL_REPOSITORY_ROOT,
    HISTORICAL_STAGE3_RELATIVE,
    HORIZON,
    OUTPUT_RELATIVE,
    PHASES,
    SCRATCH_ROOT,
    SUPPORT_TARGET_COUNT,
    TASKS,
)
from .stage4_data import _cache_arrays
from .stage4_freeze import _demo_path, _phase_windows, _valid_action_chunk
from .storage import (
    atomic_json,
    atomic_npz,
    mark_complete,
    sha256_file,
    validate_complete,
)


FRESH_LABEL = "FRESH_PERTURBED_STATE_CONFIRMATION"
PERTURBATION_MAGNITUDE_RAD = 0.0015


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _seed(*parts):
    value = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")


def _task(task_id):
    return next(dict(value) for value in TASKS if value["task_id"] == task_id)


def _stage3_snapshots():
    path = os.path.join(
        HISTORICAL_REPOSITORY_ROOT,
        HISTORICAL_STAGE3_RELATIVE,
        "episode_split.json",
    )
    return path, _load_json(path)["snapshots"]


def _method_freeze(output_root):
    required = (
        "MODEL_SELECTION.json",
        "DEVELOPMENT_RETRIEVAL.csv",
        "DEVELOPMENT_REALIZED.csv",
        "DEVELOPMENT_GATE.json",
        "HISTORICAL_EXPLORATORY_RETRIEVAL.csv",
        "HISTORICAL_EXPLORATORY_REALIZED.csv",
        "context_reversal_evaluation.json",
    )
    missing = [name for name in required if not os.path.isfile(os.path.join(output_root, name))]
    if missing:
        raise RuntimeError("development method freeze incomplete: " + ",".join(missing))
    selection = _load_json(os.path.join(output_root, "MODEL_SELECTION.json"))
    if "cr_c3_controls" not in selection:
        raise RuntimeError("matched controls have not been frozen")
    if "selected_L" not in selection.get("trust_region_selection", {}):
        raise RuntimeError("trust region has not been calibrated")
    checkpoints = []
    for family in selection["cr_c3_selection"]["family_trace"]:
        checkpoints.extend(family["checkpoints"])
    for control in selection["cr_c3_controls"]["controls"]:
        checkpoints.extend(control["checkpoints"])
    checkpoint_hashes = {
        entry["path"]: entry["sha256"] for entry in checkpoints
    }
    for relative, expected in checkpoint_hashes.items():
        path = os.path.join(output_root, relative)
        if sha256_file(path) != expected:
            raise RuntimeError("method checkpoint changed: " + relative)
    return {
        "selection_sha256": sha256_file(os.path.join(output_root, "MODEL_SELECTION.json")),
        "development_gate_sha256": sha256_file(os.path.join(output_root, "DEVELOPMENT_GATE.json")),
        "development_retrieval_sha256": sha256_file(
            os.path.join(output_root, "DEVELOPMENT_RETRIEVAL.csv")
        ),
        "development_realized_sha256": sha256_file(
            os.path.join(output_root, "DEVELOPMENT_REALIZED.csv")
        ),
        "historical_exploratory_retrieval_sha256": sha256_file(
            os.path.join(output_root, "HISTORICAL_EXPLORATORY_RETRIEVAL.csv")
        ),
        "historical_exploratory_realized_sha256": sha256_file(
            os.path.join(output_root, "HISTORICAL_EXPLORATORY_REALIZED.csv")
        ),
        "checkpoint_hashes": dict(sorted(checkpoint_hashes.items())),
        "selected_family": selection["cr_c3_selection"]["selected_family"],
        "selected_L": int(selection["trust_region_selection"]["selected_L"]),
        "selected_gamma_analysis_only": float(
            selection["bounded_correction_selection"]["selected_gamma"]
        ),
        "primary_method": "CR_TR_C3_K64",
        "atlas_algorithm": "deterministic predicted-space k-medoids",
        "K": 64,
        "metrics_and_thresholds": "METHOD_DEFINITIONS.json",
    }


def freeze_fresh_split(project_root, output_root=None):
    """Freeze 160 exact perturbed vectors without creating a simulator."""
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    destination = os.path.join(output_root, "FRESH_CONFIRMATION_SPLIT.json")
    if os.path.exists(destination):
        raise RuntimeError("fresh split already exists; refusing to overwrite")
    method_freeze = _method_freeze(output_root)
    stage3_path, snapshots = _stage3_snapshots()
    used = {
        (row["task_id"], int(row["episode_id"]), int(row["snapshot_index"]))
        for row in snapshots
    }
    by_episode = {}
    for row in snapshots:
        key = (row["task_id"], int(row["episode_id"]))
        by_episode.setdefault(key, {})[row["phase"]] = int(row["snapshot_index"])
    with np.load(os.path.join(output_root, "training_support_bank.npz"), allow_pickle=False) as data:
        support_residual = np.asarray(data["residuals"], dtype=np.float64)
    action_path = os.path.join(
        HISTORICAL_REPOSITORY_ROOT,
        "experiments/r13_p15_ncea/stage2/action_bank.npz",
    )
    with np.load(action_path, allow_pickle=False) as data:
        action_bank = np.asarray(data["residuals"], dtype=np.float64)
    records = []
    demo_hashes = {}
    for task in TASKS:
        task_id = task["task_id"]
        demo_path = _demo_path(task)
        demo_hashes[task_id] = sha256_file(demo_path)
        with h5py.File(demo_path, "r") as handle:
            for episode_id in HISTORICAL_EXPLORATORY_EPISODES:
                group = handle["data"]["demo_%d" % int(episode_id)]
                actions = np.asarray(group["actions"], dtype=np.float64)
                states = np.asarray(group["states"], dtype=np.float64)
                anchors = by_episode[(task_id, int(episode_id))]
                windows = _phase_windows(anchors, len(actions) - HORIZON)
                for phase in PHASES:
                    low, high = windows[phase]
                    candidates = []
                    for index in range(low, high + 1):
                        if (task_id, int(episode_id), index) in used:
                            continue
                        valid, continuous = _valid_action_chunk(
                            actions, index, action_bank, support_residual
                        )
                        if not valid:
                            continue
                        tie = _seed(
                            CONFIRMATION_SELECTION_SEED,
                            task_id,
                            episode_id,
                            phase,
                            index,
                        )
                        candidates.append((tie, index, continuous))
                    if not candidates:
                        raise RuntimeError(
                            "no executable unused fresh timestep for %s/e%d/%s"
                            % (task_id, episode_id, phase)
                        )
                    _, index, continuous = min(candidates, key=lambda value: value[0])
                    base_state = np.asarray(states[index], dtype=np.float64)
                    perturbed = base_state.copy()
                    joint_id = _seed(
                        CONFIRMATION_SELECTION_SEED, task_id, episode_id, phase, "joint"
                    ) % 7
                    sign = -1 if _seed(
                        CONFIRMATION_SELECTION_SEED, task_id, episode_id, phase, "sign"
                    ) % 2 else 1
                    flat_index = 1 + int(joint_id)
                    perturbed[flat_index] += sign * PERTURBATION_MAGNITUDE_RAD
                    base_actions = np.asarray(
                        actions[index : index + HORIZON], dtype=np.float64
                    )
                    key = "%s__e%02d__%s__freshp" % (task_id, episode_id, phase)
                    records.append(
                        {
                            "key": key,
                            "split": "fresh_confirmation",
                            "evidence_label": FRESH_LABEL,
                            "new_episode_claim": False,
                            "task_id": task_id,
                            "episode_id": int(episode_id),
                            "phase": phase,
                            "snapshot_index": int(index),
                            "base_recorded_state": base_state.tolist(),
                            "perturbed_state": perturbed.tolist(),
                            "base_recorded_state_sha256": _array_hash(base_state),
                            "snapshot_state_sha256": _array_hash(perturbed),
                            "base_action_sha256": _array_hash(base_actions),
                            "state_vector_length": len(perturbed),
                            "perturbation": {
                                "kind": "panda_joint_qpos_offset",
                                "flattened_state_index": flat_index,
                                "panda_joint_index": int(joint_id),
                                "delta_radians": sign * PERTURBATION_MAGNITUDE_RAD,
                                "l2": PERTURBATION_MAGNITUDE_RAD,
                                "bounded": True,
                            },
                            "max_abs_base_continuous": float(np.max(np.abs(continuous))),
                            "candidate_bank_executable_without_clipping": True,
                            "support_bank_executable_without_clipping": True,
                            "source_demo_sha256": demo_hashes[task_id],
                        }
                    )
    if len(records) != len(TASKS) * len(HISTORICAL_EXPLORATORY_EPISODES) * len(PHASES):
        raise AssertionError(len(records))
    payload = {
        "schema_version": "stage4-fresh-perturbed-confirmation-v1",
        "evidence_label": FRESH_LABEL,
        "is_new_episode_claim": False,
        "source_priority": 3,
        "source_1_status": "UNAVAILABLE_OFFICIAL_FILES_END_AT_DEMO_49",
        "source_2_status": "NO_FROZEN_NOMINAL_GENERATOR_AVAILABLE",
        "selection_seed": CONFIRMATION_SELECTION_SEED,
        "selection_rule": (
            "For each task, episode 40-49 and phase, select one hash-minimum "
            "unused executable H=4 timestep inside the frozen phase window; "
            "apply an exact +/-0.0015 rad offset to one hash-selected Panda "
            "arm qpos coordinate in the serialized state."
        ),
        "state_perturbation_generated_without_simulator": True,
        "simulator_branch_execution_before_this_file": False,
        "sacrificial_replay_evidence": "snapshot_restore_validation.json",
        "stage3_snapshot_manifest_sha256": sha256_file(stage3_path),
        "action_bank_sha256": sha256_file(action_path),
        "support_bank_sha256": sha256_file(
            os.path.join(output_root, "training_support_bank.npz")
        ),
        "method_freeze": method_freeze,
        "state_count": len(records),
        "task_counts": dict(sorted(Counter(row["task_id"] for row in records).items())),
        "phase_counts": dict(sorted(Counter(row["phase"] for row in records).items())),
        "episode_cluster_count": len(
            {(row["task_id"], row["episode_id"]) for row in records}
        ),
        "records": records,
    }
    atomic_json(destination, payload)
    return {
        "path": destination,
        "sha256": sha256_file(destination),
        "state_count": len(records),
        "next_required_action": "commit this exact file before simulator execution",
    }


def _fresh_paths(key, scratch_root=SCRATCH_ROOT):
    return {
        kind: os.path.join(
            scratch_root, "fresh_%s_shards" % kind, key + ".npz"
        )
        for kind in ("context", "support", "candidate")
    }


def _assert_split_committed(project_root, split_path):
    relative = os.path.relpath(split_path, project_root).replace(os.sep, "/")
    committed = subprocess.check_output(
        ["git", "show", "HEAD:" + relative], cwd=project_root
    )
    with open(split_path, "rb") as handle:
        working = handle.read()
    if committed != working:
        raise RuntimeError("fresh split must be byte-identical to current Git HEAD")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()


def _worker_unit(record):
    task_index = next(
        index for index, value in enumerate(TASKS) if value["task_id"] == record["task_id"]
    )
    return task_index * len(HISTORICAL_EXPLORATORY_EPISODES) + (
        int(record["episode_id"]) - min(HISTORICAL_EXPLORATORY_EPISODES)
    )


def collect_fresh_worker(
    project_root,
    output_root=None,
    worker_index=0,
    worker_count=1,
    scratch_root=SCRATCH_ROOT,
):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    split_path = os.path.join(output_root, "FRESH_CONFIRMATION_SPLIT.json")
    freeze_commit = _assert_split_committed(project_root, split_path)
    split = _load_json(split_path)
    replay = _load_json(os.path.join(output_root, "snapshot_restore_validation.json"))
    if not replay.get("passed") or replay.get("confirmation_state_executed"):
        raise RuntimeError("separate sacrificial replay firewall failed")
    support_bank, action_bank = _load_banks(output_root)
    paths = config.resolved_paths()
    records = [
        row
        for row in split["records"]
        if _worker_unit(row) % int(worker_count) == int(worker_index)
    ]
    completed = []
    runtime = None
    current_task = None
    current_episode = None
    episode = None
    try:
        for ordinal, record in enumerate(records):
            shard_paths = _fresh_paths(record["key"], scratch_root)
            valid = {kind: validate_complete(path)[0] for kind, path in shard_paths.items()}
            if all(valid.values()):
                completed.append({"key": record["key"], "status": "resumed"})
                continue
            if current_task != record["task_id"]:
                if runtime is not None:
                    runtime.close()
                runtime = LiberoTaskRuntime(
                    _task(record["task_id"]), paths["libero_source"], paths["dataset_root"]
                )
                current_task = record["task_id"]
                current_episode = None
            if current_episode != int(record["episode_id"]):
                episode = runtime.load_episode(record["episode_id"])
                runtime.initialize_episode_model(episode)
                current_episode = int(record["episode_id"])
            index = int(record["snapshot_index"])
            state = np.asarray(record["perturbed_state"], dtype=np.float64)
            base = np.asarray(
                episode["actions"][index : index + config.CHUNK_HORIZON], dtype=np.float64
            )
            if _array_hash(state) != record["snapshot_state_sha256"]:
                raise RuntimeError("fresh state hash mismatch")
            if _array_hash(base) != record["base_action_sha256"]:
                raise RuntimeError("fresh base action hash mismatch")
            snapshot = runtime.snapshot_from_recorded_state(
                state, episode["actions"][:index]
            )
            nominal = runtime.execute_chunk(snapshot, base)
            if not valid["context"]:
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
            if not valid["support"]:
                _write_support(
                    shard_paths["support"], record, base, support_bank, nominal, runtime, snapshot
                )
            if not valid["candidate"]:
                _write_candidates(
                    shard_paths["candidate"], record, base, action_bank, nominal, runtime, snapshot
                )
            post = {kind: validate_complete(path)[0] for kind, path in shard_paths.items()}
            if not all(post.values()):
                raise RuntimeError("fresh shard incomplete for " + record["key"])
            completed.append({"key": record["key"], "status": "created"})
            print(
                "STAGE4_FRESH_COMPLETE worker=%d/%d ordinal=%d/%d key=%s"
                % (worker_index, worker_count, ordinal + 1, len(records), record["key"]),
                flush=True,
            )
    finally:
        if runtime is not None:
            runtime.close()
    manifest = os.path.join(
        scratch_root,
        "fresh_worker_manifests",
        "worker_%02d_of_%02d.json" % (worker_index, worker_count),
    )
    atomic_json(
        manifest,
        {
            "worker_index": int(worker_index),
            "worker_count": int(worker_count),
            "fresh_split_sha256": sha256_file(split_path),
            "fresh_split_freeze_commit": freeze_commit,
            "records": completed,
        },
    )
    return {"records": len(completed), "manifest": manifest}


def verify_fresh(project_root, output_root=None, scratch_root=SCRATCH_ROOT):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    split_path = os.path.join(output_root, "FRESH_CONFIRMATION_SPLIT.json")
    freeze_commit = _assert_split_committed(project_root, split_path)
    split = _load_json(split_path)
    failures = []
    files = []
    branch_counts = Counter()
    combined = hashlib.sha256()
    for record in split["records"]:
        for kind, path in sorted(_fresh_paths(record["key"], scratch_root).items()):
            valid, evidence = validate_complete(path)
            if not valid:
                failures.append({"key": record["key"], "kind": kind, "reason": evidence})
                continue
            with np.load(path, allow_pickle=False) as data:
                branches = 1 if kind == "context" else len(data["residual_action"])
            expected = 1 if kind == "context" else (
                SUPPORT_TARGET_COUNT + 1 if kind == "support" else ACTION_BANK_SIZE + 1
            )
            if branches != expected:
                failures.append(
                    {"key": record["key"], "kind": kind, "reason": "branch_count"}
                )
            digest = sha256_file(path)
            relative = os.path.relpath(path, scratch_root).replace(os.sep, "/")
            combined.update(relative.encode() + b"\0" + digest.encode() + b"\0")
            branch_counts[kind] += int(branches)
            files.append({"key": record["key"], "kind": kind, "sha256": digest, "bytes": os.path.getsize(path)})
    result = {
        "evidence_label": FRESH_LABEL,
        "new_episode_claim": False,
        "fresh_split_freeze_commit": freeze_commit,
        "fresh_split_sha256": sha256_file(split_path),
        "states": len(split["records"]),
        "files": files,
        "file_count": len(files),
        "branch_counts": dict(sorted(branch_counts.items())),
        "combined_path_sha256_digest": combined.hexdigest(),
        "failures": failures,
        "passed": not failures and len(files) == 3 * len(split["records"]),
    }
    atomic_json(os.path.join(output_root, "fresh_collection_manifest.json"), result)
    if not result["passed"]:
        raise RuntimeError("fresh collection incomplete")
    return result


def load_fresh_records(project_root, output_root=None, scratch_root=SCRATCH_ROOT):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    split = _load_json(os.path.join(output_root, "FRESH_CONFIRMATION_SPLIT.json"))
    records = []
    for meta in split["records"]:
        shards = {}
        for kind, path in _fresh_paths(meta["key"], scratch_root).items():
            valid, evidence = validate_complete(path)
            if not valid:
                raise RuntimeError("incomplete fresh shard %s: %s" % (path, evidence))
            with np.load(path, allow_pickle=False) as data:
                shards[kind] = {name: np.asarray(data[name]).copy() for name in data.files}
        records.append(
            {
                "meta": dict(meta),
                "context": shards["context"],
                "support": shards["support"],
                "candidate": shards["candidate"],
            }
        )
    return records


def build_fresh_cache(project_root, output_root=None, scratch_root=SCRATCH_ROOT):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    records = load_fresh_records(project_root, output_root, scratch_root)
    with np.load(os.path.join(output_root, "stage4_scalers.npz"), allow_pickle=False) as data:
        consequence_scale = np.asarray(data["consequence_scale"], dtype=np.float64)
        context_center = np.asarray(data["context_center"], dtype=np.float32)
        context_scale = np.asarray(data["context_scale"], dtype=np.float32)
    arrays = _cache_arrays(records, consequence_scale, context_center, context_scale)
    path = os.path.join(scratch_root, "derived", "fresh_confirmation_matrix_cache.npz")
    atomic_npz(path, **arrays)
    mark_complete(
        path,
        {
            "kind": "stage4_fresh_confirmation_matrix_cache",
            "states": len(records),
            "evidence_label": FRESH_LABEL,
        },
    )
    result = {"path": path, "sha256": sha256_file(path), "states": len(records)}
    atomic_json(os.path.join(output_root, "fresh_confirmation_dataset.json"), result)
    return result


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("freeze", "collect", "verify", "build-cache")
    )
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--scratch-root", default=SCRATCH_ROOT)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze_fresh_split(args.project_root, args.output_root)
    elif args.command == "collect":
        result = collect_fresh_worker(
            args.project_root,
            args.output_root,
            args.worker_index,
            args.worker_count,
            args.scratch_root,
        )
    elif args.command == "verify":
        result = verify_fresh(args.project_root, args.output_root, args.scratch_root)
    else:
        result = build_fresh_cache(args.project_root, args.output_root, args.scratch_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
