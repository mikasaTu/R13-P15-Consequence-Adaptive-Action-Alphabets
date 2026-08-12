"""Simulator-facing Stage 1 pipeline stages."""

from __future__ import annotations

import csv
import datetime as dt
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys

import numpy as np

from . import __version__, config
from .env_adapter import FEATURE_NAMES, LiberoTaskRuntime, perturb_continuous_chunk
from .math_utils import deterministic_directions
from .storage import (
    atomic_json,
    atomic_npz,
    atomic_text,
    mark_complete,
    sha256_file,
    sha256_tree,
    validate_complete,
)


CONTACT_MODE_TO_ID = {
    "no_contact": 0,
    "onset": 1,
    "persistent": 2,
    "release": 3,
}


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _run_text(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as error:
        return "ERROR:%s" % (error,)


def scaffold(output_root, project_root):
    os.makedirs(output_root, exist_ok=True)
    source = os.path.join(project_root, "PREREGISTRATION_TEMPLATE.md")
    destination = os.path.join(output_root, "PREREGISTRATION.md")
    if not os.path.exists(destination):
        shutil.copyfile(source, destination)
    schema = {
        "schema_version": "caaa-libero-consequence-v1",
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "rotation_representation": "continuous_6d_first_two_rotation_columns",
        "continuous_jacobian_excludes": [
            "success",
            "categorical_contact_transition",
        ],
        "contact_modes": list(CONTACT_MODE_TO_ID.keys()),
        "immediate_definition": "state after H=4 OSC_POSE controls",
        "settled_definition": "immediate plus three zero-delta-pose steps with final gripper command held",
        "mask_policy": "unavailable task dimensions are masked and never replaced by pseudo-constants",
        "scale_policy": "median/MAD-IQR robust scale from train episodes only",
    }
    atomic_json(os.path.join(output_root, "consequence_schema.json"), schema)
    for directory in (
        "work/branch_shards",
        "work/quantization_plans",
        "work/quantized_shards",
        "alphabet_codebooks",
    ):
        os.makedirs(os.path.join(output_root, directory), exist_ok=True)


def _package_versions(names):
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except Exception:
            result[name] = None
    return result


def freeze_environment(paths, project_root, output_root, hash_demos=True, task_limit=None):
    tasks = config.TASKS[:task_limit] if task_limit else config.TASKS
    demo_records = []
    task_metadata = []
    for task in tasks:
        runtime = LiberoTaskRuntime(task, paths["libero_source"], paths["dataset_root"])
        try:
            metadata = runtime.dataset_metadata()
        finally:
            runtime.close()
        if metadata["num_demos"] < config.N_EPISODES or metadata["successful_demos"] < config.N_EPISODES:
            raise RuntimeError("insufficient successful demonstrations for %s" % task["task_id"])
        demo_record = {
            "task_id": task["task_id"],
            "path": metadata["demo_path"],
            "bytes": os.path.getsize(metadata["demo_path"]),
            "sha256": sha256_file(metadata["demo_path"]) if hash_demos else None,
        }
        demo_records.append(demo_record)
        task_metadata.append(
            {
                "task": dict(task),
                "dataset": metadata,
                "frozen_episode_ids": list(range(config.N_EPISODES)),
            }
        )
    torch_cuda = None
    torch_version = None
    try:
        import torch

        torch_cuda = torch.version.cuda
        torch_version = torch.__version__
    except Exception:
        pass
    source_hash = sha256_tree(paths["libero_source"])
    project_hash = sha256_tree(project_root)
    lock = {
        "created_utc": utc_now(),
        "project_version": __version__,
        "project_source_tree_sha256": project_hash,
        "project_git_commit": _run_text(["git", "-C", project_root, "rev-parse", "HEAD"]),
        "project_git_status": _run_text(["git", "-C", project_root, "status", "--short"]),
        "libero": {
            "source_path": paths["libero_source"],
            "source_tree_sha256": source_hash,
            "upstream_commit": config.UPSTREAM_LIBERO_COMMIT,
            "upstream_comparison": "byte-identical diff -qr against a shallow official checkout; .git and __pycache__ excluded",
            "suite": config.SUITE,
            "standard_libero_not_plus_perturbations": True,
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "kernel": platform.release(),
        },
        "packages": _package_versions(
            [
                "libero",
                "robosuite",
                "mujoco",
                "torch",
                "numpy",
                "scipy",
                "h5py",
                "zarr",
                "pandas",
                "pyarrow",
            ]
        ),
        "torch": {"version": torch_version, "cuda_build": torch_cuda},
        "cuda": {
            "nvidia_smi": _run_text(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv,noheader"]),
            "local_gpu_use_contract": "CPU smoke or at most one GPU; simulator launched with one visible GPU at most",
        },
        "sapien": {"applicable": False, "reason": "LIBERO uses MuJoCo/robosuite"},
        "controller": {
            "robot": "Panda",
            "mode": config.CONTROL_MODE,
            "frequency_hz": config.CONTROL_FREQUENCY_HZ,
            "action_dim": config.ACTION_DIM,
            "chunk_horizon": config.CHUNK_HORIZON,
            "continuous_chunk_dim": config.CHUNK_CONTINUOUS_DIM,
            "gripper_policy": "copied unchanged from each demonstration chunk",
        },
        "paths": paths,
        "demonstrations": demo_records,
        "tasks": task_metadata,
    }
    atomic_json(os.path.join(output_root, "environment_lock.json"), lock)
    return lock


def select_snapshots(paths, output_root, task_limit=None, episode_limit=None):
    tasks = config.TASKS[:task_limit] if task_limit else config.TASKS
    n_episodes = int(episode_limit or config.N_EPISODES)
    records = []
    for task in tasks:
        runtime = LiberoTaskRuntime(task, paths["libero_source"], paths["dataset_root"])
        try:
            for episode_id in range(n_episodes):
                episode = runtime.load_episode(episode_id)
                indices, notes, scan = runtime.select_phase_indices(episode)
                split = "train" if episode_id in config.TRAIN_EPISODES else (
                    "calibration" if episode_id in config.CALIBRATION_EPISODES else "test"
                )
                records.append(
                    {
                        "task_id": task["task_id"],
                        "task_name": task["task_name"],
                        "episode_id": episode_id,
                        "split": split,
                        "phase_indices": indices,
                        "selection_notes": notes,
                        "episode_length": int(len(episode["actions"])),
                        "recorded_success": bool(np.max(episode["rewards"]) > 0),
                        "contact_fraction": float(np.mean(scan["contact"])),
                        "target_contact_fraction": float(np.mean(scan["target_contact"])),
                        "progress_min_max": [float(np.min(scan["progress"])), float(np.max(scan["progress"]))],
                    }
                )
        finally:
            runtime.close()
    split = {
        "created_utc": utc_now(),
        "global_seed": config.GLOBAL_SEED,
        "suite": config.SUITE,
        "tasks": [dict(task) for task in tasks],
        "episode_split": {
            "train": list(config.TRAIN_EPISODES),
            "calibration": list(config.CALIBRATION_EPISODES),
            "test": list(config.TEST_EPISODES),
        },
        "records": records,
    }
    atomic_json(os.path.join(output_root, "task_and_seed_split.json"), split)
    return split


def load_split(output_root):
    with open(os.path.join(output_root, "task_and_seed_split.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _record_lookup(split):
    return {
        (record["task_id"], record["episode_id"]): record
        for record in split["records"]
    }


def _compare_rollouts(left, right, tolerance):
    metrics = {}
    for key, a, b in (
        ("final_state", left["final_state"], right["final_state"]),
        ("immediate_consequence", left["immediate"]["vector"], right["immediate"]["vector"]),
        ("settled_consequence", left["settled"]["vector"], right["settled"]["vector"]),
    ):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        metrics[key + "_max_abs"] = float(np.max(np.abs(a - b)))
        metrics[key + "_l2"] = float(np.linalg.norm(a - b))
    metrics["contact_mode_equal"] = left["contact_mode"] == right["contact_mode"]
    metrics["success_equal"] = left["settled"]["success"] == right["settled"]["success"]
    metrics["reward_equal"] = left["step_rewards"] == right["step_rewards"]
    metrics["done_equal"] = left["step_dones"] == right["step_dones"]
    passed = (
        metrics["final_state_max_abs"] <= tolerance
        and metrics["immediate_consequence_max_abs"] <= tolerance
        and metrics["settled_consequence_max_abs"] <= tolerance
        and metrics["contact_mode_equal"]
        and metrics["success_equal"]
        and metrics["reward_equal"]
        and metrics["done_equal"]
    )
    return passed, metrics


def validate_replay(paths, output_root, task_limit=None, episode_limit=2, phase_limit=None, tolerance=1e-12):
    split = load_split(output_root)
    lookup = _record_lookup(split)
    tasks = config.TASKS[:task_limit] if task_limit else config.TASKS
    directions = deterministic_directions(config.CHUNK_CONTINUOUS_DIM, config.GLOBAL_SEED)
    tests = []
    for task in tasks:
        runtime = LiberoTaskRuntime(task, paths["libero_source"], paths["dataset_root"])
        try:
            for episode_id in range(int(episode_limit)):
                episode = runtime.load_episode(episode_id)
                runtime.initialize_episode_model(episode)
                record = lookup[(task["task_id"], episode_id)]
                phases = list(config.PHASES[:phase_limit]) if phase_limit else list(config.PHASES)
                for phase in phases:
                    index = int(record["phase_indices"][phase])
                    base_actions = episode["actions"][index : index + config.CHUNK_HORIZON]
                    snapshot = runtime.snapshot_from_recorded_state(
                        episode["states"][index], episode["actions"][:index]
                    )
                    base_cont = runtime.continuous_chunk(base_actions)
                    pert_cont, _ = perturb_continuous_chunk(base_cont, directions[0], 0.05, 1)
                    pert_actions = runtime.replace_continuous_chunk(base_actions, pert_cont)
                    a_first = runtime.execute_chunk(snapshot, base_actions)
                    a_second = runtime.execute_chunk(snapshot, base_actions)
                    b_first = runtime.execute_chunk(snapshot, pert_actions)
                    a_after_b = runtime.execute_chunk(snapshot, base_actions)
                    b_after_a = runtime.execute_chunk(snapshot, pert_actions)
                    same_a, same_a_metrics = _compare_rollouts(a_first, a_second, tolerance)
                    order_a, order_a_metrics = _compare_rollouts(a_first, a_after_b, tolerance)
                    order_b, order_b_metrics = _compare_rollouts(b_first, b_after_a, tolerance)
                    tests.append(
                        {
                            "task_id": task["task_id"],
                            "episode_id": episode_id,
                            "phase": phase,
                            "snapshot_index": index,
                            "same_action_twice": {"passed": same_a, "metrics": same_a_metrics},
                            "a_after_b": {"passed": order_a, "metrics": order_a_metrics},
                            "b_after_a": {"passed": order_b, "metrics": order_b_metrics},
                            "passed": bool(same_a and order_a and order_b),
                        }
                    )
        finally:
            runtime.close()
    failures = [test for test in tests if not test["passed"]]
    result = {
        "created_utc": utc_now(),
        "tolerance": float(tolerance),
        "tests": tests,
        "failed_tests": failures,
        "n_tests": len(tests),
        "n_failed": len(failures),
        "passed": not failures,
        "gate": "PASS" if not failures else "BLOCKED_NONDETERMINISTIC_BRANCHING",
    }
    atomic_json(os.path.join(output_root, "branch_replay_validation.json"), result)
    return result


def _branch_shard_path(output_root, task_id, episode_id, phase):
    return os.path.join(
        output_root,
        "work",
        "branch_shards",
        task_id,
        "episode_%02d_%s.npz" % (int(episode_id), phase),
    )


def _pack_rollouts(rollouts):
    return {
        "initial": np.stack([row["initial"]["vector"] for row in rollouts]),
        "immediate": np.stack([row["immediate"]["vector"] for row in rollouts]),
        "settled": np.stack([row["settled"]["vector"] for row in rollouts]),
        "mask": np.stack([row["settled"]["mask"] for row in rollouts]),
        "initial_success": np.asarray([row["initial"]["success"] for row in rollouts], dtype=np.uint8),
        "immediate_success": np.asarray([row["immediate"]["success"] for row in rollouts], dtype=np.uint8),
        "settled_success": np.asarray([row["settled"]["success"] for row in rollouts], dtype=np.uint8),
        "initial_progress": np.asarray([row["initial"]["progress"] for row in rollouts], dtype=np.float64),
        "immediate_progress": np.asarray([row["immediate"]["progress"] for row in rollouts], dtype=np.float64),
        "settled_progress": np.asarray([row["settled"]["progress"] for row in rollouts], dtype=np.float64),
        "contact_mode": np.asarray([CONTACT_MODE_TO_ID[row["contact_mode"]] for row in rollouts], dtype=np.int8),
        "contact_sequence": np.asarray([row["contact_sequence"] for row in rollouts], dtype=np.uint8),
        "final_state": np.stack([row["final_state"] for row in rollouts]),
    }


def collect_branches(
    paths,
    output_root,
    task_limit=None,
    task_ids=None,
    episode_limit=None,
    phase_limit=None,
    direction_limit=None,
    require_replay_pass=True,
):
    if require_replay_pass:
        with open(os.path.join(output_root, "branch_replay_validation.json"), "r", encoding="utf-8") as handle:
            validation = json.load(handle)
        if not validation.get("passed"):
            raise RuntimeError("BLOCKED_NONDETERMINISTIC_BRANCHING")
    split = load_split(output_root)
    lookup = _record_lookup(split)
    tasks = config.TASKS[:task_limit] if task_limit else config.TASKS
    if task_ids is not None:
        requested = set(task_ids)
        tasks = tuple(task for task in tasks if task["task_id"] in requested)
        missing = requested - set(task["task_id"] for task in tasks)
        if missing:
            raise KeyError("unknown task ids: %s" % sorted(missing))
    n_episodes = int(episode_limit or config.N_EPISODES)
    n_directions = int(direction_limit or config.PERTURBATION_DIRECTIONS)
    directions = deterministic_directions(config.CHUNK_CONTINUOUS_DIM, config.GLOBAL_SEED)[:n_directions]
    completed = []
    for task in tasks:
        runtime = LiberoTaskRuntime(task, paths["libero_source"], paths["dataset_root"])
        try:
            for episode_id in range(n_episodes):
                episode = runtime.load_episode(episode_id)
                runtime.initialize_episode_model(episode)
                record = lookup[(task["task_id"], episode_id)]
                phases = list(config.PHASES[:phase_limit]) if phase_limit else list(config.PHASES)
                for phase in phases:
                    shard = _branch_shard_path(output_root, task["task_id"], episode_id, phase)
                    valid, evidence = validate_complete(shard)
                    if valid:
                        completed.append({"path": shard, "status": "resumed", "evidence": evidence})
                        continue
                    index = int(record["phase_indices"][phase])
                    base_actions = episode["actions"][index : index + config.CHUNK_HORIZON].copy()
                    base_cont = runtime.continuous_chunk(base_actions)
                    snapshot = runtime.snapshot_from_recorded_state(
                        episode["states"][index], episode["actions"][:index]
                    )
                    rollouts = [runtime.execute_chunk(snapshot, base_actions)]
                    action_cont = [base_cont]
                    delta_action = [np.zeros_like(base_cont)]
                    action_full = [base_actions]
                    direction_ids = [-1]
                    signs = [0]
                    radii = [0.0]
                    for direction_id, direction in enumerate(directions):
                        for radius in config.PERTURBATION_RADII:
                            for sign in config.PERTURBATION_SIGNS:
                                candidate, delta = perturb_continuous_chunk(base_cont, direction, radius, sign)
                                candidate_actions = runtime.replace_continuous_chunk(base_actions, candidate)
                                rollouts.append(runtime.execute_chunk(snapshot, candidate_actions))
                                action_cont.append(candidate)
                                delta_action.append(delta)
                                action_full.append(candidate_actions)
                                direction_ids.append(direction_id)
                                signs.append(sign)
                                radii.append(radius)
                    packed = _pack_rollouts(rollouts)
                    atomic_npz(
                        shard,
                        task_id=np.asarray(task["task_id"]),
                        task_name=np.asarray(task["task_name"]),
                        episode_id=np.asarray(episode_id, dtype=np.int32),
                        split=np.asarray(record["split"]),
                        phase=np.asarray(phase),
                        snapshot_index=np.asarray(index, dtype=np.int32),
                        snapshot_state=np.asarray(snapshot["sim_state"], dtype=np.float64),
                        base_actions=np.asarray(base_actions, dtype=np.float64),
                        action_cont=np.asarray(action_cont, dtype=np.float64),
                        delta_action=np.asarray(delta_action, dtype=np.float64),
                        action_full=np.asarray(action_full, dtype=np.float64),
                        direction=np.asarray(direction_ids, dtype=np.int16),
                        sign=np.asarray(signs, dtype=np.int8),
                        radius=np.asarray(radii, dtype=np.float64),
                        **packed
                    )
                    marker = mark_complete(
                        shard,
                        {
                            "task_id": task["task_id"],
                            "episode_id": episode_id,
                            "phase": phase,
                            "snapshot_index": index,
                            "branches": len(rollouts),
                            "created_utc": utc_now(),
                            "pai_run_id": os.environ.get("PAI_CANARY_RUN_ID"),
                            "pai_nonce": os.environ.get("PAI_CANARY_NONCE"),
                        },
                    )
                    completed.append({"path": shard, "marker": marker, "status": "created"})
                    print(
                        "BRANCH_SHARD_COMPLETE task=%s episode=%d phase=%s branches=%d"
                        % (task["task_id"], episode_id, phase, len(rollouts)),
                        flush=True,
                    )
        finally:
            runtime.close()
    manifest_suffix = "all"
    if task_ids is not None:
        manifest_suffix = "_".join(sorted(task["task_id"] for task in tasks))
    atomic_json(
        os.path.join(
            output_root,
            "work",
            "branch_collection_%s.json" % manifest_suffix,
        ),
        {
            "created_utc": utc_now(),
            "pai_run_id": os.environ.get("PAI_CANARY_RUN_ID"),
            "pai_nonce": os.environ.get("PAI_CANARY_NONCE"),
            "task_ids": [task["task_id"] for task in tasks],
            "shards": completed,
            "count": len(completed),
        },
    )
    return completed


def smoke(paths, project_root, output_root):
    scaffold(output_root, project_root)
    freeze_environment(paths, project_root, output_root, hash_demos=False, task_limit=1)
    select_snapshots(paths, output_root, task_limit=1, episode_limit=1)
    validation = validate_replay(
        paths,
        output_root,
        task_limit=1,
        episode_limit=1,
        phase_limit=1,
        tolerance=1e-12,
    )
    if not validation["passed"]:
        return validation
    collect_branches(
        paths,
        output_root,
        task_limit=1,
        episode_limit=1,
        phase_limit=1,
        direction_limit=2,
        require_replay_pass=True,
    )
    shard = _branch_shard_path(output_root, config.TASKS[0]["task_id"], 0, config.PHASES[0])
    with np.load(shard, allow_pickle=False) as data:
        evidence = {
            "created_utc": utc_now(),
            "passed": bool(data["settled"].shape == (9, len(FEATURE_NAMES))),
            "branch_shape": list(data["settled"].shape),
            "action_shape": list(data["action_full"].shape),
            "all_finite": bool(np.all(np.isfinite(data["settled"]))),
            "replay_validation": validation,
            "scope": "one task, one episode, one phase, two directions; CPU/no-render smoke",
        }
    evidence["passed"] = bool(evidence["passed"] and evidence["all_finite"])
    atomic_json(os.path.join(output_root, "LOCAL_SMOKE.json"), evidence)
    return evidence
