"""Metric-independent fresh trajectory production and confirmation firewall."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import config
from .env_adapter import LiberoTaskRuntime
from .stage5_config import (
    GENERATOR_ARCHITECTURE,
    GENERATOR_REQUIRED_SUCCESSES_PER_TASK,
    GENERATOR_SACRIFICIAL_SEEDS,
    OUTPUT_RELATIVE,
    PHASES,
    SCRATCH_ROOT,
    TASKS,
    rollout_seeds,
)
from .stage5_generator import checkpoint_path, load_numpy_generator, predict_numpy
from .storage import atomic_json, atomic_npz, mark_complete, sha256_file, validate_complete


TRAJECTORY_SCHEMA = "stage5-fresh-state-bc-trajectory-v1"


def _task(task_id):
    return next(dict(value) for value in TASKS if value["task_id"] == task_id)


def _trajectory_path(scratch_root, task_id, seed):
    return os.path.join(scratch_root, "fresh_trajectories", task_id, "seed_%d.npz" % int(seed))


def _attempt_path(scratch_root, task_id, seed):
    return os.path.join(scratch_root, "fresh_attempts", task_id, "seed_%d.json" % int(seed))


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _reset_for_seed(runtime, seed):
    np.random.seed(int(seed) % (2**32 - 1))
    runtime.env.seed(int(seed) % (2**31 - 1))
    runtime.env.reset()
    runtime.env.sim.forward()
    runtime.env._post_process()
    runtime._geom_sets = runtime._build_geom_sets()


def rollout_once(runtime, generator, task_index, seed, maximum_steps=None):
    maximum_steps = int(maximum_steps or GENERATOR_ARCHITECTURE["maximum_rollout_steps"])
    _reset_for_seed(runtime, seed)
    task_one_hot = np.eye(len(TASKS), dtype=np.float32)[int(task_index)]
    previous = np.zeros(config.ACTION_DIM, dtype=np.float32)
    states, actions, chunks = [], [], []
    observable, masks, contacts, progress = [], [], [], []
    success = bool(runtime.env.check_success())
    for _ in range(maximum_steps):
        measured = runtime.measure()
        if measured["success"]:
            success = True
            break
        chunk = predict_numpy(
            generator,
            np.asarray(measured["vector"], dtype=np.float32),
            np.asarray(measured["mask"], dtype=bool),
            previous,
            task_one_hot,
        )
        action = np.asarray(chunk[0], dtype=np.float64)
        if action.shape != (config.ACTION_DIM,) or not np.isfinite(action).all():
            raise RuntimeError("invalid generator action")
        states.append(runtime.env.sim.get_state().flatten().copy())
        actions.append(action.copy())
        chunks.append(np.asarray(chunk, dtype=np.float32))
        observable.append(np.asarray(measured["vector"], dtype=np.float32))
        masks.append(np.asarray(measured["mask"], dtype=bool))
        contacts.append(bool(measured["contacts"]["relevant"]))
        progress.append(float(measured["progress"]))
        runtime.env.step(action)
        previous = action.astype(np.float32)
        success = bool(runtime.env.check_success())
        if success:
            break
    return {
        "schema_version": np.asarray(TRAJECTORY_SCHEMA),
        "seed": np.asarray(int(seed), dtype=np.int64),
        "task_index": np.asarray(int(task_index), dtype=np.int8),
        "success": np.asarray(bool(success)),
        "sim_state": np.asarray(states, dtype=np.float64),
        "executed_action": np.asarray(actions, dtype=np.float32),
        "predicted_h4_chunk": np.asarray(chunks, dtype=np.float32),
        "observable_state": np.asarray(observable, dtype=np.float32),
        "observable_mask": np.asarray(masks, dtype=bool),
        "current_contact": np.asarray(contacts, dtype=bool),
        "task_progress": np.asarray(progress, dtype=np.float32),
    }


def rollout_task(
    project_root,
    task_id,
    libero_source=config.LIBERO_SOURCE_DEFAULT,
    dataset_root=config.DATASET_ROOT_DEFAULT,
    output_root=None,
    scratch_root=SCRATCH_ROOT,
):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    if not os.path.isfile(os.path.join(output_root, "MODEL_SELECTION.json")):
        raise RuntimeError("development choices are not frozen")
    checkpoint = checkpoint_path(output_root)
    generator = load_numpy_generator(checkpoint)
    task = _task(task_id)
    task_index = [value["task_id"] for value in TASKS].index(task_id)
    successes = []
    attempts = []
    runtime = LiberoTaskRuntime(task, libero_source, dataset_root)
    started = time.time()
    try:
        for seed in rollout_seeds()[task_id]:
            attempt_path = _attempt_path(scratch_root, task_id, seed)
            if os.path.isfile(attempt_path):
                attempt = _load_json(attempt_path)
                if attempt.get("checkpoint_sha256") != sha256_file(checkpoint):
                    raise RuntimeError("attempt used another generator checkpoint")
                attempts.append(attempt)
                if attempt["success"]:
                    successes.append(attempt)
            else:
                values = rollout_once(runtime, generator, task_index, seed)
                trajectory_path = _trajectory_path(scratch_root, task_id, seed)
                if bool(values["success"].item()):
                    atomic_npz(trajectory_path, **values)
                    mark_complete(
                        trajectory_path,
                        {
                            "kind": "stage5_fresh_nominal_trajectory",
                            "schema_version": TRAJECTORY_SCHEMA,
                            "task_id": task_id,
                            "seed": int(seed),
                            "metric_scores_read": False,
                        },
                    )
                    trajectory_sha = sha256_file(trajectory_path)
                else:
                    trajectory_sha = None
                attempt = {
                    "task_id": task_id,
                    "seed": int(seed),
                    "success": bool(values["success"].item()),
                    "steps": int(len(values["executed_action"])),
                    "final_progress": float(values["task_progress"][-1]) if len(values["task_progress"]) else 0.0,
                    "trajectory_path": trajectory_path if trajectory_sha else None,
                    "trajectory_sha256": trajectory_sha,
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "metric_scores_read": False,
                }
                atomic_json(attempt_path, attempt)
                attempts.append(attempt)
                if attempt["success"]:
                    successes.append(attempt)
                print(
                    "fresh-rollout task=%s seed=%d success=%s steps=%d successes=%d elapsed=%.1fs"
                    % (task_id, seed, attempt["success"], attempt["steps"], len(successes), time.time() - started),
                    flush=True,
                )
            if len(successes) >= GENERATOR_REQUIRED_SUCCESSES_PER_TASK:
                break
    finally:
        runtime.close()
    summary = {
        "task_id": task_id,
        "generator_checkpoint_sha256": sha256_file(checkpoint),
        "attempt_count": len(attempts),
        "success_count": len(successes),
        "required_success_count": GENERATOR_REQUIRED_SUCCESSES_PER_TASK,
        "sufficient": len(successes) >= GENERATOR_REQUIRED_SUCCESSES_PER_TASK,
        "accepted": successes[:GENERATOR_REQUIRED_SUCCESSES_PER_TASK],
        "acceptance": "first environment-success trajectories in ascending frozen seed order",
        "metric_scores_read": False,
    }
    atomic_json(os.path.join(output_root, "FRESH_TRAJECTORY_%s.json" % task_id.upper()), summary)
    return summary


def replay_validation(
    project_root,
    libero_source=config.LIBERO_SOURCE_DEFAULT,
    dataset_root=config.DATASET_ROOT_DEFAULT,
    output_root=None,
):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    generator = load_numpy_generator(checkpoint_path(output_root))
    rows = []
    for task_index, task in enumerate(TASKS):
        runtime = LiberoTaskRuntime(task, libero_source, dataset_root)
        try:
            for seed in GENERATOR_SACRIFICIAL_SEEDS[task["task_id"]]:
                left = rollout_once(runtime, generator, task_index, seed, maximum_steps=80)
                right = rollout_once(runtime, generator, task_index, seed, maximum_steps=80)
                state_equal = np.array_equal(left["sim_state"], right["sim_state"])
                action_equal = np.array_equal(left["executed_action"], right["executed_action"])
                rows.append(
                    {
                        "task_id": task["task_id"],
                        "seed": int(seed),
                        "state_exact": bool(state_equal),
                        "action_exact": bool(action_equal),
                        "success_equal": bool(left["success"].item() == right["success"].item()),
                        "steps_equal": len(left["executed_action"]) == len(right["executed_action"]),
                        "passed": bool(state_equal and action_equal and left["success"].item() == right["success"].item()),
                    }
                )
        finally:
            runtime.close()
    result = {
        "kind": "sacrificial fresh generator deterministic replay validation",
        "confirmation_states_executed": False,
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
    }
    atomic_json(os.path.join(output_root, "FRESH_REPLAY_VALIDATION.json"), result)
    return result


def _load_trajectory(path):
    valid, evidence = validate_complete(path)
    if not valid:
        raise RuntimeError("incomplete trajectory %s: %s" % (path, evidence))
    with np.load(path, allow_pickle=False) as data:
        result = {name: np.asarray(data[name]).copy() for name in data.files}
    if str(result["schema_version"].item()) != TRAJECTORY_SCHEMA:
        raise RuntimeError("trajectory schema changed")
    return result


def _executable_indices(values, local_residuals, fresh_residuals):
    chunks = np.asarray(values["predicted_h4_chunk"], dtype=np.float64)
    nominal = chunks[:, :, :6].reshape(len(chunks), -1)
    local_valid = np.all(
        np.abs(nominal[:, None, :] + local_residuals[None, :, :]) <= 1.0,
        axis=2,
    )
    fresh_valid = np.all(
        np.abs(nominal[:, None, :] + fresh_residuals[None, :, :]) <= 1.0,
        axis=2,
    )
    return np.flatnonzero(np.all(local_valid, axis=1) & np.all(fresh_valid, axis=1)), local_valid, fresh_valid


def _choose_indices(values, executable):
    executable = np.asarray(executable, dtype=np.int64)
    if len(executable) < len(PHASES):
        raise RuntimeError("fewer than four executable states")
    contact = np.asarray(values["current_contact"], dtype=bool)
    progress = np.asarray(values["task_progress"], dtype=np.float64)
    onset = np.flatnonzero(contact & np.concatenate(([True], ~contact[:-1])))
    fallback_anchor = False
    if len(onset):
        anchor = int(onset[0])
    else:
        delta = np.diff(progress, prepend=progress[0])
        anchor = int(np.argmax(delta))
        fallback_anchor = True
    desired = {
        "free_space": 0,
        "pre_contact": max(anchor - 1, 0),
        "contact_onset": anchor,
        "post_contact": min(anchor + 1, len(contact) - 1),
    }
    pools = {
        "free_space": executable[(executable <= anchor // 3) & (~contact[executable])],
        "pre_contact": executable[(executable < anchor) & (~contact[executable])],
        "contact_onset": executable,
        "post_contact": executable[(executable > anchor) & contact[executable]],
    }
    selected = {}
    audit = {}
    used = set()
    for phase in PHASES:
        pool = np.asarray([value for value in pools[phase] if int(value) not in used], dtype=np.int64)
        fallback = False
        if not len(pool):
            pool = np.asarray([value for value in executable if int(value) not in used], dtype=np.int64)
            fallback = True
        if not len(pool):
            raise RuntimeError("cannot select four unique executable states")
        index = int(min(pool.tolist(), key=lambda value: (abs(int(value) - desired[phase]), int(value))))
        selected[phase] = index
        used.add(index)
        audit[phase] = {"index": index, "desired_index": int(desired[phase]), "fallback": fallback}
    return selected, {"anchor": anchor, "contact_absent_fallback": fallback_anchor, "selection": audit}


def freeze_confirmation_split(project_root, output_root=None, scratch_root=SCRATCH_ROOT):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    destination = os.path.join(output_root, "FRESH_CONFIRMATION_SPLIT.json")
    if os.path.exists(destination):
        raise RuntimeError("fresh confirmation split already exists")
    selection_path = os.path.join(output_root, "MODEL_SELECTION.json")
    if not os.path.isfile(selection_path):
        raise RuntimeError("model selection not frozen")
    with np.load(os.path.join(output_root, "LOCAL_BANK.npz"), allow_pickle=False) as data:
        local_residuals = np.asarray(data["residuals"], dtype=np.float64)
    with np.load(os.path.join(output_root, "FRESH_TARGET_BANK.npz"), allow_pickle=False) as data:
        fresh_residuals = np.asarray(data["residuals"], dtype=np.float64)
    records = []
    shortages = []
    summaries = {}
    for task in TASKS:
        task_id = task["task_id"]
        summary_path = os.path.join(output_root, "FRESH_TRAJECTORY_%s.json" % task_id.upper())
        summary = _load_json(summary_path)
        summaries[task_id] = summary
        if not summary["sufficient"]:
            shortages.append({"task_id": task_id, "success_count": summary["success_count"], "reason": "fewer_than_12_environment_successes"})
            continue
        for trajectory_index, accepted in enumerate(summary["accepted"]):
            path = accepted["trajectory_path"]
            if sha256_file(path) != accepted["trajectory_sha256"]:
                raise RuntimeError("fresh trajectory hash changed")
            values = _load_trajectory(path)
            executable, local_valid, fresh_valid = _executable_indices(values, local_residuals, fresh_residuals)
            try:
                selected, audit = _choose_indices(values, executable)
            except RuntimeError as error:
                shortages.append({"task_id": task_id, "seed": int(accepted["seed"]), "reason": str(error), "executable_state_count": int(len(executable))})
                continue
            for phase in PHASES:
                index = int(selected[phase])
                records.append(
                    {
                        "task_id": task_id,
                        "trajectory_index": int(trajectory_index),
                        "source_episode_id": int(accepted["seed"]),
                        "rollout_seed": int(accepted["seed"]),
                        "phase": phase,
                        "state_index": index,
                        "state_key": "%s__seed%d__%s" % (task_id, int(accepted["seed"]), phase),
                        "trajectory_path": path,
                        "trajectory_sha256": accepted["trajectory_sha256"],
                        "current_contact": bool(values["current_contact"][index]),
                        "task_progress": float(values["task_progress"][index]),
                        "valid_local_candidates": int(np.sum(local_valid[index])),
                        "valid_fresh_targets": int(np.sum(fresh_valid[index])),
                        "selection_audit": audit["selection"][phase],
                        "contact_anchor": int(audit["anchor"]),
                        "contact_absent_fallback": bool(audit["contact_absent_fallback"]),
                    }
                )
    complete = not shortages and len(records) == len(TASKS) * GENERATOR_REQUIRED_SUCCESSES_PER_TASK * len(PHASES)
    payload = {
        "kind": "fresh policy trajectory confirmation firewall",
        "evidence_label": "FRESH_POLICY_TRAJECTORY_CONFIRMATION",
        "complete": complete,
        "record_count": len(records),
        "expected_record_count": len(TASKS) * GENERATOR_REQUIRED_SUCCESSES_PER_TASK * len(PHASES),
        "records": records,
        "shortages": shortages,
        "trajectory_summaries": summaries,
        "model_selection_sha256": sha256_file(selection_path),
        "phase_selection_rule_sha256": sha256_file(os.path.join(output_root, "FRESH_PHASE_SELECTION_RULE.json")),
        "local_bank_sha256": sha256_file(os.path.join(output_root, "LOCAL_BANK.npz")),
        "fresh_target_bank_sha256": sha256_file(os.path.join(output_root, "FRESH_TARGET_BANK.npz")),
        "confirmation_branches_executed_before_freeze": False,
        "metric_scores_used_for_trajectory_acceptance_or_state_selection": False,
        "failure_disposition": "BLOCKED_NO_FRESH_TRAJECTORIES",
    }
    atomic_json(destination, payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rollout = sub.add_parser("rollout-task")
    rollout.add_argument("--task-id", required=True, choices=[task["task_id"] for task in TASKS])
    sub.add_parser("replay-validation")
    sub.add_parser("freeze-split")
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--libero-source", default=config.LIBERO_SOURCE_DEFAULT)
    parser.add_argument("--dataset-root", default=config.DATASET_ROOT_DEFAULT)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--scratch-root", default=SCRATCH_ROOT)
    args = parser.parse_args(argv)
    if args.command == "rollout-task":
        result = rollout_task(args.project_root, args.task_id, args.libero_source, args.dataset_root, args.output_root, args.scratch_root)
    elif args.command == "replay-validation":
        result = replay_validation(args.project_root, args.libero_source, args.dataset_root, args.output_root)
    else:
        result = freeze_confirmation_split(args.project_root, args.output_root, args.scratch_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
