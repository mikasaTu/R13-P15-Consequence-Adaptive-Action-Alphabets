"""Development and historical exploratory evaluation for Stage 5 CICR-DLA."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np

from .stage3_data import CONTEXT_SLICES
from .stage3_metrics import (
    paired_episode_bootstrap,
    ranking_metrics,
    realized_rows,
    write_csv,
)
from .stage4_data import historical_records
from .stage5_config import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONTACT_SENSITIVE_TASKS,
    GATES,
    LOCAL_BANK_SIZE,
    MATCHED_CONTROLS,
    MODEL_SEEDS,
    OUTPUT_RELATIVE,
    PHASES,
    PRIMARY_K,
    SCRATCH_ROOT,
    TASK_IDS,
)
from .stage5_data import cache_path, load_cache
from .stage5_models import (
    CONTEXT_METHOD,
    PROPOSED,
    action_distance_matrix,
    load_context_checkpoint,
    load_static_checkpoint,
    predict_context,
    predict_static,
)
from .stage5_oracle import _assign, deterministic_kmedoids_precomputed
from .storage import atomic_json, sha256_file


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _device(name):
    import torch

    device = torch.device(name or "cpu")
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        raise RuntimeError("Stage 5 permits one visible evaluation GPU")
    if device.type == "cpu":
        torch.set_num_threads(min(16, max(1, os.cpu_count() or 1)))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    return device


def _manifest_entries(output_root):
    manifest = _load_json(os.path.join(output_root, "MODEL_TRAINING_MANIFEST.json"))
    for entry in manifest["entries"]:
        path = os.path.join(output_root, entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError("checkpoint hash mismatch: " + entry["path"])
    return manifest


def _select_entries(manifest, method=None, control=None, selected_temperature=True):
    output = []
    tau = float(manifest["selected_temperature"])
    for entry in manifest["entries"]:
        metadata = entry["metadata"]
        if method is not None and metadata.get("method") != method:
            continue
        if control is not None and metadata.get("control") != control:
            continue
        if selected_temperature and method == "B2_STATIC_CONSEQUENCE":
            if float(metadata["temperature"]) != tau:
                continue
        output.append(entry)
    if len(output) != len(MODEL_SEEDS):
        raise RuntimeError("expected three checkpoints for %s/%s, got %d" % (method, control, len(output)))
    output.sort(key=lambda entry: int(entry["metadata"]["seed"]))
    return output


def _evaluation_inputs(cache, control):
    context = np.asarray(cache["context"], dtype=np.float32).copy()
    nominal = np.asarray(cache["nominal_action"], dtype=np.float32).copy()
    if control == "ACTION_ONLY":
        context[:] = 0.0
    elif control == "PHASE_ONLY":
        context[:] = 0.0
        context[np.arange(len(context)), np.asarray(cache["phase_index"], dtype=np.int64)] = 1.0
        left, right = CONTEXT_SLICES["task_one_hot"]
        context[:, left:right] = np.eye(right - left, dtype=np.float32)[
            np.asarray(cache["task_index"], dtype=np.int64)
        ]
    elif control == "CURRENT_CONTACT_ONLY":
        context[:] = 0.0
        left, right = CONTEXT_SLICES["current_contact"]
        context[:, left:right] = np.asarray(cache["current_contact"], dtype=np.float32)[:, None]
        left, right = CONTEXT_SLICES["task_one_hot"]
        context[:, left:right] = np.eye(right - left, dtype=np.float32)[
            np.asarray(cache["task_index"], dtype=np.int64)
        ]
    return context, nominal


def _load_score_bundle(project_root, output_root, cache, device):
    manifest = _manifest_entries(output_root)
    bundle = {}
    models_by_method = {}
    for method in ("B1_ACTION_ONLY", "B2_STATIC_CONSEQUENCE"):
        entries = _select_entries(manifest, method=method)
        models = [
            load_static_checkpoint(os.path.join(output_root, entry["path"]), device)[0]
            for entry in entries
        ]
        started = time.perf_counter()
        seed_scores = [predict_static(model, cache, device) for model in models]
        elapsed = time.perf_counter() - started
        weights = [model.positive_weight().detach().cpu().numpy() for model in models]
        condition = float(np.mean([np.max(value) / max(np.min(value), 1e-12) for value in weights]))
        bundle[method] = {
            "scores": np.mean(seed_scores, axis=0),
            "seed_scores": seed_scores,
            "entries": entries,
            "latency_ms_per_query": 1000.0 * elapsed / (len(models) * len(cache["context"]) * len(cache["target_residual"])),
            "modulation_norm_by_state": np.zeros(len(cache["context"]), dtype=np.float64),
            "condition_number_by_state": np.full(len(cache["context"]), condition, dtype=np.float64),
        }
        models_by_method[method] = models
    for control in (PROPOSED,) + tuple(MATCHED_CONTROLS):
        entries = _select_entries(manifest, method=CONTEXT_METHOD, control=control)
        models = [
            load_context_checkpoint(os.path.join(output_root, entry["path"]), device)[0]
            for entry in entries
        ]
        context, nominal = _evaluation_inputs(cache, control)
        started = time.perf_counter()
        seed_scores = [predict_context(model, cache, device, context, nominal) for model in models]
        elapsed = time.perf_counter() - started
        import torch

        modulation_rows = []
        condition_rows = []
        with torch.no_grad():
            context_tensor = torch.as_tensor(context, device=device)
            for model in models:
                modulation = model.modulation(context_tensor).cpu().numpy()
                weight = model.positive_weight(context_tensor).cpu().numpy()
                modulation_rows.append(np.linalg.norm(modulation, axis=1))
                condition_rows.append(np.max(weight, axis=1) / np.maximum(np.min(weight, axis=1), 1e-12))
        key = CONTEXT_METHOD if control == PROPOSED else "CONTROL_" + control
        bundle[key] = {
            "scores": np.mean(seed_scores, axis=0),
            "seed_scores": seed_scores,
            "entries": entries,
            "models": models,
            "eval_context": context,
            "eval_nominal": nominal,
            "latency_ms_per_query": 1000.0 * elapsed / (len(models) * len(cache["context"]) * len(cache["target_residual"])),
            "modulation_norm_by_state": np.mean(modulation_rows, axis=0),
            "condition_number_by_state": np.mean(condition_rows, axis=0),
        }
        models_by_method[key] = models
    # Frozen covariance-whitened action geometry is the B0 score surface.
    action_score = action_distance_matrix(project_root, output_root)
    bundle["B0_CURRENT_CONTACT_KMEANS"] = {
        "scores": np.broadcast_to(action_score[None], cache["true_distance"].shape).copy(),
        "seed_scores": [],
        "entries": [],
        "latency_ms_per_query": 0.0,
        "modulation_norm_by_state": np.zeros(len(cache["context"]), dtype=np.float64),
        "condition_number_by_state": np.ones(len(cache["context"]), dtype=np.float64),
    }
    return bundle, models_by_method, manifest


def _candidate_action_distance(output_root):
    with np.load(os.path.join(output_root, "LOCAL_BANK.npz"), allow_pickle=False) as data:
        candidates = np.asarray(data["residuals"], dtype=np.float64)
        whitener = np.asarray(data["train_covariance_whitener"], dtype=np.float64)
    difference = candidates[:, None, :] - candidates[None, :, :]
    whitened = np.einsum("ijd,ed->ije", difference, whitener)
    distance = np.sqrt(np.mean(whitened ** 2, axis=-1))
    np.fill_diagonal(distance, 0.0)
    return distance


def _candidate_metric_distance(model, context, nominal, candidates, device):
    import torch

    values = np.asarray(candidates, dtype=np.float32)
    with torch.no_grad():
        candidate = torch.as_tensor(values, device=device)
        nominal_tensor = torch.as_tensor(np.repeat(nominal[None], len(values), axis=0), device=device)
        z = model.base.encode(nominal_tensor, candidate) if hasattr(model, "base") else model.encode(nominal_tensor, candidate)
        difference = z[:, None, :] - z[None, :, :]
        if hasattr(model, "base"):
            context_tensor = torch.as_tensor(context[None], device=device)
            weight = model.positive_weight(context_tensor)[0]
        else:
            weight = model.positive_weight()
        distance = torch.sum(weight[None, None, :] * difference ** 2, dim=-1)
    result = distance.cpu().numpy().astype(np.float64)
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)
    return result


def _atlas_decodings(output_root, cache, bundle, models_by_method, device):
    source_ids = np.asarray(cache["candidate_source_index"], dtype=np.int64)
    candidates = np.asarray(cache["candidate_residual"], dtype=np.float32)
    action_distance = _candidate_action_distance(output_root)
    output = {}
    atlas = {}
    for method, values in bundle.items():
        full = np.argmin(values["scores"], axis=2).astype(np.int64)
        k64 = np.empty_like(full)
        state_atlases = []
        for state in range(len(cache["context"])):
            if method == "B0_CURRENT_CONTACT_KMEANS":
                candidate_distance = action_distance
            else:
                per_seed = []
                for model in models_by_method[method]:
                    if method.startswith("CONTROL_"):
                        control = method[len("CONTROL_") :]
                        context, nominal = _evaluation_inputs(cache, control)
                    else:
                        context = cache["context"]
                        nominal = cache["nominal_action"]
                    per_seed.append(
                        _candidate_metric_distance(
                            model,
                            np.asarray(context[state], dtype=np.float32),
                            np.asarray(nominal[state], dtype=np.float32),
                            candidates,
                            device,
                        )
                    )
                candidate_distance = np.mean(per_seed, axis=0)
            medoids = deterministic_kmedoids_precomputed(
                candidate_distance, PRIMARY_K, source_ids
            )
            k64[state] = _assign(values["scores"][state], medoids, source_ids)
            state_atlases.append(medoids.astype(np.int64))
        output[method] = {"FULL": full, "K64": k64}
        atlas[method] = np.stack(state_atlases)
    return output, atlas


def _ranking_rows(cache, bundle, decoded, atlases, split):
    rows = []
    truth = np.asarray(cache["true_distance"], dtype=np.float64)
    source_ids = np.asarray(cache["candidate_source_index"], dtype=np.int64)
    for method, paths in decoded.items():
        for path, selected_matrix in paths.items():
            method_name = method + "__" + path
            for state in range(len(truth)):
                medoid_set = set(atlases[method][state].tolist()) if path == "K64" else None
                for target in range(truth.shape[1]):
                    score = np.asarray(bundle[method]["scores"][state, target], dtype=np.float64).copy()
                    if medoid_set is not None:
                        keep = np.asarray(sorted(medoid_set), dtype=np.int64)
                        masked = np.full_like(score, np.max(score) + max(np.ptp(score), 1.0) * 1000.0)
                        masked[keep] = score[keep]
                        score = masked
                    selected = int(selected_matrix[state, target])
                    value = ranking_metrics(truth[state, target], score, selected)
                    value.update(
                        {
                            "split": split,
                            "method": method_name,
                            "base_method": method,
                            "retrieval_path": path,
                            "task_id": str(cache["task_id"][state]),
                            "episode_id": int(cache["episode_id"][state]),
                            "phase": str(cache["phase"][state]),
                            "state_key": str(cache["key"][state]),
                            "state_index": int(state),
                            "target_id": int(target),
                            "direction_family_id": int(cache["direction_family_id"][target]),
                            "selected_source_index": int(source_ids[selected]),
                            "atlas_size": LOCAL_BANK_SIZE if path == "FULL" else PRIMARY_K,
                            "valid_bank_size": LOCAL_BANK_SIZE,
                            "context_modulation_norm": float(bundle[method]["modulation_norm_by_state"][state]),
                            "metric_condition_number": float(bundle[method]["condition_number_by_state"][state]),
                            "inference_latency_ms": float(bundle[method]["latency_ms_per_query"]),
                        }
                    )
                    rows.append(value)
    return rows


def _realized_rows(records, cache, decoded, bundle, split):
    keys = [str(record["meta"]["key"]) for record in records]
    if keys != cache["key"].astype(str).tolist():
        raise RuntimeError("record/cache order mismatch for " + split)
    source_ids = np.asarray(cache["candidate_source_index"], dtype=np.int64)
    scale = np.asarray(cache["consequence_scale"], dtype=np.float64)
    rows = []
    for state, record in enumerate(records):
        for method, paths in decoded.items():
            for path, selected_matrix in paths.items():
                method_name = method + "__" + path
                local = np.asarray(selected_matrix[state], dtype=np.int64)
                original = source_ids[local]
                produced = realized_rows(
                    record,
                    original,
                    method_name,
                    scale,
                    latency_ms=float(bundle[method]["latency_ms_per_query"]),
                    extra={
                        "split": split,
                        "evidence": (
                            "STAGE5_DEVELOPMENT"
                            if split == "development"
                            else "STAGE5_HISTORICAL_EXPLORATORY"
                        ),
                        "base_method": method,
                        "retrieval_path": path,
                        "atlas_size": LOCAL_BANK_SIZE if path == "FULL" else PRIMARY_K,
                        "valid_bank_size": LOCAL_BANK_SIZE,
                    },
                )
                for target, row in enumerate(produced):
                    row["local_bank_index"] = int(local[target])
                    row["source_bank_index"] = int(original[target])
                    row["direction_family_id"] = int(cache["direction_family_id"][target])
                    row["target_id"] = int(target)
                rows.extend(produced)
    return rows


RANK_METRICS = (
    "pairwise_accuracy",
    "candidate_distance_spearman",
    "kendall_tau",
    "ndcg_at_16",
    "oracle_neighbor_recall_at_1",
    "oracle_neighbor_recall_at_8",
    "oracle_regret",
    "context_modulation_norm",
    "metric_condition_number",
    "inference_latency_ms",
)
REALIZED_METRICS = (
    "balanced_task_effect_error",
    "object_pose_error",
    "tcp_object_relative_pose_error",
    "contact_mode_preserved",
    "task_progress_abs_error",
    "action_reconstruction_rmse",
    "clipped",
    "valid_bank_size",
    "inference_latency_ms",
    "error_group_object_pose",
    "error_group_tcp_object_relative_pose",
    "error_group_contact_constraint",
    "error_group_gripper_articulation",
    "error_group_task_progress",
)


def _summaries(rows, metrics, utilization=False):
    methods = sorted({str(row["method"]) for row in rows})
    partitions = [("pooled", "ALL", "ALL")]
    partitions += [("task", task, "ALL") for task in TASK_IDS]
    partitions += [("phase", "ALL", phase) for phase in PHASES]
    output = []
    for method in methods:
        for level, task, phase in partitions:
            selected = [
                row for row in rows
                if row["method"] == method
                and (task == "ALL" or row["task_id"] == task)
                and (phase == "ALL" or row["phase"] == phase)
            ]
            if not selected:
                continue
            summary = {
                "method": method,
                "level": level,
                "task_id": task,
                "phase": phase,
                "n": len(selected),
            }
            for metric in metrics:
                summary[metric] = float(np.mean([float(row[metric]) for row in selected]))
            if utilization:
                state_groups = defaultdict(list)
                for row in selected:
                    state_groups[(row["task_id"], int(row["episode_id"]), row["phase"])].append(
                        int(row["local_bank_index"])
                    )
                perplexities = []
                used = set()
                for indices in state_groups.values():
                    used.update(indices)
                    counts = np.bincount(indices, minlength=LOCAL_BANK_SIZE)
                    probability = counts[counts > 0] / float(np.sum(counts))
                    perplexities.append(float(np.exp(-np.sum(probability * np.log(probability)))))
                atlas_size = int(round(np.mean([row["atlas_size"] for row in selected])))
                summary["unique_codes"] = len(used)
                summary["code_perplexity"] = float(np.mean(perplexities))
                summary["normalized_code_utilization"] = float(np.mean(perplexities)) / max(atlas_size, 1)
                summary["atlas_size"] = atlas_size
            output.append(summary)
    return output


def _reversal_metrics(output_root, cache, bundle, split):
    import pandas as pd

    frame = pd.read_parquet(os.path.join(output_root, "CONTEXT_REVERSAL_PAIRS.parquet"))
    frame = frame[frame["split"].astype(str) == split]
    output = []
    for method, values in bundle.items():
        score = values["scores"]
        s1 = frame["state_s1"].to_numpy(dtype=np.int64)
        s2 = frame["state_s2"].to_numpy(dtype=np.int64)
        target = frame["target_id"].to_numpy(dtype=np.int64)
        i = frame["candidate_i"].to_numpy(dtype=np.int64)
        j = frame["candidate_j"].to_numpy(dtype=np.int64)
        side1 = score[s1, target, i] < score[s1, target, j]
        side2 = score[s2, target, j] < score[s2, target, i]
        output.append(
            {
                "method": method,
                "pairs": int(len(frame)),
                "side_s1_accuracy": float(np.mean(side1)),
                "side_s2_accuracy": float(np.mean(side2)),
                "side_accuracy": float(np.mean(np.concatenate((side1, side2)))),
                "joint_reversal_accuracy": float(np.mean(side1 & side2)),
            }
        )
    return output


def _method_summary(rows, method, metric, level="pooled", task="ALL"):
    value = next(
        row for row in rows
        if row["method"] == method and row["level"] == level and row["task_id"] == task
    )
    return float(value[metric])


def _gain(baseline, method):
    return (float(baseline) - float(method)) / max(float(baseline), 1e-12)


def _gate_development(output_root, cache, bundle, ranking_summary, realized_summary, realized_raw, reversal):
    p1_full = CONTEXT_METHOD + "__FULL"
    p1_k64 = CONTEXT_METHOD + "__K64"
    b2_full = "B2_STATIC_CONSEQUENCE__FULL"
    b2_k64 = "B2_STATIC_CONSEQUENCE__K64"
    p1_error = _method_summary(realized_summary, p1_full, "balanced_task_effect_error")
    b2_error = _method_summary(realized_summary, b2_full, "balanced_task_effect_error")
    p1_gain = _gain(b2_error, p1_error)
    bootstrap_p1 = paired_episode_bootstrap(
        realized_raw, p1_full, b2_full, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
    )
    p1_task_gains = {
        task: _gain(
            _method_summary(realized_summary, b2_full, "balanced_task_effect_error", "task", task),
            _method_summary(realized_summary, p1_full, "balanced_task_effect_error", "task", task),
        )
        for task in TASK_IDS
    }
    p1_regret = _method_summary(ranking_summary, p1_full, "oracle_regret")
    b2_regret = _method_summary(ranking_summary, b2_full, "oracle_regret")
    p1_ndcg = _method_summary(ranking_summary, p1_full, "ndcg_at_16")
    b2_ndcg = _method_summary(ranking_summary, b2_full, "ndcg_at_16")
    reversal_by_method = {row["method"]: row for row in reversal}
    p1_reversal = reversal_by_method[CONTEXT_METHOD]["joint_reversal_accuracy"]
    b2_reversal = reversal_by_method["B2_STATIC_CONSEQUENCE"]["joint_reversal_accuracy"]
    # Per-seed direction is computed from the frozen cache truth without model
    # averaging, which is identical to the primary realized metric definition.
    seed_directions = []
    truth = np.asarray(cache["true_distance"], dtype=np.float64)
    for seed_index, seed in enumerate(MODEL_SEEDS):
        p1_selected = np.argmin(bundle[CONTEXT_METHOD]["seed_scores"][seed_index], axis=2)
        b2_selected = np.argmin(bundle["B2_STATIC_CONSEQUENCE"]["seed_scores"][seed_index], axis=2)
        grid = np.indices(p1_selected.shape)
        p1_value = float(np.mean(truth[grid[0], grid[1], p1_selected]))
        b2_value = float(np.mean(truth[grid[0], grid[1], b2_selected]))
        seed_directions.append({"seed": int(seed), "gain": _gain(b2_value, p1_value), "improved": p1_value < b2_value})
    control_retention = {}
    increment = b2_error - p1_error
    for control in ("ACTION_ONLY", "JOINT_STATE_NOMINAL_SHUFFLED", "CONSEQUENCE_LABEL_SHUFFLED", "NO_REVERSAL_LOSS"):
        method = "CONTROL_%s__FULL" % control
        value = _method_summary(realized_summary, method, "balanced_task_effect_error")
        control_retention[control] = (b2_error - value) / max(increment, 1e-12)
    gate1_checks = {
        "realized_pooled_gain": {"value": p1_gain, "threshold": GATES["context_identifiable"]["realized_pooled_gain_min"], "passed": p1_gain >= GATES["context_identifiable"]["realized_pooled_gain_min"]},
        "paired_ci_lower": {"value": float(bootstrap_p1["pooled"]["ci95"][0]), "threshold": 0.0, "passed": float(bootstrap_p1["pooled"]["ci95"][0]) > 0.0},
        "tasks_improved": {"value": int(sum(value > 0 for value in p1_task_gains.values())), "threshold": 3, "passed": sum(value > 0 for value in p1_task_gains.values()) >= 3},
        "contact_tasks_improved": {"value": int(sum(p1_task_gains[task] > 0 for task in CONTACT_SENSITIVE_TASKS)), "threshold": 2, "passed": sum(p1_task_gains[task] > 0 for task in CONTACT_SENSITIVE_TASKS) >= 2},
        "oracle_regret_reduction": {"value": _gain(b2_regret, p1_regret), "threshold": 0.10, "passed": _gain(b2_regret, p1_regret) >= 0.10},
        "ndcg16_gain": {"value": p1_ndcg - b2_ndcg, "threshold": 0.05, "passed": p1_ndcg - b2_ndcg >= 0.05},
        "joint_reversal_accuracy": {"value": p1_reversal, "threshold": 0.35, "passed": p1_reversal >= 0.35},
        "joint_reversal_gain": {"value": p1_reversal - b2_reversal, "threshold": 0.15, "passed": p1_reversal - b2_reversal >= 0.15},
        "all_seed_directions": {"value": [row["improved"] for row in seed_directions], "threshold": [True, True, True], "passed": all(row["improved"] for row in seed_directions)},
        "joint_shuffle_retention": {"value": control_retention["JOINT_STATE_NOMINAL_SHUFFLED"], "threshold_max": 0.25, "passed": control_retention["JOINT_STATE_NOMINAL_SHUFFLED"] <= 0.25},
        "label_shuffle_retention": {"value": control_retention["CONSEQUENCE_LABEL_SHUFFLED"], "threshold_max": 0.25, "passed": control_retention["CONSEQUENCE_LABEL_SHUFFLED"] <= 0.25},
        "action_only_retention": {"value": control_retention["ACTION_ONLY"], "threshold_max": 0.50, "passed": control_retention["ACTION_ONLY"] <= 0.50},
        "no_reversal_does_not_reproduce": {"value": control_retention["NO_REVERSAL_LOSS"], "threshold_max": 1.0, "passed": control_retention["NO_REVERSAL_LOSS"] < 1.0},
    }
    gate1_passed = all(row["passed"] for row in gate1_checks.values())
    # Static B2 screen against the stronger deployable B0/B1 comparator.
    static_candidates = ("B0_CURRENT_CONTACT_KMEANS__FULL", "B1_ACTION_ONLY__FULL")
    static_baseline = min(static_candidates, key=lambda method: _method_summary(realized_summary, method, "balanced_task_effect_error"))
    static_error = _method_summary(realized_summary, static_baseline, "balanced_task_effect_error")
    static_gain = _gain(static_error, b2_error)
    static_bootstrap = paired_episode_bootstrap(realized_raw, b2_full, static_baseline, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED + 1)
    static_task_gains = {
        task: _gain(
            _method_summary(realized_summary, static_baseline, "balanced_task_effect_error", "task", task),
            _method_summary(realized_summary, b2_full, "balanced_task_effect_error", "task", task),
        ) for task in TASK_IDS
    }
    static_regret = _method_summary(ranking_summary, static_baseline, "oracle_regret")
    static_ndcg = _method_summary(ranking_summary, static_baseline, "ndcg_at_16")
    static_checks = {
        "realized_pooled_gain": {"value": static_gain, "threshold": 0.05, "passed": static_gain >= 0.05},
        "paired_ci_lower": {"value": float(static_bootstrap["pooled"]["ci95"][0]), "threshold": 0.0, "passed": float(static_bootstrap["pooled"]["ci95"][0]) > 0.0},
        "tasks_improved": {"value": int(sum(value > 0 for value in static_task_gains.values())), "threshold": 3, "passed": sum(value > 0 for value in static_task_gains.values()) >= 3},
        "contact_tasks_improved": {"value": int(sum(static_task_gains[task] > 0 for task in CONTACT_SENSITIVE_TASKS)), "threshold": 2, "passed": sum(static_task_gains[task] > 0 for task in CONTACT_SENSITIVE_TASKS) >= 2},
        "oracle_regret_reduction": {"value": _gain(static_regret, b2_regret), "threshold": 0.10, "passed": _gain(static_regret, b2_regret) >= 0.10},
        "ndcg16_gain": {"value": b2_ndcg - static_ndcg, "threshold": 0.05, "passed": b2_ndcg - static_ndcg >= 0.05},
    }
    static_passed = all(row["passed"] for row in static_checks.values())
    # Gate 2 K=64 against the strongest deployable dynamic baseline.
    k64_candidates = (
        "B0_CURRENT_CONTACT_KMEANS__K64",
        "B1_ACTION_ONLY__K64",
        "B2_STATIC_CONSEQUENCE__K64",
    )
    k64_baseline = min(k64_candidates, key=lambda method: _method_summary(realized_summary, method, "balanced_task_effect_error"))
    k64_base_error = _method_summary(realized_summary, k64_baseline, "balanced_task_effect_error")
    k64_error = _method_summary(realized_summary, p1_k64, "balanced_task_effect_error")
    k64_gain = _gain(k64_base_error, k64_error)
    k64_task_gains = {
        task: _gain(
            _method_summary(realized_summary, k64_baseline, "balanced_task_effect_error", "task", task),
            _method_summary(realized_summary, p1_k64, "balanced_task_effect_error", "task", task),
        ) for task in TASK_IDS
    }
    base_rmse = _method_summary(realized_summary, k64_baseline, "action_reconstruction_rmse")
    p1_rmse = _method_summary(realized_summary, p1_k64, "action_reconstruction_rmse")
    base_contact = _method_summary(realized_summary, k64_baseline, "contact_mode_preserved")
    p1_contact = _method_summary(realized_summary, p1_k64, "contact_mode_preserved")
    utilization = _method_summary(realized_summary, p1_k64, "normalized_code_utilization")
    clipping = _method_summary(realized_summary, p1_k64, "clipped")
    valid = _method_summary(realized_summary, p1_k64, "valid_bank_size")
    retention = k64_gain / max(p1_gain, 1e-12)
    gate2_checks = {
        "realized_gain": {"value": k64_gain, "threshold": 0.08, "passed": k64_gain >= 0.08},
        "full_gain_retention": {"value": retention, "threshold": 0.75, "passed": retention >= 0.75},
        "tasks_improved": {"value": int(sum(value > 0 for value in k64_task_gains.values())), "threshold": 3, "passed": sum(value > 0 for value in k64_task_gains.values()) >= 3},
        "contact_tasks_improved": {"value": int(sum(k64_task_gains[task] > 0 for task in CONTACT_SENSITIVE_TASKS)), "threshold": 2, "passed": sum(k64_task_gains[task] > 0 for task in CONTACT_SENSITIVE_TASKS) >= 2},
        "action_rmse_degradation": {"value": (p1_rmse - base_rmse) / max(base_rmse, 1e-12), "threshold_max": 0.20, "passed": (p1_rmse - base_rmse) / max(base_rmse, 1e-12) <= 0.20},
        "contact_preservation_drop": {"value": base_contact - p1_contact, "threshold_max": 0.01, "passed": base_contact - p1_contact <= 0.01},
        "normalized_utilization": {"value": utilization, "threshold": 0.25, "passed": utilization >= 0.25},
        "clipping": {"value": clipping, "threshold_max": 0.0, "passed": clipping <= 0.0},
        "valid_bank_size": {"value": valid, "threshold": 96, "passed": valid >= 96},
    }
    gate2_passed = all(row["passed"] for row in gate2_checks.values())
    return {
        "gate1_context_identifiable": {"passed": gate1_passed, "checks": gate1_checks, "task_gains": p1_task_gains, "seed_directions": seed_directions, "control_gain_retention": control_retention, "bootstrap": bootstrap_p1},
        "static_consequence_value": {"passed": static_passed, "baseline": static_baseline, "checks": static_checks, "task_gains": static_task_gains, "bootstrap": static_bootstrap},
        "gate2_dynamic_k64": {"passed": gate2_passed, "baseline": k64_baseline, "checks": gate2_checks, "task_gains": k64_task_gains},
    }


def evaluate_split(project_root, split, output_root=None, scratch_root=SCRATCH_ROOT, device_name="cpu"):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    if split not in ("development", "historical_exploratory"):
        raise KeyError(split)
    if split == "historical_exploratory" and not os.path.isfile(os.path.join(output_root, "MODEL_SELECTION.json")):
        raise RuntimeError("development choices must be frozen before exploratory evaluation")
    device = _device(device_name)
    cache = load_cache(cache_path(scratch_root, split))
    record_split = "development" if split == "development" else "confirmation"
    records = historical_records(record_split)
    bundle, models_by_method, manifest = _load_score_bundle(project_root, output_root, cache, device)
    decoded, atlases = _atlas_decodings(output_root, cache, bundle, models_by_method, device)
    ranking_raw = _ranking_rows(cache, bundle, decoded, atlases, split)
    realized_raw = _realized_rows(records, cache, decoded, bundle, split)
    ranking_summary = _summaries(ranking_raw, RANK_METRICS)
    realized_summary = _summaries(realized_raw, REALIZED_METRICS, utilization=True)
    reversal = (
        _reversal_metrics(output_root, cache, bundle, "development")
        if split == "development"
        else []
    )
    import pandas as pd

    prefix = "DEVELOPMENT" if split == "development" else "HISTORICAL_EXPLORATORY"
    pd.DataFrame(ranking_raw).to_parquet(os.path.join(output_root, prefix.lower() + "_ranking_rows.parquet"), index=False)
    pd.DataFrame(realized_raw).to_parquet(os.path.join(output_root, prefix.lower() + "_realized_rows.parquet"), index=False)
    write_csv(os.path.join(output_root, prefix + "_RANKING.csv"), ranking_summary)
    write_csv(os.path.join(output_root, prefix + "_REALIZED.csv"), realized_summary)
    controls = [row for row in ranking_summary + realized_summary if row["method"].startswith("CONTROL_")]
    for path in ("FULL", "K64"):
        proposed = decoded[CONTEXT_METHOD][path]
        for baseline in ("B2_STATIC_CONSEQUENCE",) + tuple("CONTROL_" + value for value in MATCHED_CONTROLS):
            controls.append(
                {
                    "row_type": "selected_code_intervention",
                    "method": CONTEXT_METHOD + "__" + path,
                    "intervention": baseline,
                    "level": "pooled",
                    "task_id": "ALL",
                    "phase": "ALL",
                    "selected_code_change_fraction": float(np.mean(proposed != decoded[baseline][path])),
                    "n": int(proposed.size),
                }
            )
    controls += [{**row, "level": "reversal", "task_id": "ALL", "phase": "ALL"} for row in reversal]
    write_csv(os.path.join(output_root, prefix + "_CONTROLS.csv"), controls)
    atlas_payload = {
        method: {
            "state_keys": cache["key"].astype(str).tolist(),
            "local_medoids": values.tolist(),
            "source_medoids": np.asarray(cache["candidate_source_index"], dtype=np.int64)[values].tolist(),
        }
        for method, values in atlases.items()
    }
    atomic_json(os.path.join(output_root, prefix + "_ATLASES.json"), atlas_payload)
    result = {
        "split": split,
        "states": int(len(cache["context"])),
        "targets_per_state": int(cache["true_distance"].shape[1]),
        "candidate_count": int(cache["true_distance"].shape[2]),
        "methods": sorted(decoded),
        "reversal": reversal,
    }
    if split == "development":
        gate = _gate_development(output_root, cache, bundle, ranking_summary, realized_summary, realized_raw, reversal)
        atomic_json(os.path.join(output_root, "DEVELOPMENT_GATE.json"), gate)
        selected_entries = []
        for entry in manifest["entries"]:
            metadata = entry["metadata"]
            if metadata["method"] == "B2_STATIC_CONSEQUENCE" and float(metadata["temperature"]) != float(manifest["selected_temperature"]):
                continue
            selected_entries.append({"path": entry["path"], "sha256": entry["sha256"], "method": metadata["method"], "control": metadata.get("control"), "seed": metadata["seed"]})
        selection = {
            "selected_architecture": CONTEXT_METHOD,
            "selected_temperature": float(manifest["selected_temperature"]),
            "selected_proposed_path": CONTEXT_METHOD + "__K64",
            "primary_k": PRIMARY_K,
            "local_bank_size": LOCAL_BANK_SIZE,
            "checkpoint_entries": selected_entries,
            "checkpoint_count": len(selected_entries),
            "model_training_manifest_sha256": sha256_file(os.path.join(output_root, "MODEL_TRAINING_MANIFEST.json")),
            "development_gate": gate,
            "fresh_confirmation_choices_frozen": True,
            "development_may_not_refit": True,
        }
        atomic_json(os.path.join(output_root, "MODEL_SELECTION.json"), selection)
        result["gate"] = gate
    atomic_json(os.path.join(output_root, prefix + "_EVALUATION.json"), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("development", "historical_exploratory"))
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--scratch-root", default=SCRATCH_ROOT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    result = evaluate_split(args.project_root, args.split, args.output_root, args.scratch_root, args.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
