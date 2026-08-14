"""Ranking, realized-effect, utilization, and bootstrap metrics for Stage 3."""

from __future__ import annotations

import csv
import math
import os

import numpy as np

from .stage2_analysis import (
    ACTION_BANK_SIZE,
    PRIMARY_GROUPS,
    _evaluate_decoded,
    effect_embedding,
)
from .stage3_config import PRIMARY_K
from .stage3_data import effect, true_distance_matrix


def stable_fps(values, k, frozen_ids=None):
    """Order-invariant FPS with ties resolved by the lowest frozen ID."""
    values = np.asarray(values, dtype=np.float64)
    ids = (
        np.arange(len(values), dtype=np.int64)
        if frozen_ids is None
        else np.asarray(frozen_ids, dtype=np.int64)
    )
    if len(values) < int(k) or len(ids) != len(values):
        raise ValueError("bad FPS inputs")
    position_by_id = {int(identifier): index for index, identifier in enumerate(ids)}
    first_id = int(np.min(ids))
    chosen_positions = [position_by_id[first_id]]
    chosen_ids = [first_id]
    minimum = np.sum((values - values[chosen_positions[0]]) ** 2, axis=1)
    minimum[chosen_positions[0]] = -1.0
    while len(chosen_positions) < int(k):
        maximum = float(np.max(minimum))
        tied = np.flatnonzero(np.isclose(minimum, maximum, rtol=0.0, atol=1e-15))
        position = int(tied[np.argmin(ids[tied])])
        chosen_positions.append(position)
        chosen_ids.append(int(ids[position]))
        distance = np.sum((values - values[position]) ** 2, axis=1)
        minimum = np.minimum(minimum, distance)
        minimum[np.asarray(chosen_positions, dtype=np.int64)] = -1.0
    return np.asarray(chosen_ids, dtype=np.int64)


def nearest_by_distance(target, candidate, candidate_ids=None):
    target = np.asarray(target, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    ids = (
        np.arange(len(candidate), dtype=np.int64)
        if candidate_ids is None
        else np.asarray(candidate_ids, dtype=np.int64)
    )
    distance = np.sum((target[:, None, :] - candidate[None, :, :]) ** 2, axis=2)
    decoded = []
    for row in distance:
        minimum = float(np.min(row))
        tied = np.flatnonzero(np.isclose(row, minimum, rtol=0.0, atol=1e-15))
        decoded.append(int(np.min(ids[tied])))
    return np.asarray(decoded, dtype=np.int64)


def argmin_stable(values, ids=None):
    values = np.asarray(values, dtype=np.float64)
    ids = np.arange(len(values), dtype=np.int64) if ids is None else np.asarray(ids)
    minimum = float(np.min(values))
    tied = np.flatnonzero(np.isclose(values, minimum, rtol=0.0, atol=1e-15))
    return int(np.min(ids[tied]))


def true_oracle_decoded(record, consequence_scale, k=PRIMARY_K):
    candidate_effect = effect(record["candidate"])[1:]
    candidate_mask = np.asarray(record["candidate"]["mask"][1:], dtype=bool)
    candidate_mode = np.asarray(record["candidate"]["contact_mode"][1:], dtype=np.int64)
    target_effect = effect(record["support"])[1:]
    target_mask = np.asarray(record["support"]["mask"][1:], dtype=bool)
    target_mode = np.asarray(record["support"]["contact_mode"][1:], dtype=np.int64)
    candidate_embedding = effect_embedding(
        candidate_effect, candidate_mask, candidate_mode, consequence_scale
    )
    target_embedding = effect_embedding(
        target_effect, target_mask, target_mode, consequence_scale
    )
    atlas = stable_fps(candidate_embedding, int(k))
    decoded = nearest_by_distance(
        target_embedding, candidate_embedding[atlas], candidate_ids=atlas
    )
    return decoded, atlas


def full_oracle_decoded(record, consequence_scale):
    matrix = true_distance_matrix(record, consequence_scale)
    return np.asarray([argmin_stable(row) for row in matrix], dtype=np.int64)


def realized_rows(record, decoded, method, consequence_scale, latency_ms=0.0, extra=None):
    values = dict(extra or {})
    values["inference_latency_ms"] = float(latency_ms)
    rows = _evaluate_decoded(record, decoded, method, consequence_scale, extra=values)
    for row in rows:
        row["object_pose_error"] = row["error_group_object_pose"]
        row["tcp_object_relative_pose_error"] = row[
            "error_group_tcp_object_relative_pose"
        ]
    return rows


def _rankdata(values):
    from scipy.stats import rankdata

    return rankdata(np.asarray(values, dtype=np.float64), method="average")


def ranking_metrics(true_distance, predicted_distance, selected_bank_index=None, tau=0.15):
    from scipy.stats import kendalltau, spearmanr

    true_distance = np.asarray(true_distance, dtype=np.float64)
    predicted_distance = np.asarray(predicted_distance, dtype=np.float64)
    true_order = np.argsort(true_distance, kind="mergesort")
    predicted_order = np.lexsort((np.arange(len(predicted_distance)), predicted_distance))
    selected = (
        int(predicted_order[0])
        if selected_bank_index is None
        else int(selected_bank_index)
    )
    if np.ptp(true_distance) == 0.0 or np.ptp(predicted_distance) == 0.0:
        spear = 0.0
        kendall = 0.0
    else:
        spear = spearmanr(true_distance, predicted_distance).statistic
        kendall = kendalltau(true_distance, predicted_distance, variant="b").statistic
    if not np.isfinite(spear):
        spear = 0.0
    if not np.isfinite(kendall):
        kendall = 0.0
    relevance = np.exp(-true_distance / float(tau))
    limit = min(16, len(true_distance))
    discount = 1.0 / np.log2(np.arange(2, 2 + limit))
    dcg = float(np.sum(relevance[predicted_order[:limit]] * discount))
    ideal = float(np.sum(relevance[true_order[:limit]] * discount))
    top1 = int(predicted_order[0] == true_order[0])
    recall8 = float(len(set(predicted_order[:8].tolist()) & set(true_order[:8].tolist()))) / 8.0
    return {
        "pairwise_accuracy": float(0.5 * (kendall + 1.0)),
        "candidate_distance_spearman": float(spear),
        "kendall_tau": float(kendall),
        "ndcg_at_16": dcg / max(ideal, 1e-12),
        "oracle_neighbor_recall_at_1": top1,
        "oracle_neighbor_recall_at_8": recall8,
        "oracle_regret": float(true_distance[selected] - np.min(true_distance)),
        "selected_bank_index": selected,
    }


def summarize_retrieval(rows, baseline_method=None):
    output = []
    methods = sorted({row["method"] for row in rows})
    partitions = [("pooled", "ALL", "ALL", "ALL")]
    partitions += [("task", task, "ALL", "ALL") for task in sorted({r["task_id"] for r in rows})]
    partitions += [("phase", "ALL", phase, "ALL") for phase in sorted({r["phase"] for r in rows})]
    partitions += [
        ("direction_family", "ALL", "ALL", family)
        for family in sorted({str(r["direction_family_id"]) for r in rows})
    ]
    baseline = {}
    if baseline_method:
        for level, task, phase, family in partitions:
            selected = [
                row
                for row in rows
                if row["method"] == baseline_method
                and (task == "ALL" or row["task_id"] == task)
                and (phase == "ALL" or row["phase"] == phase)
                and (family == "ALL" or str(row["direction_family_id"]) == family)
            ]
            if selected:
                baseline[(level, task, phase, family)] = float(
                    np.mean([row["oracle_regret"] for row in selected])
                )
    metric_names = (
        "pairwise_accuracy",
        "candidate_distance_spearman",
        "kendall_tau",
        "ndcg_at_16",
        "oracle_neighbor_recall_at_1",
        "oracle_neighbor_recall_at_8",
        "oracle_regret",
        "inference_latency_ms",
    )
    for method in methods:
        for level, task, phase, family in partitions:
            selected = [
                row
                for row in rows
                if row["method"] == method
                and (task == "ALL" or row["task_id"] == task)
                and (phase == "ALL" or row["phase"] == phase)
                and (family == "ALL" or str(row["direction_family_id"]) == family)
            ]
            if not selected:
                continue
            summary = {
                "split": selected[0]["split"],
                "method": method,
                "level": level,
                "task_id": task,
                "phase": phase,
                "direction_family_id": family,
                "n": len(selected),
            }
            for metric in metric_names:
                summary[metric] = float(np.mean([row[metric] for row in selected]))
            key = (level, task, phase, family)
            if key in baseline:
                base = baseline[key]
                summary["o1_baseline_gap_fraction_closed"] = (
                    (base - summary["oracle_regret"]) / base if base > 0 else 0.0
                )
            else:
                summary["o1_baseline_gap_fraction_closed"] = float("nan")
            output.append(summary)
    return output


def summarize_realized(rows, k=PRIMARY_K):
    output = []
    methods = sorted({row["method"] for row in rows})
    partitions = [("pooled", "ALL", "ALL")]
    partitions += [("task", task, "ALL") for task in sorted({r["task_id"] for r in rows})]
    partitions += [("phase", "ALL", phase) for phase in sorted({r["phase"] for r in rows})]
    metric_names = (
        "balanced_task_effect_error",
        "object_pose_error",
        "tcp_object_relative_pose_error",
        "contact_mode_preserved",
        "task_progress_abs_error",
        "action_reconstruction_rmse",
        "clipped",
        "inference_latency_ms",
    )
    metric_names += tuple("error_group_" + name for name in PRIMARY_GROUPS)
    for method in methods:
        for level, task, phase in partitions:
            selected = [
                row
                for row in rows
                if row["method"] == method
                and (task == "ALL" or row["task_id"] == task)
                and (phase == "ALL" or row["phase"] == phase)
            ]
            if not selected:
                continue
            decoded = np.asarray(
                [row["decoded_bank_index"] for row in selected if row["decoded_bank_index"] >= 0],
                dtype=np.int64,
            )
            counts = np.bincount(decoded, minlength=ACTION_BANK_SIZE) if len(decoded) else np.zeros(ACTION_BANK_SIZE)
            probability = counts[counts > 0] / max(float(np.sum(counts)), 1.0)
            perplexity = (
                float(np.exp(-np.sum(probability * np.log(probability))))
                if len(probability)
                else 0.0
            )
            state_perplexities = []
            state_keys = sorted(
                {(row["task_id"], row["episode_id"], row["phase"]) for row in selected}
            )
            for state_key in state_keys:
                indices = [
                    row["decoded_bank_index"]
                    for row in selected
                    if (row["task_id"], row["episode_id"], row["phase"]) == state_key
                    and row["decoded_bank_index"] >= 0
                ]
                if indices:
                    state_counts = np.bincount(indices, minlength=ACTION_BANK_SIZE)
                    p = state_counts[state_counts > 0] / float(np.sum(state_counts))
                    state_perplexities.append(float(np.exp(-np.sum(p * np.log(p)))))
            summary = {
                "split": selected[0]["split"],
                "method": method,
                "level": level,
                "task_id": task,
                "phase": phase,
                "n": len(selected),
                "unique_codes": int(np.sum(counts > 0)),
                "code_perplexity": perplexity,
                "normalized_code_utilization": (
                    float(np.mean(state_perplexities)) / float(k)
                    if state_perplexities
                    else 0.0
                ),
            }
            for metric in metric_names:
                summary[metric] = float(np.mean([row[metric] for row in selected]))
            output.append(summary)
    return output


def write_csv(path, rows, fieldnames=None, float_significant_digits=12):
    """Write ordinary CSV while keeping large row-level artifacts Git-friendly.

    Twelve significant digits preserve substantially more precision than the
    simulator/replay tolerances used for statistical reporting, while avoiding
    Python's variable 17-digit float representation in hundreds of thousands
    of repeated quantization rows.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({name for row in rows for name in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = {
                name: (
                    format(value, ".%dg" % int(float_significant_digits))
                    if isinstance(value, (float, np.floating)) and np.isfinite(value)
                    else value
                )
                for name, value in row.items()
            }
            writer.writerow(serialized)


def paired_episode_bootstrap(rows, method, baseline, replicates, seed):
    selected = [row for row in rows if row["method"] in (method, baseline)]
    tasks = sorted({row["task_id"] for row in selected})
    episodes_by_task = {
        task: sorted({int(row["episode_id"]) for row in selected if row["task_id"] == task})
        for task in tasks
    }
    lookup = {}
    for row in selected:
        key = (row["method"], row["task_id"], int(row["episode_id"]))
        lookup.setdefault(key, []).append(float(row["balanced_task_effect_error"]))
    episode_difference = {}
    for task in tasks:
        for episode in episodes_by_task[task]:
            method_value = np.mean(lookup[(method, task, episode)])
            baseline_value = np.mean(lookup[(baseline, task, episode)])
            episode_difference[(task, episode)] = float(baseline_value - method_value)
    rng = np.random.RandomState(int(seed))
    draws = np.empty(int(replicates), dtype=np.float64)
    task_draws = dict((task, np.empty(int(replicates), dtype=np.float64)) for task in tasks)
    for replicate in range(int(replicates)):
        pooled = []
        for task in tasks:
            episodes = episodes_by_task[task]
            sampled = rng.choice(episodes, size=len(episodes), replace=True)
            values = [episode_difference[(task, int(episode))] for episode in sampled]
            task_draws[task][replicate] = float(np.mean(values))
            pooled.extend(values)
        draws[replicate] = float(np.mean(pooled))
    point = float(np.mean(list(episode_difference.values())))
    return {
        "method": method,
        "baseline": baseline,
        "replicates": int(replicates),
        "seed": int(seed),
        "paired_difference_definition": "baseline error - method error",
        "pooled": {
            "point": point,
            "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "probability_positive": float(np.mean(draws > 0.0)),
        },
        "by_task": {
            task: {
                "point": float(
                    np.mean(
                        [
                            value
                            for (row_task, _), value in episode_difference.items()
                            if row_task == task
                        ]
                    )
                ),
                "ci95": [
                    float(np.percentile(task_draws[task], 2.5)),
                    float(np.percentile(task_draws[task], 97.5)),
                ],
            }
            for task in tasks
        },
    }
