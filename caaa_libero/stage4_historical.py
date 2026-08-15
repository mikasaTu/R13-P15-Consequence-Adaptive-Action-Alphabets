"""Frozen-checkpoint Stage 4 failure decomposition and context audit.

All simulator outcomes consumed here were already generated in Stage 3.  The
module never launches LIBERO and never retrains a model.  It exposes the exact
C3 full-bank path that Stage 3 did not report separately from C3 K=64.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from collections import defaultdict

import numpy as np

from .stage2_analysis import CONTINUOUS_INDICES, PRIMARY_GROUPS, balanced_error, effect_embedding
from .stage3_analysis import (
    _action_assign,
    _ensemble_embedding,
    _load_baseline_codebooks,
    load_trained_models,
)
from .stage3_config import SCRATCH_ROOT as HISTORICAL_SCRATCH_ROOT
from .stage3_data import (
    CONTEXT_SLICES,
    HISTORY_CONTROL_SLICES,
    STATE_CONTROL_SLICES,
    effect,
    load_records,
    raw_context,
    true_distance_matrix,
)
from .stage3_metrics import (
    argmin_stable,
    full_oracle_decoded,
    nearest_by_distance,
    realized_rows,
    stable_fps,
    true_oracle_decoded,
    write_csv,
)
from .stage4_config import (
    ACTION_BANK_SIZE,
    CR_PAIR_MARGIN_QUANTILE,
    HISTORICAL_REPOSITORY_ROOT,
    HISTORICAL_STAGE3_RELATIVE,
    OUTPUT_RELATIVE,
    PRIMARY_K,
    REVERSAL_PAIR_SEED,
)
from .storage import atomic_json, sha256_file


SHORT_METHODS = ("B2", "O_FULL", "O_K64", "C3_FULL", "C3_K64", "C5")
SPLIT_LABELS = {
    "development": "DEVELOPMENT_EPISODES_36_39",
    "confirmation": "HISTORICAL_EXPLORATORY_EPISODES_40_49",
}


def _device(name):
    import torch

    device = torch.device(name or "cpu")
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one local GPU for Stage 4")
    return device


def _pair_matrix(models, context, targets, candidates, device, batch_size=8192):
    """Vectorized ensemble pair-ranker matrix for one state."""
    import torch

    targets = np.asarray(targets, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    target_count = len(targets)
    candidate_count = len(candidates)
    flat_target = np.repeat(targets, candidate_count, axis=0) / 0.12
    flat_candidate = np.tile(candidates, (target_count, 1)) / 0.12
    flat_context = np.repeat(
        np.asarray(context, dtype=np.float32)[None, :],
        target_count * candidate_count,
        axis=0,
    )
    member_values = []
    for model in models:
        values = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(flat_target), int(batch_size)):
                stop = min(start + int(batch_size), len(flat_target))
                values.append(
                    model(
                        torch.as_tensor(flat_context[start:stop], device=device),
                        torch.as_tensor(flat_target[start:stop], device=device),
                        torch.as_tensor(flat_candidate[start:stop], device=device),
                    )
                    .cpu()
                    .numpy()
                )
        member_values.append(np.concatenate(values))
    return np.mean(np.stack(member_values), axis=0).reshape(
        target_count, candidate_count
    )


def _context_interventions(records, center, scale, seed=REVERSAL_PAIR_SEED):
    raw = np.stack([raw_context(record) for record in records])
    tasks = np.asarray([record["meta"]["task_id"] for record in records])
    rng = np.random.RandomState(int(seed))
    order = np.arange(len(records), dtype=np.int64)
    for task in sorted(set(tasks.tolist())):
        keep = np.flatnonzero(tasks == task)
        order[keep] = rng.permutation(keep)

    def normalized(value):
        return ((value - center[None, :]) / scale[None, :]).astype(np.float32)

    values = {"correct_context": normalized(raw)}
    changed = raw.copy()
    start, stop = CONTEXT_SLICES["nominal_action"]
    changed[:, start:stop] = 0.0
    values["nominal_zeroed"] = normalized(changed)

    changed = raw.copy()
    changed[:, start:stop] = raw[order, start:stop]
    values["nominal_shuffled_within_task"] = normalized(changed)

    changed = raw.copy()
    for name in STATE_CONTROL_SLICES:
        left, right = CONTEXT_SLICES[name]
        changed[:, left:right] = raw[order, left:right]
    values["state_mask_contact_shuffled_within_task"] = normalized(changed)

    changed = raw.copy()
    for name in HISTORY_CONTROL_SLICES:
        left, right = CONTEXT_SLICES[name]
        changed[:, left:right] = raw[order, left:right]
    values["history_actions_masks_shuffled_within_task"] = normalized(changed)

    changed = raw.copy()
    for name in STATE_CONTROL_SLICES + ("nominal_action",):
        left, right = CONTEXT_SLICES[name]
        changed[:, left:right] = raw[order, left:right]
    values["state_and_nominal_jointly_shuffled"] = normalized(changed)
    values["all_context_zeroed_action_pair_retained"] = np.zeros_like(
        values["correct_context"], dtype=np.float32
    )
    return values


def _support_embeddings(record, consequence_scale):
    target = effect(record["support"])[1:]
    target_mask = np.asarray(record["support"]["mask"][1:], dtype=bool)
    target_mode = np.asarray(record["support"]["contact_mode"][1:], dtype=np.int64)
    candidate = effect(record["candidate"])[1:]
    candidate_mask = np.asarray(record["candidate"]["mask"][1:], dtype=bool)
    candidate_mode = np.asarray(
        record["candidate"]["contact_mode"][1:], dtype=np.int64
    )
    return (
        effect_embedding(target, target_mask, target_mode, consequence_scale),
        effect_embedding(
            candidate, candidate_mask, candidate_mode, consequence_scale
        ),
    )


def _decomposition_branch_rows(
    records, contexts, models, codebooks, consequence_scale, device
):
    rows = []
    matrices = {}
    support_embeddings = {}
    for state_index, record in enumerate(records):
        meta = record["meta"]
        key = (meta["task_id"], int(meta["episode_id"]), meta["phase"])
        context = contexts[state_index]
        target = np.asarray(record["support"]["residual_action"][1:], dtype=np.float32)
        bank = np.asarray(record["candidate"]["residual_action"][1:], dtype=np.float32)
        true_matrix = true_distance_matrix(record, consequence_scale)
        matrices[key] = true_matrix
        support_embeddings[key] = _support_embeddings(record, consequence_scale)
        contact = str(int(bool(record["context"]["current_contact"].item())))
        b2 = _action_assign(target, bank, codebooks["B2_contact_" + contact])
        o_full = full_oracle_decoded(record, consequence_scale)
        o_k64, _ = true_oracle_decoded(record, consequence_scale, k=PRIMARY_K)
        c3_bank = _ensemble_embedding(models["C3_NC_BIENCODER"], context, bank, device)
        c3_target = _ensemble_embedding(
            models["C3_NC_BIENCODER"], context, target, device
        )
        c3_matrix = np.sum(
            (c3_target[:, None, :] - c3_bank[None, :, :]) ** 2, axis=2
        )
        c3_full = np.asarray([argmin_stable(row) for row in c3_matrix])
        c3_atlas = stable_fps(c3_bank, PRIMARY_K)
        c3_k64 = np.asarray(
            [argmin_stable(row[c3_atlas], ids=c3_atlas) for row in c3_matrix]
        )
        c4_matrix = _pair_matrix(
            models["C4_NC_PAIR_RANKER"], context, target, bank, device
        )
        c5 = np.asarray(
            [argmin_stable(row[c3_atlas], ids=c3_atlas) for row in c4_matrix]
        )
        target_family = np.asarray(
            record["support"]["direction_family_id"][1:], dtype=np.int64
        )
        for method, decoded in (
            ("B2", b2),
            ("O_FULL", o_full),
            ("O_K64", o_k64),
            ("C3_FULL", c3_full),
            ("C3_K64", c3_k64),
            ("C5", c5),
        ):
            realized = realized_rows(record, decoded, method, consequence_scale)
            for target_id, row in enumerate(realized):
                row["direction_family_id"] = int(target_family[target_id])
                rows.append(row)
    return rows, matrices, support_embeddings


def _partitions(rows):
    yield "pooled", "ALL", "ALL", "ALL", rows
    for task in sorted({row["task_id"] for row in rows}):
        yield "task", task, "ALL", "ALL", [r for r in rows if r["task_id"] == task]
    for phase in sorted({row["phase"] for row in rows}):
        yield "phase", "ALL", phase, "ALL", [r for r in rows if r["phase"] == phase]
    for family in sorted({int(row["direction_family_id"]) for row in rows}):
        yield "direction_family", "ALL", "ALL", str(family), [
            r for r in rows if int(r["direction_family_id"]) == family
        ]


def _decomposition_summary(rows, split_label):
    output = []
    metrics = [("BALANCED_TASK_EFFECT", "balanced_task_effect_error")]
    metrics.extend(
        (name, "error_group_" + name) for name in PRIMARY_GROUPS
    )
    for level, task, phase, family, partition in _partitions(rows):
        by_method = {
            method: [row for row in partition if row["method"] == method]
            for method in SHORT_METHODS
        }
        for group, metric in metrics:
            means = {
                method: float(np.mean([row[metric] for row in by_method[method]]))
                for method in SHORT_METHODS
            }
            output.append(
                {
                    "split": split_label,
                    "level": level,
                    "task_id": task,
                    "phase": phase,
                    "direction_family_id": family,
                    "consequence_group": group,
                    "n_targets": len(by_method["B2"]),
                    "B2_error": means["B2"],
                    "O_FULL_error": means["O_FULL"],
                    "O_K64_error": means["O_K64"],
                    "C3_FULL_error": means["C3_FULL"],
                    "C3_K64_error": means["C3_K64"],
                    "C5_error": means["C5"],
                    "oracle_bank_compression_loss": means["O_K64"]
                    - means["O_FULL"],
                    "learned_metric_loss": means["C3_FULL"] - means["O_FULL"],
                    "learned_compression_loss": means["C3_K64"]
                    - means["C3_FULL"],
                    "c4_override_loss": means["C5"] - means["C3_K64"],
                }
            )
    return output


def _train_margins(train_records, consequence_scale):
    gaps = defaultdict(list)
    for record in train_records:
        matrix = true_distance_matrix(record, consequence_scale)
        meta = record["meta"]
        rng = np.random.RandomState(
            int.from_bytes(
                __import__("hashlib")
                .sha256(
                    ("%d|%s|%d|%s" % (
                        REVERSAL_PAIR_SEED,
                        meta["task_id"],
                        int(meta["episode_id"]),
                        meta["phase"],
                    )).encode("utf-8")
                )
                .digest()[:4],
                "little",
            )
        )
        for _ in range(64):
            target = int(rng.randint(matrix.shape[0]))
            left, right = rng.choice(matrix.shape[1], size=2, replace=False)
            gap = abs(float(matrix[target, left] - matrix[target, right]))
            if gap > 1e-12:
                gaps[(meta["task_id"], meta["phase"])].append(gap)
    margins = {
        key: float(np.quantile(values, CR_PAIR_MARGIN_QUANTILE))
        for key, values in gaps.items()
    }
    return margins, gaps


def _factor_explained(values, labels):
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    total = float(np.var(values))
    if total <= 1e-18:
        return 0.0
    _, inverse = np.unique(labels, return_inverse=True)
    count = np.bincount(inverse)
    means = np.bincount(inverse, weights=values) / np.maximum(count, 1)
    fitted = means[inverse]
    return float(np.var(fitted) / total)


def _context_dependence_rows(
    records, split_label, matrices, embeddings, margins
):
    output = []
    grouped = defaultdict(list)
    for record in records:
        meta = record["meta"]
        grouped[(meta["task_id"], meta["phase"])].append(record)
    all_distance = []
    factor = defaultdict(list)
    for (task, phase), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: int(row["meta"]["episode_id"]))
        keys = [
            (row["meta"]["task_id"], int(row["meta"]["episode_id"]), row["meta"]["phase"])
            for row in ordered
        ]
        stack = np.stack([matrices[key] for key in keys])
        target_residuals = [
            np.asarray(row["support"]["residual_action"][1:], dtype=np.float64)
            for row in ordered
        ]
        if not all(np.array_equal(target_residuals[0], value) for value in target_residuals[1:]):
            raise RuntimeError("matched support residuals changed within %s/%s" % (task, phase))
        margin = margins[(task, phase)]
        rng = np.random.RandomState(
            int.from_bytes(
                __import__("hashlib")
                .sha256(("%d|%s|%s|%s" % (REVERSAL_PAIR_SEED, split_label, task, phase)).encode())
                .digest()[:4],
                "little",
            )
        )
        eligible = 0
        reversals = 0
        for _ in range(4096):
            state_left, state_right = rng.choice(len(ordered), size=2, replace=False)
            target = int(rng.randint(stack.shape[1]))
            candidate_left, candidate_right = rng.choice(
                stack.shape[2], size=2, replace=False
            )
            left_gap = float(
                stack[state_left, target, candidate_right]
                - stack[state_left, target, candidate_left]
            )
            right_gap = float(
                stack[state_right, target, candidate_right]
                - stack[state_right, target, candidate_left]
            )
            if abs(left_gap) > margin and abs(right_gap) > margin:
                eligible += 1
                reversals += int(left_gap * right_gap < 0.0)

        jaccard = []
        churn = []
        for left, right in itertools.combinations(range(len(ordered)), 2):
            for target in range(stack.shape[1]):
                top_left = set(np.argsort(stack[left, target], kind="mergesort")[:8])
                top_right = set(np.argsort(stack[right, target], kind="mergesort")[:8])
                jaccard.append(len(top_left & top_right) / len(top_left | top_right))
                churn.append(
                    int(
                        argmin_stable(stack[left, target])
                        != argmin_stable(stack[right, target])
                    )
                )

        target_embedding = np.stack([embeddings[key][0] for key in keys])
        candidate_embedding = np.stack([embeddings[key][1] for key in keys])
        state_best_error = []
        global_error = []
        global_regret = []
        for state in range(len(ordered)):
            others = [index for index in range(len(ordered)) if index != state]
            mean_target = np.mean(target_embedding[others], axis=0)
            mean_candidate = np.mean(candidate_embedding[others], axis=0)
            averaged_distance = np.sum(
                (mean_target[:, None, :] - mean_candidate[None, :, :]) ** 2,
                axis=2,
            )
            selected = np.asarray([argmin_stable(row) for row in averaged_distance])
            true = stack[state]
            state_best_error.extend(np.min(true, axis=1).tolist())
            chosen = true[np.arange(len(selected)), selected]
            global_error.extend(chosen.tolist())
            global_regret.extend((chosen - np.min(true, axis=1)).tolist())

        metrics = {
            "context_reversal_rate": reversals / max(eligible, 1),
            "eligible_reversal_comparison_fraction": eligible / 4096.0,
            "true_top8_jaccard_across_states": float(np.mean(jaccard)),
            "best_candidate_churn": float(np.mean(churn)),
            "state_conditioned_oracle_error": float(np.mean(state_best_error)),
            "leave_one_state_out_global_averaged_effect_oracle_error": float(
                np.mean(global_error)
            ),
            "global_averaged_effect_oracle_regret": float(np.mean(global_regret)),
        }
        for metric, value in metrics.items():
            output.append(
                {
                    "split": split_label,
                    "level": "task_phase",
                    "task_id": task,
                    "phase": phase,
                    "metric": metric,
                    "value": value,
                    "n_states": len(ordered),
                    "n_targets_per_state": stack.shape[1],
                    "train_robust_margin": margin,
                    "eligible_reversal_comparisons": eligible,
                    "valid_reversals": reversals,
                }
            )

        flat = stack.reshape(-1)
        all_distance.append(flat)
        count_per_state = stack.shape[1] * stack.shape[2]
        factor["task"].extend([task] * len(flat))
        factor["phase"].extend([phase] * len(flat))
        state_labels = np.repeat(np.arange(len(ordered)), count_per_state)
        factor["state"].extend(
            ["%s|%s|%d" % (task, phase, value) for value in state_labels]
        )
        pair_labels = np.tile(
            np.arange(stack.shape[1] * stack.shape[2]), len(ordered)
        )
        factor["action_pair"].extend(
            ["%s|%d" % (split_label, value) for value in pair_labels]
        )
    values = np.concatenate(all_distance)
    for name in ("state", "task", "phase", "action_pair"):
        output.append(
            {
                "split": split_label,
                "level": "pooled_variance",
                "task_id": "ALL",
                "phase": "ALL",
                "metric": "marginal_variance_explained_by_" + name,
                "value": _factor_explained(values, factor[name]),
                "n_states": len(records),
                "n_targets_per_state": int(len(values) / len(records) / ACTION_BANK_SIZE),
                "train_robust_margin": float("nan"),
                "eligible_reversal_comparisons": 0,
                "valid_reversals": 0,
            }
        )
    return output


def _intervention_rows(
    records,
    contexts,
    models,
    consequence_scale,
    device,
    true_matrices,
):
    branch_rows = []
    for state_index, record in enumerate(records):
        meta = record["meta"]
        key = (meta["task_id"], int(meta["episode_id"]), meta["phase"])
        true = true_matrices[key]
        targets = np.asarray(record["support"]["residual_action"][1:], dtype=np.float32)
        bank = np.asarray(record["candidate"]["residual_action"][1:], dtype=np.float32)
        target_family = np.asarray(
            record["support"]["direction_family_id"][1:], dtype=np.int64
        )
        predicted = {}
        decoded = {}
        for intervention, values in contexts.items():
            context = values[state_index]
            target_embedding = _ensemble_embedding(
                models["C3_NC_BIENCODER"], context, targets, device
            )
            bank_embedding = _ensemble_embedding(
                models["C3_NC_BIENCODER"], context, bank, device
            )
            matrix = np.sum(
                (target_embedding[:, None, :] - bank_embedding[None, :, :]) ** 2,
                axis=2,
            )
            predicted[intervention] = matrix
            decoded[intervention] = np.asarray([argmin_stable(row) for row in matrix])
        correct = predicted["correct_context"]
        correct_decoded = decoded["correct_context"]
        from scipy.stats import rankdata

        correct_rank = rankdata(correct, axis=1, method="average")
        correct_rank -= np.mean(correct_rank, axis=1, keepdims=True)
        correct_norm = np.sqrt(np.sum(correct_rank ** 2, axis=1))
        true_order = np.argsort(true, axis=1, kind="stable")
        relevance = np.exp(-true / 0.15)
        discount = 1.0 / np.log2(np.arange(2, 18))
        for intervention in contexts:
            matrix = predicted[intervention]
            chosen = decoded[intervention]
            matrix_rank = rankdata(matrix, axis=1, method="average")
            matrix_rank -= np.mean(matrix_rank, axis=1, keepdims=True)
            matrix_norm = np.sqrt(np.sum(matrix_rank ** 2, axis=1))
            correlation = np.sum(correct_rank * matrix_rank, axis=1) / np.maximum(
                correct_norm * matrix_norm, 1e-12
            )
            predicted_order = np.argsort(matrix, axis=1, kind="stable")
            realized = realized_rows(
                record, chosen, intervention, consequence_scale
            )
            for target_id in range(len(targets)):
                predicted_top = predicted_order[target_id, :16]
                true_top = true_order[target_id, :16]
                dcg = float(np.sum(relevance[target_id, predicted_top] * discount))
                ideal = float(np.sum(relevance[target_id, true_top] * discount))
                recall8 = len(
                    set(predicted_order[target_id, :8].tolist())
                    & set(true_order[target_id, :8].tolist())
                ) / 8.0
                row = {
                    "split": SPLIT_LABELS[meta["split"]],
                    "task_id": meta["task_id"],
                    "episode_id": int(meta["episode_id"]),
                    "phase": meta["phase"],
                    "direction_family_id": int(target_family[target_id]),
                    "target_id": target_id,
                    "intervention": intervention,
                    "mean_abs_distance_change": float(
                        np.mean(np.abs(matrix[target_id] - correct[target_id]))
                    ),
                    "relative_mean_abs_distance_change": float(
                        np.mean(np.abs(matrix[target_id] - correct[target_id]))
                        / max(float(np.mean(np.abs(correct[target_id]))), 1e-12)
                    ),
                    "distance_spearman_with_correct": float(correlation[target_id]),
                    "selected_code_changed": int(
                        chosen[target_id] != correct_decoded[target_id]
                    ),
                    "ndcg_at_16": dcg / max(ideal, 1e-12),
                    "recall_at_8": recall8,
                    "oracle_regret": float(
                        true[target_id, chosen[target_id]]
                        - np.min(true[target_id])
                    ),
                    "balanced_task_effect_error": realized[target_id][
                        "balanced_task_effect_error"
                    ],
                    "action_reconstruction_rmse": realized[target_id][
                        "action_reconstruction_rmse"
                    ],
                    "contact_mode_preserved": realized[target_id][
                        "contact_mode_preserved"
                    ],
                    "task_progress_abs_error": realized[target_id][
                        "task_progress_abs_error"
                    ],
                }
                branch_rows.append(row)
    return branch_rows


def _intervention_summary(rows):
    output = []
    metric_names = (
        "mean_abs_distance_change",
        "relative_mean_abs_distance_change",
        "distance_spearman_with_correct",
        "selected_code_changed",
        "ndcg_at_16",
        "recall_at_8",
        "oracle_regret",
        "balanced_task_effect_error",
        "action_reconstruction_rmse",
        "contact_mode_preserved",
        "task_progress_abs_error",
    )
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        partitions = [("pooled", "ALL", "ALL", split_rows)]
        partitions.extend(
            ("task", task, "ALL", [row for row in split_rows if row["task_id"] == task])
            for task in sorted({row["task_id"] for row in split_rows})
        )
        partitions.extend(
            ("phase", "ALL", phase, [row for row in split_rows if row["phase"] == phase])
            for phase in sorted({row["phase"] for row in split_rows})
        )
        for level, task, phase, partition in partitions:
            for intervention in sorted({row["intervention"] for row in partition}):
                selected = [
                    row for row in partition if row["intervention"] == intervention
                ]
                result = {
                    "split": split,
                    "level": level,
                    "task_id": task,
                    "phase": phase,
                    "intervention": intervention,
                    "n_targets": len(selected),
                }
                for metric in metric_names:
                    result[metric] = float(np.mean([row[metric] for row in selected]))
                output.append(result)
    return output


def run_historical_audit(project_root, output_root=None, device_name="cpu"):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    historical_output = os.path.join(
        HISTORICAL_REPOSITORY_ROOT, HISTORICAL_STAGE3_RELATIVE
    )
    device = _device(device_name)
    registry, scalers, models = load_trained_models(historical_output, device)
    consequence_scale = np.asarray(scalers["consequence_scale"], dtype=np.float64)
    center = np.asarray(scalers["context_center"], dtype=np.float64)
    scale = np.asarray(scalers["context_scale"], dtype=np.float64)
    codebooks = _load_baseline_codebooks(historical_output)

    train_records = load_records(
        HISTORICAL_REPOSITORY_ROOT,
        historical_output,
        ("train",),
        HISTORICAL_SCRATCH_ROOT,
    )
    margins, margin_samples = _train_margins(train_records, consequence_scale)
    audit_rows = []
    for (task, phase), margin in sorted(margins.items()):
        audit_rows.append(
            {
                "split": "TRAIN_EPISODES_16_31",
                "level": "task_phase",
                "task_id": task,
                "phase": phase,
                "metric": "train_robust_reversal_margin_q25",
                "value": margin,
                "n_states": 16,
                "n_targets_per_state": 96,
                "train_robust_margin": margin,
                "eligible_reversal_comparisons": len(margin_samples[(task, phase)]),
                "valid_reversals": 0,
            }
        )

    decomposition = []
    intervention_branches = []
    audit_metadata = {}
    for split in ("development", "confirmation"):
        records = load_records(
            HISTORICAL_REPOSITORY_ROOT,
            historical_output,
            (split,),
            HISTORICAL_SCRATCH_ROOT,
        )
        context_values = _context_interventions(records, center, scale)
        branch_rows, matrices, embeddings = _decomposition_branch_rows(
            records,
            context_values["correct_context"],
            models,
            codebooks,
            consequence_scale,
            device,
        )
        split_label = SPLIT_LABELS[split]
        decomposition.extend(_decomposition_summary(branch_rows, split_label))
        audit_rows.extend(
            _context_dependence_rows(
                records, split_label, matrices, embeddings, margins
            )
        )
        intervention_branches.extend(
            _intervention_rows(
                records,
                context_values,
                models,
                consequence_scale,
                device,
                matrices,
            )
        )
        audit_metadata[split_label] = {
            "states": len(records),
            "targets_per_state": int(
                len(records[0]["support"]["residual_action"]) - 1
            ),
            "candidate_count": ACTION_BANK_SIZE,
        }

    decomposition_path = os.path.join(output_root, "C3_FAILURE_DECOMPOSITION.csv")
    dependence_path = os.path.join(output_root, "CONTEXT_DEPENDENCE_AUDIT.csv")
    intervention_path = os.path.join(output_root, "C3_CONTEXT_INTERVENTIONS.csv")
    write_csv(decomposition_path, decomposition)
    write_csv(dependence_path, audit_rows)
    write_csv(intervention_path, _intervention_summary(intervention_branches))
    metadata_path = os.path.join(output_root, "historical_audit_metadata.json")
    atomic_json(
        metadata_path,
        {
            "stage3_registry_sha256": sha256_file(
                os.path.join(historical_output, "trained_model_registry.json")
            ),
            "frozen_c3_members": registry["models"]["C3_NC_BIENCODER"]["members"],
            "frozen_c4_members": registry["models"]["C4_NC_PAIR_RANKER"]["members"],
            "device": str(device),
            "splits": audit_metadata,
            "train_margin_quantile": CR_PAIR_MARGIN_QUANTILE,
            "no_simulator_execution": True,
            "no_retraining": True,
            "outputs": {
                "C3_FAILURE_DECOMPOSITION.csv": sha256_file(decomposition_path),
                "CONTEXT_DEPENDENCE_AUDIT.csv": sha256_file(dependence_path),
                "C3_CONTEXT_INTERVENTIONS.csv": sha256_file(intervention_path),
            },
        },
    )
    return {
        "decomposition_rows": len(decomposition),
        "context_dependence_rows": len(audit_rows),
        "context_intervention_rows": len(_intervention_summary(intervention_branches)),
        "metadata": metadata_path,
    }


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run_historical_audit(args.project_root, args.output_root, args.device),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
