"""Calibration-only C3 objective re-selection and frozen evaluation."""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from .stage3_analysis import _ensemble_embedding, _load_torch_state
from .stage3_config import SCRATCH_ROOT as HISTORICAL_SCRATCH_ROOT
from .stage3_data import build_pair_dataset, load_records, transformed_contexts, true_distance_matrix
from .stage3_metrics import argmin_stable, realized_rows, stable_fps, write_csv
from .stage3_models import create_biencoder, save_model, train_pair_model
from .stage4_config import (
    C3_RESELECT_SEEDS,
    HISTORICAL_REPOSITORY_ROOT,
    HISTORICAL_STAGE3_RELATIVE,
    OUTPUT_RELATIVE,
    PRIMARY_K,
    RANKING_OBJECTIVE_CANDIDATES,
)
from .storage import atomic_json, sha256_file


def _device(name):
    import torch

    device = torch.device(name or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one local GPU for Stage 4")
    return device


def deterministic_kmedoids(values, k=PRIMARY_K, max_iterations=50):
    """Deterministic predicted-space k-medoids initialized by stable FPS."""
    values = np.asarray(values, dtype=np.float64)
    medoids = stable_fps(values, int(k))
    full_distance = np.sum(
        (values[:, None, :] - values[None, :, :]) ** 2, axis=2
    )
    for _ in range(int(max_iterations)):
        distance = full_distance[:, medoids]
        assignment = np.argmin(distance, axis=1)
        updated = []
        for cluster_id, old_medoid in enumerate(medoids):
            members = np.flatnonzero(assignment == cluster_id)
            if not len(members):
                updated.append(int(old_medoid))
                continue
            cost = np.sum(full_distance[np.ix_(members, members)], axis=1)
            minimum = float(np.min(cost))
            tied = members[np.isclose(cost, minimum, rtol=0.0, atol=1e-15)]
            updated.append(int(np.min(tied)))
        updated = np.asarray(updated, dtype=np.int64)
        if len(np.unique(updated)) != int(k):
            # An empty/tied duplicate is filled by the lowest-ID point farthest
            # from the unique current set. This path is deterministic and
            # still uses predicted embeddings only.
            unique = list(dict.fromkeys(updated.tolist()))
            while len(unique) < int(k):
                remaining = np.asarray(
                    [index for index in range(len(values)) if index not in unique],
                    dtype=np.int64,
                )
                minimum = np.min(full_distance[np.ix_(remaining, unique)], axis=1)
                maximum = float(np.max(minimum))
                tied = remaining[np.isclose(minimum, maximum, rtol=0.0, atol=1e-15)]
                unique.append(int(np.min(tied)))
            updated = np.asarray(unique, dtype=np.int64)
        if np.array_equal(updated, medoids):
            break
        medoids = updated
    return medoids


def _selection_metrics(model, records, contexts, consequence_scale, device):
    regrets = []
    ndcg = []
    for state_index, record in enumerate(records):
        context = contexts[state_index]
        target = np.asarray(record["support"]["residual_action"][1:], dtype=np.float32)
        bank = np.asarray(record["candidate"]["residual_action"][1:], dtype=np.float32)
        target_embedding = _ensemble_embedding([model], context, target, device)
        bank_embedding = _ensemble_embedding([model], context, bank, device)
        predicted = np.sum(
            (target_embedding[:, None, :] - bank_embedding[None, :, :]) ** 2,
            axis=2,
        )
        true = true_distance_matrix(record, consequence_scale)
        selected = np.argmin(predicted, axis=1)
        regrets.extend(
            (true[np.arange(len(selected)), selected] - np.min(true, axis=1)).tolist()
        )
        true_order = np.argsort(true, axis=1, kind="stable")[:, :16]
        predicted_order = np.argsort(predicted, axis=1, kind="stable")[:, :16]
        relevance = np.exp(-true / 0.15)
        discount = 1.0 / np.log2(np.arange(2, 18))
        for target_id in range(len(target)):
            ideal = float(
                np.sum(relevance[target_id, true_order[target_id]] * discount)
            )
            actual = float(
                np.sum(relevance[target_id, predicted_order[target_id]] * discount)
            )
            ndcg.append(actual / max(ideal, 1e-12))
    return {
        "c3_full_mean_oracle_regret": float(np.mean(regrets)),
        "c3_full_mean_ndcg_at_16": float(np.mean(ndcg)),
        "targets": len(regrets),
    }


def _load_stage3_inputs():
    historical_output = os.path.join(
        HISTORICAL_REPOSITORY_ROOT, HISTORICAL_STAGE3_RELATIVE
    )
    train_records = load_records(
        HISTORICAL_REPOSITORY_ROOT,
        historical_output,
        ("train",),
        HISTORICAL_SCRATCH_ROOT,
    )
    calibration_records = load_records(
        HISTORICAL_REPOSITORY_ROOT,
        historical_output,
        ("calibration",),
        HISTORICAL_SCRATCH_ROOT,
    )
    with np.load(
        os.path.join(historical_output, "model_scalers.npz"), allow_pickle=False
    ) as data:
        consequence_scale = np.asarray(data["consequence_scale"], dtype=np.float64)
        context_center = np.asarray(data["context_center"], dtype=np.float64)
        context_scale = np.asarray(data["context_scale"], dtype=np.float64)
    return (
        historical_output,
        train_records,
        calibration_records,
        consequence_scale,
        context_center,
        context_scale,
    )


def _load_checkpoint(path, context_dim, device):
    return _load_torch_state(path, create_biencoder(context_dim), device)


def train_and_select(project_root, output_root=None, device_name=None):
    """Train four screening tuples, select on calibration, then make 3 members."""
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    model_root = os.path.join(output_root, "models", "c3_reselect")
    os.makedirs(model_root, exist_ok=True)
    device = _device(device_name)
    (
        historical_output,
        train_records,
        calibration_records,
        consequence_scale,
        context_center,
        context_scale,
    ) = _load_stage3_inputs()
    train_dataset = build_pair_dataset(
        train_records, consequence_scale, context_center, context_scale
    )
    calibration_dataset = build_pair_dataset(
        calibration_records, consequence_scale, context_center, context_scale
    )
    calibration_contexts = transformed_contexts(
        calibration_records, context_center, context_scale
    )

    trace = []
    screening_models = []
    for index, objective in enumerate(RANKING_OBJECTIVE_CANDIDATES):
        path = os.path.join(model_root, "screen_objective_%d.pt" % index)
        evidence_path = os.path.join(model_root, "screen_objective_%d.json" % index)
        if os.path.isfile(path) and os.path.isfile(evidence_path):
            evidence = json.load(open(evidence_path, "r", encoding="utf-8"))
            if evidence["checkpoint_sha256"] != sha256_file(path):
                raise RuntimeError("screening checkpoint hash mismatch")
            model = _load_checkpoint(path, len(context_center), device)
            metadata = evidence["training_metadata"]
            metrics = evidence["calibration_metrics"]
        else:
            started = time.perf_counter()
            model, metadata = train_pair_model(
                train_dataset,
                calibration_dataset,
                "C3_NC_BIENCODER",
                objective,
                C3_RESELECT_SEEDS[0],
                device,
            )
            metrics = _selection_metrics(
                model,
                calibration_records,
                calibration_contexts,
                consequence_scale,
                device,
            )
            metadata["wall_seconds"] = float(time.perf_counter() - started)
            save_model(path, model, metadata)
            atomic_json(
                evidence_path,
                {
                    "objective_index": index,
                    "objective": dict(objective),
                    "screening_seed": C3_RESELECT_SEEDS[0],
                    "training_metadata": metadata,
                    "calibration_metrics": metrics,
                    "checkpoint_sha256": sha256_file(path),
                    "development_or_historical_used": False,
                },
            )
        screening_models.append(model)
        trace.append(
            {
                "objective_index": index,
                "objective": dict(objective),
                "checkpoint": os.path.relpath(path, output_root),
                "checkpoint_sha256": sha256_file(path),
                "training_metadata": metadata,
                "calibration_metrics": metrics,
            }
        )

    selected_index = min(
        range(len(trace)),
        key=lambda index: (
            trace[index]["calibration_metrics"]["c3_full_mean_oracle_regret"],
            -trace[index]["calibration_metrics"]["c3_full_mean_ndcg_at_16"],
            index,
        ),
    )
    selected_objective = dict(RANKING_OBJECTIVE_CANDIDATES[selected_index])
    members = []
    member_metadata = []
    for member_index, seed in enumerate(C3_RESELECT_SEEDS):
        path = os.path.join(model_root, "C3_RESELECT_member_%d.pt" % member_index)
        evidence_path = path + ".json"
        if os.path.isfile(path) and os.path.isfile(evidence_path):
            evidence = json.load(open(evidence_path, "r", encoding="utf-8"))
            if evidence["checkpoint_sha256"] != sha256_file(path):
                raise RuntimeError("ensemble checkpoint hash mismatch")
            model = _load_checkpoint(path, len(context_center), device)
            metadata = evidence["training_metadata"]
        elif member_index == 0:
            model = screening_models[selected_index]
            metadata = dict(trace[selected_index]["training_metadata"])
            save_model(path, model, metadata)
        else:
            started = time.perf_counter()
            model, metadata = train_pair_model(
                train_dataset,
                calibration_dataset,
                "C3_NC_BIENCODER",
                selected_objective,
                seed,
                device,
            )
            metadata["wall_seconds"] = float(time.perf_counter() - started)
            save_model(path, model, metadata)
        evidence = {
            "member_index": member_index,
            "seed": seed,
            "selected_objective_index": selected_index,
            "selected_objective": selected_objective,
            "training_metadata": metadata,
            "checkpoint_sha256": sha256_file(path),
        }
        atomic_json(evidence_path, evidence)
        members.append(
            {
                "member_index": member_index,
                "seed": seed,
                "checkpoint": os.path.relpath(path, output_root),
                "checkpoint_sha256": evidence["checkpoint_sha256"],
            }
        )
        member_metadata.append(metadata)

    result = {
        "selection_split": "calibration episodes 32-35 only",
        "architecture": "exact Stage 3 C3_NC_BIENCODER",
        "screening_seed": C3_RESELECT_SEEDS[0],
        "objective_trace": trace,
        "selection_order": [
            "lowest C3_FULL calibration oracle regret",
            "highest C3_FULL calibration NDCG@16",
            "lowest frozen objective index",
        ],
        "selected_objective_index": selected_index,
        "selected_objective": selected_objective,
        "ensemble_members": members,
        "train_states": len(train_records),
        "calibration_states": len(calibration_records),
        "development_or_historical_exploratory_used_for_selection": False,
        "historical_scaler_sha256": sha256_file(
            os.path.join(historical_output, "model_scalers.npz")
        ),
        "device": str(device),
    }
    selection_path = os.path.join(output_root, "C3_RESELECTION.json")
    atomic_json(selection_path, result)
    atomic_json(
        os.path.join(output_root, "MODEL_SELECTION.json"),
        {
            "c3_reselection": result,
            "cr_c3_selection": {"status": "PENDING_EXPANDED_TRAINING"},
            "trust_region_selection": {"status": "PENDING_CALIBRATION"},
            "bounded_correction_selection": {"status": "PENDING_OPTIONAL_DIAGNOSTIC"},
            "confirmation_method_frozen": False,
        },
    )
    return {
        "selection": selection_path,
        "selected_objective_index": selected_index,
        "selected_objective": selected_objective["name"],
        "members": len(members),
    }


def _ranking_batch(true, predicted, selected):
    from scipy.stats import kendalltau, rankdata

    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    selected = np.asarray(selected, dtype=np.int64)
    true_rank = rankdata(true, axis=1, method="average")
    predicted_rank = rankdata(predicted, axis=1, method="average")
    true_centered = true_rank - np.mean(true_rank, axis=1, keepdims=True)
    pred_centered = predicted_rank - np.mean(predicted_rank, axis=1, keepdims=True)
    spearman = np.sum(true_centered * pred_centered, axis=1) / np.maximum(
        np.sqrt(np.sum(true_centered ** 2, axis=1) * np.sum(pred_centered ** 2, axis=1)),
        1e-12,
    )
    true_order = np.argsort(true, axis=1, kind="stable")
    predicted_order = np.argsort(predicted, axis=1, kind="stable")
    relevance = np.exp(-true / 0.15)
    discount = 1.0 / np.log2(np.arange(2, 18))
    rows = []
    for index in range(len(true)):
        dcg = float(np.sum(relevance[index, predicted_order[index, :16]] * discount))
        ideal = float(np.sum(relevance[index, true_order[index, :16]] * discount))
        kendall = float(kendalltau(true[index], predicted[index], variant="b").statistic)
        if not np.isfinite(kendall):
            kendall = 0.0
        rows.append(
            {
                "candidate_distance_spearman": float(spearman[index]),
                "kendall_tau": kendall,
                "ndcg_at_16": dcg / max(ideal, 1e-12),
                "oracle_neighbor_recall_at_1": int(
                    predicted_order[index, 0] == true_order[index, 0]
                ),
                "oracle_neighbor_recall_at_8": len(
                    set(predicted_order[index, :8].tolist())
                    & set(true_order[index, :8].tolist())
                )
                / 8.0,
                "oracle_regret": float(
                    true[index, selected[index]] - np.min(true[index])
                ),
            }
        )
    return rows


def _models_from_selection(output_root, selection, device, context_dim):
    return [
        _load_checkpoint(
            os.path.join(output_root, member["checkpoint"]), context_dim, device
        )
        for member in selection["ensemble_members"]
    ]


def evaluate_reselected(project_root, output_root=None, device_name="cpu"):
    import pandas as pd

    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    selection = json.load(
        open(os.path.join(output_root, "C3_RESELECTION.json"), "r", encoding="utf-8")
    )
    device = _device(device_name)
    (
        _,
        _,
        _,
        consequence_scale,
        context_center,
        context_scale,
    ) = _load_stage3_inputs()
    models = _models_from_selection(
        output_root, selection, device, len(context_center)
    )
    retrieval_rows = []
    realized = []
    for split, evidence_label in (
        ("development", "DEVELOPMENT_EPISODES_36_39"),
        ("confirmation", "HISTORICAL_EXPLORATORY_EPISODES_40_49"),
    ):
        historical_output = os.path.join(
            HISTORICAL_REPOSITORY_ROOT, HISTORICAL_STAGE3_RELATIVE
        )
        records = load_records(
            HISTORICAL_REPOSITORY_ROOT,
            historical_output,
            (split,),
            HISTORICAL_SCRATCH_ROOT,
        )
        contexts = transformed_contexts(records, context_center, context_scale)
        for state_index, record in enumerate(records):
            context = contexts[state_index]
            targets = np.asarray(
                record["support"]["residual_action"][1:], dtype=np.float32
            )
            bank = np.asarray(
                record["candidate"]["residual_action"][1:], dtype=np.float32
            )
            started = time.perf_counter()
            target_embedding = _ensemble_embedding(models, context, targets, device)
            bank_embedding = _ensemble_embedding(models, context, bank, device)
            predicted = np.sum(
                (target_embedding[:, None, :] - bank_embedding[None, :, :]) ** 2,
                axis=2,
            )
            fps = stable_fps(bank_embedding, PRIMARY_K)
            medoids = deterministic_kmedoids(bank_embedding, PRIMARY_K)
            decoded = {
                "C3_RESELECT_FULL": np.asarray(
                    [argmin_stable(row) for row in predicted]
                ),
                "C3_RESELECT_FPS64": np.asarray(
                    [argmin_stable(row[fps], ids=fps) for row in predicted]
                ),
                "C3_RESELECT_KMEDOIDS64": np.asarray(
                    [argmin_stable(row[medoids], ids=medoids) for row in predicted]
                ),
            }
            latency = 1000.0 * (time.perf_counter() - started) / len(targets)
            true = true_distance_matrix(record, consequence_scale)
            family_ids = np.asarray(
                record["support"]["direction_family_id"][1:], dtype=np.int64
            )
            for method, chosen in decoded.items():
                method_realized = realized_rows(
                    record, chosen, method, consequence_scale, latency
                )
                for target_id, row in enumerate(method_realized):
                    row["evidence_split"] = evidence_label
                    row["direction_family_id"] = int(family_ids[target_id])
                    realized.append(row)
                ranking = _ranking_batch(true, predicted, chosen)
                for target_id, row in enumerate(ranking):
                    row.update(
                        {
                            "evidence_split": evidence_label,
                            "task_id": record["meta"]["task_id"],
                            "episode_id": int(record["meta"]["episode_id"]),
                            "phase": record["meta"]["phase"],
                            "direction_family_id": int(family_ids[target_id]),
                            "target_id": target_id,
                            "method": method,
                            "selected_bank_index": int(chosen[target_id]),
                            "inference_latency_ms": latency,
                        }
                    )
                    retrieval_rows.append(row)
    retrieval_path = os.path.join(output_root, "c3_reselect_retrieval.parquet")
    realized_path = os.path.join(output_root, "c3_reselect_realized.parquet")
    pd.DataFrame(retrieval_rows).to_parquet(retrieval_path, index=False)
    pd.DataFrame(realized).to_parquet(realized_path, index=False)
    summaries = []
    for evidence_split in sorted({row["evidence_split"] for row in realized}):
        for method in sorted({row["method"] for row in realized}):
            selected_realized = [
                row
                for row in realized
                if row["evidence_split"] == evidence_split and row["method"] == method
            ]
            selected_retrieval = [
                row
                for row in retrieval_rows
                if row["evidence_split"] == evidence_split and row["method"] == method
            ]
            summaries.append(
                {
                    "evidence_split": evidence_split,
                    "method": method,
                    "level": "pooled",
                    "task_id": "ALL",
                    "n": len(selected_realized),
                    "balanced_task_effect_error": float(
                        np.mean(
                            [row["balanced_task_effect_error"] for row in selected_realized]
                        )
                    ),
                    "action_reconstruction_rmse": float(
                        np.mean(
                            [row["action_reconstruction_rmse"] for row in selected_realized]
                        )
                    ),
                    "contact_mode_preserved": float(
                        np.mean(
                            [row["contact_mode_preserved"] for row in selected_realized]
                        )
                    ),
                    "oracle_regret": float(
                        np.mean([row["oracle_regret"] for row in selected_retrieval])
                    ),
                    "ndcg_at_16": float(
                        np.mean([row["ndcg_at_16"] for row in selected_retrieval])
                    ),
                    "recall_at_8": float(
                        np.mean(
                            [
                                row["oracle_neighbor_recall_at_8"]
                                for row in selected_retrieval
                            ]
                        )
                    ),
                }
            )
            for task in sorted({row["task_id"] for row in selected_realized}):
                task_realized = [row for row in selected_realized if row["task_id"] == task]
                task_retrieval = [row for row in selected_retrieval if row["task_id"] == task]
                summaries.append(
                    {
                        "evidence_split": evidence_split,
                        "method": method,
                        "level": "task",
                        "task_id": task,
                        "n": len(task_realized),
                        "balanced_task_effect_error": float(
                            np.mean([row["balanced_task_effect_error"] for row in task_realized])
                        ),
                        "action_reconstruction_rmse": float(
                            np.mean([row["action_reconstruction_rmse"] for row in task_realized])
                        ),
                        "contact_mode_preserved": float(
                            np.mean([row["contact_mode_preserved"] for row in task_realized])
                        ),
                        "oracle_regret": float(
                            np.mean([row["oracle_regret"] for row in task_retrieval])
                        ),
                        "ndcg_at_16": float(
                            np.mean([row["ndcg_at_16"] for row in task_retrieval])
                        ),
                        "recall_at_8": float(
                            np.mean(
                                [row["oracle_neighbor_recall_at_8"] for row in task_retrieval]
                            )
                        ),
                    }
                )
    summary_path = os.path.join(output_root, "C3_RESELECT_EVALUATION.csv")
    write_csv(summary_path, summaries)
    atomic_json(
        os.path.join(output_root, "c3_reselect_evaluation_metadata.json"),
        {
            "selection_sha256": sha256_file(
                os.path.join(output_root, "C3_RESELECTION.json")
            ),
            "retrieval_parquet_sha256": sha256_file(retrieval_path),
            "realized_parquet_sha256": sha256_file(realized_path),
            "summary_sha256": sha256_file(summary_path),
            "development_or_historical_used_for_selection": False,
            "device": str(device),
        },
    )
    return {
        "retrieval_rows": len(retrieval_rows),
        "realized_rows": len(realized),
        "summary": summary_path,
    }


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train-select", "evaluate"))
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    if args.command == "train-select":
        result = train_and_select(args.project_root, args.output_root, args.device)
    else:
        result = evaluate_reselected(
            args.project_root, args.output_root, args.device or "cpu"
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
