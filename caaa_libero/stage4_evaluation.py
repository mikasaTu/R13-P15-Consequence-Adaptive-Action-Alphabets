"""Calibration, development, and historical exploratory Stage 4 evaluation."""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict

import numpy as np

from .math_utils import covariance_whitener
from .stage3_analysis import (
    _action_assign,
    _ensemble_embedding,
    _load_baseline_codebooks,
    load_trained_models,
)
from .stage3_data import (
    CONTEXT_SLICES,
    STATE_CONTROL_SLICES,
    transformed_contexts,
)
from .stage3_metrics import (
    argmin_stable,
    full_oracle_decoded,
    realized_rows,
    stable_fps,
    summarize_realized,
    true_oracle_decoded,
    write_csv,
)
from .stage4_config import (
    ACTION_BANK_SIZE,
    BOUNDED_CORRECTION_GAMMA,
    CONTACT_SENSITIVE_TASKS,
    GATES,
    HISTORICAL_REPOSITORY_ROOT,
    HISTORICAL_STAGE3_RELATIVE,
    OUTPUT_RELATIVE,
    PHASES,
    PRIMARY_K,
    SCRATCH_ROOT,
    TASK_IDS,
    TRUST_REGION_L,
)
from .stage4_data import _cache_path, historical_records, load_cache, reversal_pairs
from .stage4_historical import _pair_matrix
from .stage4_models import (
    PROPOSED_CONTROL,
    _device,
    ensemble_action_embedding,
    evaluation_cache_for_control,
    load_checkpoint,
    reversal_accuracy,
    score_cache,
)
from .stage4_reselect import deterministic_kmedoids
from .storage import atomic_json, sha256_file


HISTORICAL_OUTPUT = os.path.join(
    HISTORICAL_REPOSITORY_ROOT, HISTORICAL_STAGE3_RELATIVE
)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _aligned(records, cache):
    record_keys = [str(record["meta"]["key"]) for record in records]
    cache_keys = cache["key"].astype(str).tolist()
    if record_keys != cache_keys:
        raise RuntimeError("record/cache order mismatch")


def _load_models_from_entries(output_root, entries, device):
    models = []
    for entry in entries:
        path = os.path.join(output_root, entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError("checkpoint hash mismatch: " + path)
        model, _ = load_checkpoint(path, device)
        models.append(model)
    return models


def selected_cr_models(output_root, device):
    selection = _load_json(os.path.join(output_root, "MODEL_SELECTION.json"))
    cr = selection["cr_c3_selection"]
    definition = cr["family_trace"][cr["selected_family_index"]]
    return cr["selected_family"], _load_models_from_entries(
        output_root, definition["checkpoints"], device
    )


def control_models(output_root, control, device):
    selection = _load_json(os.path.join(output_root, "MODEL_SELECTION.json"))
    definition = next(
        value
        for value in selection["cr_c3_controls"]["controls"]
        if value["control"] == control
    )
    return _load_models_from_entries(output_root, definition["checkpoints"], device)


def _load_reselected_models(output_root, device):
    from .stage3_models import create_biencoder
    from .stage3_analysis import _load_torch_state

    selection = _load_json(os.path.join(output_root, "C3_RESELECTION.json"))
    models = []
    for entry in selection["ensemble_members"]:
        path = os.path.join(output_root, entry["checkpoint"])
        if sha256_file(path) != entry["checkpoint_sha256"]:
            raise RuntimeError("C3 reselect checkpoint hash mismatch")
        models.append(_load_torch_state(path, create_biencoder(321), device))
    return models


def _historical_methods(records, output_root, device):
    """Run every historical and C3-reselection decoder on one split."""
    registry, scalers, frozen = load_trained_models(HISTORICAL_OUTPUT, device)
    del registry
    codebooks = _load_baseline_codebooks(HISTORICAL_OUTPUT)
    contexts = transformed_contexts(
        records, scalers["context_center"], scalers["context_scale"]
    )
    reselected = _load_reselected_models(output_root, device)
    methods = defaultdict(list)
    for state, record in enumerate(records):
        targets = np.asarray(record["support"]["residual_action"][1:], dtype=np.float32)
        bank = np.asarray(record["candidate"]["residual_action"][1:], dtype=np.float32)
        true = None
        contact = str(int(bool(record["context"]["current_contact"].item())))
        action_score = np.sum(
            (targets[:, None, :] - bank[None, :, :]) ** 2, axis=2
        )
        methods["B2"].append(
            {
                "selected": _action_assign(targets, bank, codebooks["B2_contact_" + contact]),
                "score": action_score,
            }
        )
        from .stage3_data import true_distance_matrix

        true = true_distance_matrix(record, scalers["consequence_scale"])
        methods["O_FULL"].append(
            {"selected": full_oracle_decoded(record, scalers["consequence_scale"]), "score": true}
        )
        o_k64, _ = true_oracle_decoded(record, scalers["consequence_scale"], PRIMARY_K)
        methods["O_K64"].append({"selected": o_k64, "score": true})

        context = contexts[state]
        c3_target = _ensemble_embedding(
            frozen["C3_NC_BIENCODER"], context, targets, device
        )
        c3_bank = _ensemble_embedding(
            frozen["C3_NC_BIENCODER"], context, bank, device
        )
        c3_score = np.sum(
            (c3_target[:, None, :] - c3_bank[None, :, :]) ** 2, axis=2
        )
        c3_full = np.argmin(c3_score, axis=1)
        c3_atlas = stable_fps(c3_bank, PRIMARY_K)
        c3_k64 = np.asarray(
            [argmin_stable(row[c3_atlas], c3_atlas) for row in c3_score]
        )
        methods["FROZEN_C3_FULL"].append({"selected": c3_full, "score": c3_score})
        methods["FROZEN_C3_K64"].append({"selected": c3_k64, "score": c3_score})
        c4_score = _pair_matrix(
            frozen["C4_NC_PAIR_RANKER"], context, targets, bank, device
        )
        c5 = np.asarray(
            [argmin_stable(row[c3_atlas], c3_atlas) for row in c4_score]
        )
        methods["C5"].append({"selected": c5, "score": c4_score})

        r_target = _ensemble_embedding(reselected, context, targets, device)
        r_bank = _ensemble_embedding(reselected, context, bank, device)
        r_score = np.sum(
            (r_target[:, None, :] - r_bank[None, :, :]) ** 2, axis=2
        )
        fps = stable_fps(r_bank, PRIMARY_K)
        medoids = deterministic_kmedoids(r_bank, PRIMARY_K)
        methods["C3_RESELECT_FULL"].append(
            {"selected": np.argmin(r_score, axis=1), "score": r_score}
        )
        methods["C3_RESELECT_FPS64"].append(
            {
                "selected": np.asarray(
                    [argmin_stable(row[fps], fps) for row in r_score]
                ),
                "score": r_score,
            }
        )
        methods["C3_RESELECT_KMEDOIDS64"].append(
            {
                "selected": np.asarray(
                    [argmin_stable(row[medoids], medoids) for row in r_score]
                ),
                "score": r_score,
            }
        )
    return methods, scalers["consequence_scale"]


def fit_action_whitener(train_cache):
    train_actions = np.concatenate(
        (train_cache["target_residual"], train_cache["candidate_residual"]), axis=0
    )
    center, whitening, _, eigenvalues = covariance_whitener(
        train_actions, regularization=1e-6
    )
    return center, whitening, eigenvalues


def _atlas_and_decoders(models, cache, predicted, whitening, trust_l, device):
    targets = np.asarray(cache["target_residual"], dtype=np.float32)
    candidates = np.asarray(cache["candidate_residual"], dtype=np.float32)
    full = []
    k64 = []
    trusted = {int(value): [] for value in TRUST_REGION_L}
    atlases = []
    target_white = targets.dot(whitening.T)
    candidate_white = candidates.dot(whitening.T)
    action_distance = np.sum(
        (target_white[:, None, :] - candidate_white[None, :, :]) ** 2, axis=2
    )
    for state, context in enumerate(cache["context"]):
        bank_embedding = ensemble_action_embedding(
            models, context, candidates, device
        )
        atlas = deterministic_kmedoids(bank_embedding, PRIMARY_K)
        atlases.append(atlas)
        score = predicted[state]
        full.append(np.argmin(score, axis=1).astype(np.int64))
        k64.append(
            np.asarray([argmin_stable(row[atlas], atlas) for row in score])
        )
        nearest_order = np.argsort(
            action_distance[:, atlas], axis=1, kind="stable"
        )
        for l_value in TRUST_REGION_L:
            selected = []
            for target_id, row in enumerate(score):
                local = atlas[nearest_order[target_id, : int(l_value)]]
                selected.append(argmin_stable(row[local], local))
            trusted[int(l_value)].append(np.asarray(selected, dtype=np.int64))
    output = {
        "full": np.asarray(full),
        "k64": np.asarray(k64),
        "atlas": np.asarray(atlases),
        "trusted": {key: np.asarray(value) for key, value in trusted.items()},
    }
    output["selected_trust"] = output["trusted"][int(trust_l)]
    return output


def _mean_realized(records, selected, consequence_scale, method="CALIBRATION"):
    values = []
    action = []
    contact = []
    for state, record in enumerate(records):
        rows = realized_rows(
            record, selected[state], method, consequence_scale
        )
        values.extend(row["balanced_task_effect_error"] for row in rows)
        action.extend(row["action_reconstruction_rmse"] for row in rows)
        contact.extend(row["contact_mode_preserved"] for row in rows)
    return {
        "balanced_task_effect_error": float(np.mean(values)),
        "action_reconstruction_rmse": float(np.mean(action)),
        "contact_mode_preserved": float(np.mean(contact)),
    }


def calibrate_trust_region(project_root, output_root=None, device_name=None):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    device = _device(device_name)
    records = historical_records("calibration")
    cache = load_cache(_cache_path(SCRATCH_ROOT, "historical_calibration_matrix_cache"))
    _aligned(records, cache)
    family, models = selected_cr_models(output_root, device)
    predicted, member = score_cache(models, cache, device)
    train_cache = load_cache(_cache_path(SCRATCH_ROOT, "train_matrix_cache"))
    center, whitening, eigenvalues = fit_action_whitener(train_cache)
    decoder = _atlas_and_decoders(models, cache, predicted, whitening, 64, device)
    consequence_scale = cache["consequence_scale"]
    trace = []
    for l_value in TRUST_REGION_L:
        trace.append(
            {
                "L": int(l_value),
                **_mean_realized(
                    records,
                    decoder["trusted"][int(l_value)],
                    consequence_scale,
                    "CR_TR_C3_K64_L%d" % int(l_value),
                ),
            }
        )
    selected_l = min(
        trace,
        key=lambda row: (
            row["balanced_task_effect_error"],
            row["action_reconstruction_rmse"],
            row["L"],
        ),
    )["L"]

    # Optional preregistered bounded diagnostic: member dispersion is a
    # symmetric pair score and gamma=0 remains the exact unmodified sham.
    dispersion = np.std(member, axis=0) / np.maximum(
        np.mean(np.abs(member), axis=0), 1e-6
    )
    gamma_trace = []
    for gamma in BOUNDED_CORRECTION_GAMMA:
        corrected = predicted * np.exp(float(gamma) * np.tanh(dispersion))
        corrected_decoder = _atlas_and_decoders(
            models, cache, corrected, whitening, selected_l, device
        )
        gamma_trace.append(
            {
                "gamma": float(gamma),
                **_mean_realized(
                    records,
                    corrected_decoder["selected_trust"],
                    consequence_scale,
                    "CR_TR_C3_K64_GAMMA_%.2f" % float(gamma),
                ),
            }
        )
    selected_gamma = min(
        gamma_trace,
        key=lambda row: (
            row["balanced_task_effect_error"],
            row["action_reconstruction_rmse"],
            row["gamma"],
        ),
    )["gamma"]
    selection_path = os.path.join(output_root, "MODEL_SELECTION.json")
    selection = _load_json(selection_path)
    selection["trust_region_selection"] = {
        "split": "calibration episodes 32-35 only",
        "selected_family": family,
        "trace": trace,
        "selected_L": int(selected_l),
        "selection_order": [
            "lowest calibration realized BALANCED_TASK_EFFECT",
            "lowest calibration action reconstruction RMSE",
            "smallest L",
        ],
        "whitener_fit_split": "expanded train actions only",
        "whitener_center": center.tolist(),
        "whitener": whitening.tolist(),
        "covariance_eigenvalues": eigenvalues.tolist(),
        "development_or_historical_used": False,
    }
    selection["bounded_correction_selection"] = {
        "analysis_only": True,
        "symmetric_score": "ensemble distance coefficient of variation",
        "trace": gamma_trace,
        "selected_gamma": float(selected_gamma),
        "gamma_zero_exact_sham": True,
    }
    atomic_json(selection_path, selection)
    return {
        "family": family,
        "selected_L": int(selected_l),
        "selected_gamma": float(selected_gamma),
    }


def _joint_state_nominal_shuffle(cache, seed=13150403):
    changed = dict(cache)
    context = np.asarray(cache["context"], dtype=np.float32).copy()
    task = cache["task_id"].astype(str)
    order = np.arange(len(context), dtype=np.int64)
    rng = np.random.RandomState(int(seed))
    for task_id in TASK_IDS:
        keep = np.flatnonzero(task == task_id)
        order[keep] = rng.permutation(keep)
    for name in STATE_CONTROL_SLICES + ("nominal_action",):
        left, right = CONTEXT_SLICES[name]
        context[:, left:right] = context[order, left:right]
    changed["context"] = context
    return changed


def _cr_methods(cache, output_root, device):
    selection = _load_json(os.path.join(output_root, "MODEL_SELECTION.json"))
    trust_l = int(selection["trust_region_selection"]["selected_L"])
    selected_gamma = float(
        selection["bounded_correction_selection"]["selected_gamma"]
    )
    whitening = np.asarray(
        selection["trust_region_selection"]["whitener"], dtype=np.float64
    )
    family, models = selected_cr_models(output_root, device)
    predicted, member = score_cache(models, cache, device)
    decoder = _atlas_and_decoders(
        models, cache, predicted, whitening, trust_l, device
    )
    methods = {
        "CR_C3_FULL": {"selected": decoder["full"], "score": predicted},
        "CR_C3_K64": {"selected": decoder["k64"], "score": predicted},
        "CR_TR_C3_K64": {
            "selected": decoder["selected_trust"],
            "score": predicted,
        },
    }
    dispersion = np.std(member, axis=0) / np.maximum(
        np.mean(np.abs(member), axis=0), 1e-6
    )
    corrected = predicted * np.exp(selected_gamma * np.tanh(dispersion))
    corrected_decoder = _atlas_and_decoders(
        models, cache, corrected, whitening, trust_l, device
    )
    methods["CR_TR_C3_K64_BOUNDED_DIAGNOSTIC"] = {
        "selected": corrected_decoder["selected_trust"],
        "score": corrected,
    }
    for member_index, member_score in enumerate(member):
        member_decoder = _atlas_and_decoders(
            [models[member_index]], cache, member_score, whitening, trust_l, device
        )
        methods["CR_TR_C3_K64_MEMBER_%d" % member_index] = {
            "selected": member_decoder["selected_trust"],
            "score": member_score,
            "analysis_only": True,
        }

    for control, method_name in (
        ("ACTION_ONLY", "ACTION_ONLY"),
        ("CONTEXT_SHUFFLED", "CONTEXT_SHUFFLED"),
        ("NOMINAL_SHUFFLED", "NOMINAL_SHUFFLED"),
        ("CONSEQUENCE_LABEL_SHUFFLED", "SHUFFLED_EFFECT"),
        ("REVERSAL_LABEL_SHUFFLED", "REVERSAL_LABEL_SHUFFLED"),
        ("NO_REVERSAL_LOSS", "NO_REVERSAL_LOSS"),
    ):
        control_ensemble = control_models(output_root, control, device)
        control_cache = evaluation_cache_for_control(cache, control)
        control_score, _ = score_cache(control_ensemble, control_cache, device)
        control_decoder = _atlas_and_decoders(
            control_ensemble,
            control_cache,
            control_score,
            whitening,
            trust_l,
            device,
        )
        methods[method_name + "_FULL"] = {
            "selected": control_decoder["full"],
            "score": control_score,
        }
        methods[method_name + "_TR_K64"] = {
            "selected": control_decoder["selected_trust"],
            "score": control_score,
        }

    shuffled_cache = _joint_state_nominal_shuffle(cache)
    shuffled_score, _ = score_cache(models, shuffled_cache, device)
    shuffled_decoder = _atlas_and_decoders(
        models, shuffled_cache, shuffled_score, whitening, trust_l, device
    )
    methods["CR_C3_FULL_STATE_NOMINAL_SHUFFLE"] = {
        "selected": shuffled_decoder["full"],
        "score": shuffled_score,
    }
    return family, methods, predicted


def _bulk_rank_rows(true, score, selected):
    from scipy.stats import kendalltau, rankdata

    true = np.asarray(true, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    selected = np.asarray(selected, dtype=np.int64)
    true_rank = rankdata(true, axis=1, method="average")
    pred_rank = rankdata(score, axis=1, method="average")
    true_center = true_rank - np.mean(true_rank, axis=1, keepdims=True)
    pred_center = pred_rank - np.mean(pred_rank, axis=1, keepdims=True)
    spearman = np.sum(true_center * pred_center, axis=1) / np.maximum(
        np.sqrt(
            np.sum(true_center ** 2, axis=1) * np.sum(pred_center ** 2, axis=1)
        ),
        1e-12,
    )
    true_order = np.argsort(true, axis=1, kind="stable")
    pred_order = np.argsort(score, axis=1, kind="stable")
    relevance = np.exp(-true / 0.15)
    discount = 1.0 / np.log2(np.arange(2, 18))
    actual = np.sum(
        np.take_along_axis(relevance, pred_order[:, :16], axis=1) * discount,
        axis=1,
    )
    ideal = np.sum(
        np.take_along_axis(relevance, true_order[:, :16], axis=1) * discount,
        axis=1,
    )
    rows = []
    for target in range(len(true)):
        tau = float(kendalltau(true[target], score[target], variant="b").statistic)
        if not np.isfinite(tau):
            tau = 0.0
        rows.append(
            {
                "candidate_distance_spearman": float(spearman[target]),
                "kendall_tau": tau,
                "ndcg_at_16": float(actual[target] / max(ideal[target], 1e-12)),
                "oracle_neighbor_recall_at_1": int(
                    pred_order[target, 0] == true_order[target, 0]
                ),
                "oracle_neighbor_recall_at_8": len(
                    set(pred_order[target, :8].tolist())
                    & set(true_order[target, :8].tolist())
                )
                / 8.0,
                "oracle_regret": float(
                    true[target, selected[target]] - np.min(true[target])
                ),
            }
        )
    return rows


def evaluate_split(
    project_root,
    split,
    output_root=None,
    device_name="cpu",
):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    device = _device(device_name)
    records = historical_records(split)
    cache = load_cache(_cache_path(SCRATCH_ROOT, "historical_%s_matrix_cache" % split))
    _aligned(records, cache)
    historical, consequence_scale = _historical_methods(records, output_root, device)
    family, cr, proposed_score = _cr_methods(cache, output_root, device)
    methods = dict(historical)
    for method, value in cr.items():
        methods[method] = [
            {"selected": value["selected"][state], "score": value["score"][state]}
            for state in range(len(records))
        ]

    retrieval = []
    realized = []
    started = time.perf_counter()
    family_ids = np.asarray(cache["direction_family_id"], dtype=np.int64)
    for method in sorted(methods):
        for state, record in enumerate(records):
            payload = methods[method][state]
            true = np.asarray(cache["true_distance"][state], dtype=np.float32)
            ranking = _bulk_rank_rows(true, payload["score"], payload["selected"])
            for target, row in enumerate(ranking):
                row.update(
                    {
                        "split": split,
                        "task_id": str(cache["task_id"][state]),
                        "episode_id": int(cache["episode_id"][state]),
                        "phase": str(cache["phase"][state]),
                        "snapshot_index": int(cache["snapshot_index"][state]),
                        "state_key": str(cache["key"][state]),
                        "direction_family_id": int(family_ids[target]),
                        "target_id": target,
                        "method": method,
                        "selected_bank_index": int(payload["selected"][target]),
                        "inference_latency_ms": 0.0,
                    }
                )
                retrieval.append(row)
            values = realized_rows(
                record,
                payload["selected"],
                method,
                consequence_scale,
                latency_ms=0.0,
                extra={"state_key": str(cache["key"][state])},
            )
            for target, row in enumerate(values):
                row["direction_family_id"] = int(family_ids[target])
                row["target_id"] = target
                realized.append(row)
    elapsed = float(time.perf_counter() - started)
    if split == "development":
        retrieval_name = "DEVELOPMENT_RETRIEVAL.csv"
        realized_name = "DEVELOPMENT_REALIZED.csv"
        summary_name = "development_evaluation_summary.json"
    elif split == "confirmation":
        retrieval_name = "HISTORICAL_EXPLORATORY_RETRIEVAL.csv"
        realized_name = "HISTORICAL_EXPLORATORY_REALIZED.csv"
        summary_name = "historical_exploratory_evaluation_summary.json"
    else:
        retrieval_name = "calibration_retrieval.csv"
        realized_name = "calibration_realized.csv"
        summary_name = "calibration_evaluation_summary.json"
    # Compute summaries before removing row-level aliases/constants.
    realized_summary = summarize_realized(realized, PRIMARY_K)
    # One historical state exists for every (task, episode, phase) tuple, so
    # the verbose state key is a lossless join to the frozen split manifest,
    # not an independent measurement.  Avoid repeating it in large ordinary
    # CSV artifacts so they remain publishable with normal Git.
    omitted_redundant_fields = ("state_key",)
    if split == "confirmation":
        # This split has 2.5x as many states as development.  These columns are
        # either constant for the file or exact aliases of retained per-group
        # metrics; their values remain in the JSON summary.  Omitting them keeps
        # the exploratory row artifact below normal Git's per-file ceiling.
        omitted_redundant_fields += (
            "split",
            "inference_latency_ms",
            "clipped",
            "object_pose_error",
            "tcp_object_relative_pose_error",
        )
    for row in retrieval:
        for field in omitted_redundant_fields:
            row.pop(field, None)
    for row in realized:
        for field in omitted_redundant_fields:
            row.pop(field, None)
    write_csv(os.path.join(output_root, retrieval_name), retrieval)
    write_csv(os.path.join(output_root, realized_name), realized)
    method_means = {
        method: float(
            np.mean(
                [
                    row["balanced_task_effect_error"]
                    for row in realized
                    if row["method"] == method
                ]
            )
        )
        for method in sorted(methods)
    }
    result = {
        "source_split": split,
        "historical_exploratory_only": split == "confirmation",
        "selected_cr_family": family,
        "states": len(records),
        "targets_per_state": len(family_ids),
        "methods": len(methods),
        "retrieval_rows": len(retrieval),
        "realized_rows": len(realized),
        "csv_omitted_redundant_fields": list(omitted_redundant_fields),
        "csv_join_key": ["task_id", "episode_id", "phase"],
        "wall_seconds_excluding_model_scoring": elapsed,
        "method_balanced_task_effect": method_means,
        "realized_summary": realized_summary,
        "retrieval_sha256": sha256_file(os.path.join(output_root, retrieval_name)),
        "realized_sha256": sha256_file(os.path.join(output_root, realized_name)),
    }
    atomic_json(os.path.join(output_root, summary_name), result)
    return result


def _summary_lookup(summary, method, level="pooled", task_id="ALL", phase="ALL"):
    return next(
        row
        for row in summary["realized_summary"]
        if row["method"] == method
        and row["level"] == level
        and row["task_id"] == task_id
        and row["phase"] == phase
    )


def _relative_gain(baseline, method):
    return (float(baseline) - float(method)) / max(float(baseline), 1e-12)


def _task_improvements(summary, method, baseline="B2"):
    values = {}
    for task in TASK_IDS:
        base = _summary_lookup(summary, baseline, "task", task)[
            "balanced_task_effect_error"
        ]
        value = _summary_lookup(summary, method, "task", task)[
            "balanced_task_effect_error"
        ]
        values[task] = _relative_gain(base, value)
    return values


def compute_development_gate(project_root, output_root=None):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    development = _load_json(
        os.path.join(output_root, "development_evaluation_summary.json")
    )
    historical = _load_json(
        os.path.join(output_root, "historical_exploratory_evaluation_summary.json")
    )
    dev = development["method_balanced_task_effect"]
    hist = historical["method_balanced_task_effect"]
    gate_a_tasks = _task_improvements(development, "O_K64")
    gate_a = {
        "relative_gain": _relative_gain(dev["B2"], dev["O_K64"]),
        "task_relative_gains": gate_a_tasks,
        "tasks_improved": sum(value > 0 for value in gate_a_tasks.values()),
        "contact_tasks_improved": sum(
            gate_a_tasks[task] > 0 for task in CONTACT_SENSITIVE_TASKS
        ),
    }
    gate_a["passed"] = bool(
        gate_a["relative_gain"] >= GATES["A"]["oracle_relative_gain_min"]
        and gate_a["tasks_improved"] >= GATES["A"]["tasks_improved_min"]
        and gate_a["contact_tasks_improved"]
        >= GATES["A"]["contact_sensitive_tasks_improved_min"]
    )

    cr_method = "CR_C3_FULL"
    gate_b_tasks = _task_improvements(development, cr_method)
    dev_gain = _relative_gain(dev["B2"], dev[cr_method])
    hist_gain = _relative_gain(hist["B2"], hist[cr_method])
    frozen_gain = _relative_gain(dev["FROZEN_C3_FULL"], dev[cr_method])
    shuffle_gain = _relative_gain(
        dev["B2"], dev["CR_C3_FULL_STATE_NOMINAL_SHUFFLE"]
    )
    action_only_gain = _relative_gain(dev["B2"], dev["ACTION_ONLY_FULL"])
    label_gain = _relative_gain(dev["B2"], dev["SHUFFLED_EFFECT_FULL"])
    # Reversal accuracies are persisted by a dedicated audit after evaluation;
    # initialize explicitly and update below when available.
    reversal_path = os.path.join(output_root, "context_reversal_evaluation.json")
    reversal = _load_json(reversal_path)
    reversal_gain = (
        reversal["development"]["CR_C3_FULL"]["joint_context_reversal_accuracy"]
        - reversal["development"]["FROZEN_C3_FULL"]["joint_context_reversal_accuracy"]
    )
    gate_b = {
        "development_relative_gain": dev_gain,
        "historical_exploratory_relative_gain": hist_gain,
        "frozen_c3_relative_gain": frozen_gain,
        "task_relative_gains": gate_b_tasks,
        "tasks_improved": sum(value > 0 for value in gate_b_tasks.values()),
        "contact_tasks_improved": sum(
            gate_b_tasks[task] > 0 for task in CONTACT_SENSITIVE_TASKS
        ),
        "context_reversal_accuracy_gain_points": reversal_gain,
        "joint_state_nominal_shuffle_gain": shuffle_gain,
        "joint_state_nominal_shuffle_gain_retention": shuffle_gain
        / max(dev_gain, 1e-12),
        "action_only_gain": action_only_gain,
        "consequence_label_shuffled_gain": label_gain,
    }
    gate_b["passed"] = bool(
        dev_gain >= GATES["B"]["episodes_36_39_gain_min"]
        and hist_gain >= GATES["B"]["historical_episodes_40_49_gain_min"]
        and dev_gain >= GATES["B"]["pooled_development_gain_min"]
        and gate_b["tasks_improved"] >= GATES["B"]["tasks_improved_min"]
        and gate_b["contact_tasks_improved"]
        >= GATES["B"]["contact_sensitive_tasks_improved_min"]
        and frozen_gain >= GATES["B"]["frozen_c3_gain_min"]
        and reversal_gain >= GATES["B"]["reversal_accuracy_gain_points_min"]
        and gate_b["joint_state_nominal_shuffle_gain_retention"]
        <= GATES["B"]["joint_state_nominal_shuffle_gain_retention_max"]
        and action_only_gain < dev_gain
        and label_gain < dev_gain
    )
    gate_b["context_independent"] = bool(
        dev_gain > 0
        and gate_b["joint_state_nominal_shuffle_gain_retention"] > 0.5
    )

    tr_method = "CR_TR_C3_K64"
    gate_c_tasks = _task_improvements(development, tr_method)
    tr = _summary_lookup(development, tr_method)
    full = _summary_lookup(development, cr_method)
    b2 = _summary_lookup(development, "B2")
    tr_gain = _relative_gain(
        b2["balanced_task_effect_error"], tr["balanced_task_effect_error"]
    )
    full_gain = _relative_gain(
        b2["balanced_task_effect_error"], full["balanced_task_effect_error"]
    )
    action_degradation = (
        tr["action_reconstruction_rmse"] - b2["action_reconstruction_rmse"]
    ) / max(b2["action_reconstruction_rmse"], 1e-12)
    contact_drop = b2["contact_mode_preserved"] - tr["contact_mode_preserved"]
    gate_c = {
        "relative_gain": tr_gain,
        "full_gain_retention": tr_gain / max(full_gain, 1e-12),
        "task_relative_gains": gate_c_tasks,
        "tasks_improved": sum(value > 0 for value in gate_c_tasks.values()),
        "contact_tasks_improved": sum(
            gate_c_tasks[task] > 0 for task in CONTACT_SENSITIVE_TASKS
        ),
        "action_rmse_degradation": action_degradation,
        "contact_preservation_drop_points": contact_drop,
        "normalized_code_utilization": tr["normalized_code_utilization"],
        "clipping_rate": tr["clipped"],
    }
    gate_c["passed"] = bool(
        tr_gain >= GATES["C"]["realized_relative_gain_min"]
        and gate_c["full_gain_retention"] >= GATES["C"]["full_gain_retention_min"]
        and gate_c["tasks_improved"] >= GATES["C"]["tasks_improved_min"]
        and gate_c["contact_tasks_improved"]
        >= GATES["C"]["contact_sensitive_tasks_improved_min"]
        and action_degradation <= GATES["C"]["action_rmse_degradation_max"]
        and contact_drop <= GATES["C"]["contact_preservation_drop_max_points"]
        and tr["normalized_code_utilization"]
        >= GATES["C"]["normalized_utilization_min"]
        and tr["clipped"] <= GATES["C"]["clipping_rate_max"]
    )
    result = {
        "all_experiments_continue_after_failure": True,
        "gate_A_oracle_headroom": gate_a,
        "gate_B_learned_consequence_metric": gate_b,
        "gate_C_k64_alphabet": gate_c,
        "confirmation_not_yet_used": True,
    }
    atomic_json(os.path.join(output_root, "DEVELOPMENT_GATE.json"), result)
    return result


def evaluate_context_reversals(project_root, output_root=None, device_name="cpu"):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    device = _device(device_name)
    # Read all preregistered robust margins from metadata rather than inferring
    # them from realized pairs.  A physically state-invariant stratum (notably
    # free space) can correctly contain zero strict reversals; indexing a pair
    # row would either crash or tempt an invalid relaxed/fabricated label.
    reversal_metadata = _load_json(
        os.path.join(output_root, "context_reversal_metadata.json")
    )
    margins = {
        (task, phase): float(reversal_metadata["margins"][f"{task}/{phase}"])
        for task in TASK_IDS
        for phase in PHASES
    }
    family, models = selected_cr_models(output_root, device)
    _, old_scalers, frozen_models = load_trained_models(HISTORICAL_OUTPUT, device)
    result = {"selected_cr_family": family}
    for split in ("development", "confirmation"):
        records = historical_records(split)
        cache = load_cache(
            _cache_path(SCRATCH_ROOT, "historical_%s_matrix_cache" % split)
        )
        pairs = {
            name: np.asarray([row[name] for row in reversal_pairs(cache, margins)])
            for name in (
                "state_s1",
                "state_s2",
                "target_id",
                "candidate_i",
                "candidate_j",
                "margin",
                "task_id",
                "phase",
            )
        }
        proposed, _ = score_cache(models, cache, device)
        old_context = transformed_contexts(
            records, old_scalers["context_center"], old_scalers["context_scale"]
        )
        old_score = np.empty_like(proposed)
        for state, record in enumerate(records):
            target = np.asarray(record["support"]["residual_action"][1:], dtype=np.float32)
            bank = np.asarray(record["candidate"]["residual_action"][1:], dtype=np.float32)
            target_embedding = _ensemble_embedding(
                frozen_models["C3_NC_BIENCODER"], old_context[state], target, device
            )
            bank_embedding = _ensemble_embedding(
                frozen_models["C3_NC_BIENCODER"], old_context[state], bank, device
            )
            old_score[state] = np.sum(
                (target_embedding[:, None, :] - bank_embedding[None, :, :]) ** 2,
                axis=2,
            )
        result[split] = {
            "CR_C3_FULL": reversal_accuracy(proposed, pairs),
            "FROZEN_C3_FULL": reversal_accuracy(old_score, pairs),
        }
    atomic_json(os.path.join(output_root, "context_reversal_evaluation.json"), result)
    return result


def evaluate_fresh_confirmation(
    project_root,
    output_root=None,
    device_name="cpu",
):
    from .stage3_metrics import paired_episode_bootstrap
    from .stage4_config import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
    from .stage4_fresh import FRESH_LABEL, load_fresh_records

    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    device = _device(device_name)
    records = load_fresh_records(project_root, output_root, SCRATCH_ROOT)
    cache = load_cache(_cache_path(SCRATCH_ROOT, "fresh_confirmation_matrix_cache"))
    _aligned(records, cache)
    historical, consequence_scale = _historical_methods(records, output_root, device)
    family, cr, _ = _cr_methods(cache, output_root, device)
    # Confirmation reports all frozen historical/proposed/control arms.  No
    # setting is reselected from these outcomes.
    methods = dict(historical)
    for method, value in cr.items():
        methods[method] = [
            {"selected": value["selected"][state], "score": value["score"][state]}
            for state in range(len(records))
        ]
    retrieval = []
    realized = []
    family_ids = np.asarray(cache["direction_family_id"], dtype=np.int64)
    for method in sorted(methods):
        for state, record in enumerate(records):
            payload = methods[method][state]
            true = np.asarray(cache["true_distance"][state], dtype=np.float32)
            ranking = _bulk_rank_rows(true, payload["score"], payload["selected"])
            for target, row in enumerate(ranking):
                row.update(
                    {
                        "split": "fresh_confirmation",
                        "evidence_label": FRESH_LABEL,
                        "new_episode_claim": False,
                        "task_id": str(cache["task_id"][state]),
                        "episode_id": int(cache["episode_id"][state]),
                        "phase": str(cache["phase"][state]),
                        "snapshot_index": int(cache["snapshot_index"][state]),
                        "state_key": str(cache["key"][state]),
                        "direction_family_id": int(family_ids[target]),
                        "target_id": target,
                        "method": method,
                        "selected_bank_index": int(payload["selected"][target]),
                        "inference_latency_ms": 0.0,
                    }
                )
                retrieval.append(row)
            values = realized_rows(
                record,
                payload["selected"],
                method,
                consequence_scale,
                latency_ms=0.0,
                extra={
                    "snapshot_index": int(cache["snapshot_index"][state]),
                    "state_key": str(cache["key"][state]),
                    "evidence_label": FRESH_LABEL,
                    "new_episode_claim": False,
                },
            )
            for target, row in enumerate(values):
                row["direction_family_id"] = int(family_ids[target])
                row["target_id"] = target
                realized.append(row)
    # Compute summaries before removing row-level aliases/constants.
    summary_rows = summarize_realized(realized, PRIMARY_K)
    retrieval_path = os.path.join(output_root, "CONFIRMATION_RETRIEVAL.csv")
    realized_path = os.path.join(output_root, "CONFIRMATION_REALIZED.csv")
    # The evidence label and no-new-episode assertion are constants frozen in
    # the split and summary manifests.  The long state key is losslessly joined
    # from (task, episode, phase, snapshot_index).  One preregistered stratum
    # needed a second timestep from the same source episode, so the snapshot is
    # deliberately retained in both CSVs.  Do not repeat constant strings in
    # ~450k CSV rows:
    # preserving only non-redundant columns keeps the required ordinary CSVs
    # below GitHub's normal-Git per-file limit without compression or LFS.
    omitted_redundant_fields = (
        "evidence_label",
        "new_episode_claim",
        "state_key",
        "split",
        "inference_latency_ms",
        "clipped",
        "object_pose_error",
        "tcp_object_relative_pose_error",
    )
    for row in retrieval:
        for field in omitted_redundant_fields:
            row.pop(field, None)
    for row in realized:
        for field in omitted_redundant_fields:
            row.pop(field, None)
    write_csv(retrieval_path, retrieval)
    write_csv(realized_path, realized)
    method_means = {
        method: float(
            np.mean(
                [
                    row["balanced_task_effect_error"]
                    for row in realized
                    if row["method"] == method
                ]
            )
        )
        for method in sorted(methods)
    }
    bootstrap = {
        "evidence_label": FRESH_LABEL,
        "new_episode_claim": False,
        "cluster_unit": "source task/episode; all four states and targets resampled together",
        "primary": paired_episode_bootstrap(
            realized,
            "CR_TR_C3_K64",
            "B2",
            BOOTSTRAP_REPLICATES,
            BOOTSTRAP_SEED,
        ),
        "full_bank": paired_episode_bootstrap(
            realized,
            "CR_C3_FULL",
            "B2",
            BOOTSTRAP_REPLICATES,
            BOOTSTRAP_SEED + 1,
        ),
        "context_shuffled_control": paired_episode_bootstrap(
            realized,
            "CONTEXT_SHUFFLED_TR_K64",
            "B2",
            BOOTSTRAP_REPLICATES,
            BOOTSTRAP_SEED + 2,
        ),
    }
    atomic_json(os.path.join(output_root, "BOOTSTRAP_RESULTS.json"), bootstrap)
    result = {
        "evidence_label": FRESH_LABEL,
        "new_episode_claim": False,
        "selected_cr_family": family,
        "states": len(records),
        "source_episode_clusters": len(
            {(row["task_id"], row["episode_id"]) for row in realized}
        ),
        "method_balanced_task_effect": method_means,
        "realized_summary": summary_rows,
        "retrieval_rows": len(retrieval),
        "realized_rows": len(realized),
        "csv_omitted_redundant_fields": list(omitted_redundant_fields),
        "csv_join_key": ["task_id", "episode_id", "phase", "snapshot_index"],
        "retrieval_sha256": sha256_file(retrieval_path),
        "realized_sha256": sha256_file(realized_path),
    }
    atomic_json(os.path.join(output_root, "confirmation_evaluation_summary.json"), result)
    return result


def compute_final_disposition(project_root, output_root=None):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    development_gate = _load_json(os.path.join(output_root, "DEVELOPMENT_GATE.json"))
    confirmation = _load_json(
        os.path.join(output_root, "confirmation_evaluation_summary.json")
    )
    bootstrap = _load_json(os.path.join(output_root, "BOOTSTRAP_RESULTS.json"))
    means = confirmation["method_balanced_task_effect"]
    primary = _summary_lookup(confirmation, "CR_TR_C3_K64")
    baseline = _summary_lookup(confirmation, "B2")
    context_control = _summary_lookup(confirmation, "CONTEXT_SHUFFLED_TR_K64")
    primary_gain = _relative_gain(
        baseline["balanced_task_effect_error"],
        primary["balanced_task_effect_error"],
    )
    context_gain = _relative_gain(
        baseline["balanced_task_effect_error"],
        context_control["balanced_task_effect_error"],
    )
    task_gains = _task_improvements(confirmation, "CR_TR_C3_K64")
    action_degradation = (
        primary["action_reconstruction_rmse"]
        - baseline["action_reconstruction_rmse"]
    ) / max(baseline["action_reconstruction_rmse"], 1e-12)
    contact_drop = (
        baseline["contact_mode_preserved"] - primary["contact_mode_preserved"]
    )
    member_gains = {
        "member_%d" % index: _relative_gain(
            means["B2"], means["CR_TR_C3_K64_MEMBER_%d" % index]
        )
        for index in range(3)
    }
    go = {
        "pooled_relative_gain": primary_gain,
        "paired_ci95": bootstrap["primary"]["pooled"]["ci95"],
        "task_relative_gains": task_gains,
        "tasks_improved": sum(value > 0 for value in task_gains.values()),
        "contact_tasks_improved": sum(
            task_gains[task] > 0 for task in CONTACT_SENSITIVE_TASKS
        ),
        "action_rmse_degradation": action_degradation,
        "contact_preservation_drop_points": contact_drop,
        "normalized_code_utilization": primary["normalized_code_utilization"],
        "clipping_rate": primary["clipped"],
        "context_shuffled_gain": context_gain,
        "context_shuffled_gain_retention": context_gain / max(primary_gain, 1e-12),
        "member_relative_gains": member_gains,
        "all_seed_directions_same": all(value > 0 for value in member_gains.values()),
    }
    go["passed"] = bool(
        primary_gain >= GATES["GO"]["pooled_gain_min"]
        and bootstrap["primary"]["pooled"]["ci95"][0]
        > GATES["GO"]["paired_ci_lower_bound_exclusive"]
        and go["tasks_improved"] >= GATES["GO"]["tasks_improved_min"]
        and go["contact_tasks_improved"]
        >= GATES["GO"]["contact_sensitive_tasks_improved_min"]
        and action_degradation <= GATES["GO"]["action_rmse_degradation_max"]
        and contact_drop <= GATES["GO"]["contact_preservation_drop_max_points"]
        and primary["normalized_code_utilization"]
        >= GATES["GO"]["normalized_utilization_min"]
        and primary["clipped"] <= GATES["GO"]["clipping_rate_max"]
        and go["context_shuffled_gain_retention"]
        <= GATES["GO"]["context_shuffled_gain_retention_max"]
        and go["all_seed_directions_same"]
    )

    gate_a = development_gate["gate_A_oracle_headroom"]
    gate_b = development_gate["gate_B_learned_consequence_metric"]
    gate_c = development_gate["gate_C_k64_alphabet"]
    development_summary = _load_json(
        os.path.join(output_root, "development_evaluation_summary.json")
    )
    phase_gains = {
        phase: _relative_gain(
            _summary_lookup(development_summary, "B2", "phase", "ALL", phase)[
                "balanced_task_effect_error"
            ],
            _summary_lookup(
                development_summary, "CR_TR_C3_K64", "phase", "ALL", phase
            )["balanced_task_effect_error"],
        )
        for phase in PHASES
    }
    contact_only = bool(
        phase_gains["contact_onset"] > 0
        and phase_gains["post_contact"] > 0
        and phase_gains["free_space"] <= 0
        and phase_gains["pre_contact"] <= 0
    )
    if not gate_a["passed"]:
        disposition = "REJECT_CONSEQUENCE_HEADROOM"
    elif not gate_b["passed"]:
        disposition = (
            "STATIC_EFFECT_METRIC_ONLY"
            if gate_b.get("context_independent")
            else "REJECT_LEARNED_CONSEQUENCE_METRIC"
        )
    elif not gate_c["passed"]:
        disposition = (
            "NARROW_TO_CONTACT_CONSEQUENCE_METRIC"
            if contact_only
            else "PIVOT_TO_CONSEQUENCE_RETRIEVAL_STEERING"
        )
    else:
        disposition = "GO_TO_SMALL_BC" if go["passed"] else "CONFIRMATION_FAILED"
    result = {
        "evidence_label": confirmation["evidence_label"],
        "new_episode_claim": False,
        "development_gates": development_gate,
        "confirmation_gate": go,
        "development_phase_relative_gains": phase_gains,
        "contact_only_pattern": contact_only,
        "final_disposition": disposition,
        "exact_one_disposition": True,
    }
    atomic_json(os.path.join(output_root, "final_disposition.json"), result)
    return result


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "calibrate",
            "evaluate",
            "reversals",
            "gate",
            "evaluate-fresh",
            "final",
        ),
    )
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--split", choices=("calibration", "development", "confirmation"), default="development"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        result = calibrate_trust_region(
            args.project_root, args.output_root, args.device
        )
    elif args.command == "evaluate":
        result = evaluate_split(
            args.project_root, args.split, args.output_root, args.device
        )
    elif args.command == "reversals":
        result = evaluate_context_reversals(
            args.project_root, args.output_root, args.device
        )
    elif args.command == "gate":
        result = compute_development_gate(args.project_root, args.output_root)
    elif args.command == "evaluate-fresh":
        result = evaluate_fresh_confirmation(
            args.project_root, args.output_root, args.device
        )
    else:
        result = compute_final_disposition(args.project_root, args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
