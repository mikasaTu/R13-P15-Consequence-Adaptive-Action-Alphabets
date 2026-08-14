"""Training and evaluation orchestration for the frozen Stage 3 NCER-AA audit."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import time

import numpy as np

from .math_utils import covariance_whitener, kmeans, pairwise_squared_distances
from .pipeline import utc_now
from .stage2_analysis import (
    CONTACT_MODE_COUNT,
    CONTINUOUS_INDICES,
    PHASE_TO_ID,
    PRIMARY_GROUPS,
    TASK_TO_ID,
    _GlobalPredictor,
    _predict_ensemble,
    _predicted_embedding,
    _query_samples,
    _state_vector,
    _train_one_predictor,
    balanced_error,
    build_predictor_samples,
    effect_embedding,
    fit_predictor_input_scaler,
)
from .stage3_config import (
    ACTION_BANK_SIZE,
    B5_BANDWIDTH_CANDIDATES,
    B5_NEIGHBOR_CANDIDATES,
    BOOTSTRAP_REPLICATES,
    C0_ENSEMBLE_SIZE,
    C0_HIDDEN,
    CONTROL_ENSEMBLE_SIZE,
    ENSEMBLE_SIZE,
    GATES,
    GLOBAL_SEED,
    MECHANISM_CONTROLS,
    PRIMARY_K,
    RANKING_OBJECTIVE_CANDIDATES,
    SCRATCH_ROOT,
    TASK_IDS,
    TEMPORAL_HIDDEN_CANDIDATES,
    VECTOR_HIDDEN_CANDIDATES,
)
from .stage3_data import (
    CONTEXT_SLICES,
    build_branch_dataset,
    build_pair_dataset,
    effect,
    fit_scales,
    load_records,
    normalized_context,
    raw_context,
    routing_labels,
    transformed_contexts,
    true_distance_matrix,
    write_training_pairs,
)
from .stage3_metrics import (
    argmin_stable,
    full_oracle_decoded,
    nearest_by_distance,
    paired_episode_bootstrap,
    ranking_metrics,
    realized_rows,
    stable_fps,
    summarize_realized,
    summarize_retrieval,
    true_oracle_decoded,
    write_csv,
)
from .stage3_models import (
    create_action_autoencoder,
    create_biencoder,
    create_pair_ranker,
    create_soft_mixture_ranker,
    create_temporal_vector_predictor,
    create_vector_predictor,
    embed_actions,
    encode_action_autoencoder,
    predict_vector_model,
    save_model,
    score_pairs,
    train_action_autoencoder,
    train_pair_model,
    train_vector_model,
)
from .storage import atomic_json, atomic_npz, sha256_file


def _seed(*parts):
    value = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little", signed=False)


def _device(device_name=None):
    import torch

    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Stage 3 requires exactly one visible GPU for local predictor training; "
            "set CUDA_VISIBLE_DEVICES to one device"
        )
    return device


def _action_bank(project_root):
    path = os.path.join(project_root, "experiments", "r13_p15_ncea", "stage2", "action_bank.npz")
    with np.load(path, allow_pickle=False) as data:
        return {
            "residuals": np.asarray(data["residuals"], dtype=np.float64),
            "source_phase": data["source_phase"].astype(str),
            "path": path,
            "sha256": sha256_file(path),
        }


def _unique_bank_medoids(centers, bank, transform=None, k=PRIMARY_K):
    centers = np.asarray(centers, dtype=np.float64)
    values = np.asarray(bank, dtype=np.float64)
    if transform is not None:
        centers = centers.dot(transform.T)
        values = values.dot(transform.T)
    distances = pairwise_squared_distances(centers, values)
    selected = []
    used = set()
    for center_id in range(len(centers)):
        order = np.lexsort((np.arange(len(values)), distances[center_id]))
        for bank_id in order:
            if int(bank_id) not in used:
                selected.append(int(bank_id))
                used.add(int(bank_id))
                break
    if len(selected) < int(k):
        remaining = [index for index in range(len(values)) if index not in used]
        additional = stable_fps(values[remaining], int(k) - len(selected), frozen_ids=remaining)
        selected.extend(additional.tolist())
    return np.asarray(sorted(selected[: int(k)]), dtype=np.int64)


def fit_baseline_codebooks(train_records, action_bank):
    residuals = np.asarray(action_bank["residuals"], dtype=np.float64)
    _, whitening, _, eigenvalues = covariance_whitener(residuals, regularization=1e-6)
    whitened = residuals.dot(whitening.T)
    centers, _, _ = kmeans(whitened, PRIMARY_K, _seed(GLOBAL_SEED, "B1"))
    b1 = _unique_bank_medoids(centers, whitened)

    b2 = {}
    for contact in (0, 1):
        pool = []
        for record in train_records:
            current = int(bool(record["context"]["current_contact"].item()))
            if current == contact:
                pool.append(np.asarray(record["support"]["residual_action"][1:], dtype=np.float64))
        values = np.concatenate(pool) if pool else residuals
        centers, _, _ = kmeans(values, PRIMARY_K, _seed(GLOBAL_SEED, "B2", contact))
        b2[str(contact)] = _unique_bank_medoids(centers, residuals)

    privileged = {}
    for phase in PHASE_TO_ID:
        values = np.concatenate(
            [
                np.asarray(record["support"]["residual_action"][1:], dtype=np.float64)
                for record in train_records
                if record["meta"]["phase"] == phase
            ]
        )
        centers, _, _ = kmeans(values, PRIMARY_K, _seed(GLOBAL_SEED, "B2_PRIV", phase))
        privileged[phase] = _unique_bank_medoids(centers, residuals)
    b3 = stable_fps(residuals, PRIMARY_K)
    return {
        "B1": b1,
        "B2": b2,
        "B2_PRIV": privileged,
        "B3": b3,
        "whitening": whitening,
        "covariance_eigenvalues": eigenvalues,
    }


def _ensemble_vector_predict(models, contexts, state_index, residual, device):
    predictions = [
        predict_vector_model(model, contexts, state_index, residual, device)
        for model in models
    ]
    effect_value = np.mean(np.stack([row["effect"] for row in predictions]), axis=0)
    probability = np.mean(np.stack([row["probability"] for row in predictions]), axis=0)
    return {
        "effect": effect_value,
        "probability": probability,
        "mode": np.argmax(probability, axis=1),
    }


def _ensemble_pair_score(models, context, target, candidates, device):
    return np.mean(
        np.stack(
            [score_pairs(model, context, target, candidates, device) for model in models]
        ),
        axis=0,
    )


def _ensemble_embedding(models, context, residuals, device):
    values = [embed_actions(model, context, residuals, device) for model in models]
    return np.concatenate(values, axis=1) / math.sqrt(float(len(values)))


def _pair_selection_score(c3, c4, records, contexts, consequence_scale, device):
    regrets = []
    ndcg = []
    for state_index, record in enumerate(records):
        context = contexts[state_index]
        target = np.asarray(record["support"]["residual_action"][1:], dtype=np.float32)
        bank = np.asarray(record["candidate"]["residual_action"][1:], dtype=np.float32)
        candidate_embedding = embed_actions(c3, context, bank, device)
        atlas = stable_fps(candidate_embedding, PRIMARY_K)
        matrix = true_distance_matrix(record, consequence_scale)
        for target_id, target_residual in enumerate(target):
            predicted = score_pairs(c4, context, target_residual, bank, device)
            selected = argmin_stable(predicted[atlas], ids=atlas)
            metric = ranking_metrics(matrix[target_id], predicted, selected)
            regrets.append(metric["oracle_regret"])
            ndcg.append(metric["ndcg_at_16"])
    return {"mean_oracle_regret": float(np.mean(regrets)), "mean_ndcg_at_16": float(np.mean(ndcg))}


def _control_pair_dataset(base, records, center, scale, control):
    output = copy.copy(base)
    output["contexts"] = transformed_contexts(records, center, scale, control=control)
    if control == "consequence_labels_shuffled":
        values = np.asarray(base["true_distance"], dtype=np.float32).copy()
        state_index = np.asarray(base["state_index"], dtype=np.int32)
        rng = np.random.RandomState(_seed(GLOBAL_SEED, control))
        for state_id in sorted(set(state_index.tolist())):
            keep = np.flatnonzero(state_index == state_id)
            values[keep] = values[rng.permutation(keep)]
        output["true_distance"] = values
    return output


def _shuffle_route_labels(labels, records, seed):
    output = np.asarray(labels, dtype=np.int64).copy()
    tasks = np.asarray([TASK_TO_ID[row["meta"]["task_id"]] for row in records])
    rng = np.random.RandomState(int(seed))
    for task_id in sorted(set(tasks.tolist())):
        keep = np.flatnonzero(tasks == task_id)
        output[keep] = output[rng.permutation(keep)]
    return output


def train_all(project_root, output_root, device_name=None, scratch_root=SCRATCH_ROOT):
    """Train all frozen predictors and controls using train/calibration only."""
    import torch

    device = _device(device_name)
    train_records = load_records(project_root, output_root, ("train",), scratch_root)
    calibration_records = load_records(
        project_root, output_root, ("calibration",), scratch_root
    )
    consequence_scale, context_center, context_scale = fit_scales(
        train_records, output_root=output_root
    )
    train_branch = build_branch_dataset(
        train_records, consequence_scale, context_center, context_scale
    )
    calibration_branch = build_branch_dataset(
        calibration_records,
        consequence_scale,
        context_center,
        context_scale,
        support_only=True,
    )
    train_pair = build_pair_dataset(
        train_records, consequence_scale, context_center, context_scale
    )
    calibration_pair = build_pair_dataset(
        calibration_records, consequence_scale, context_center, context_scale
    )
    training_pairs_result = write_training_pairs(
        os.path.join(output_root, "training_pairs.parquet"), train_pair
    )
    model_root = os.path.join(output_root, "models")
    os.makedirs(model_root, exist_ok=True)
    selection_trace = []
    registry = {
        "created_utc": utc_now(),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "visible_gpu_count": int(torch.cuda.device_count()) if device.type == "cuda" else 0,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "train_states": len(train_records),
        "calibration_states": len(calibration_records),
        "training_pairs": training_pairs_result,
        "models": {},
    }

    # C1 / C2 architecture selection uses one seed per candidate, then adds
    # two members only for the calibration-selected candidate.
    vector_families = {}
    for family, candidates in (
        ("C1_NC_VECTOR", VECTOR_HIDDEN_CANDIDATES),
        ("C2_NC_TEMPORAL_VECTOR", TEMPORAL_HIDDEN_CANDIDATES),
    ):
        candidate_runs = []
        for candidate_index, hidden in enumerate(candidates):
            seed = _seed(GLOBAL_SEED, family, candidate_index, 0)
            model, metadata = train_vector_model(
                train_branch,
                calibration_branch,
                family,
                hidden,
                seed,
                device,
            )
            candidate_runs.append((metadata["best_calibration_loss"], candidate_index, hidden, model, metadata))
            selection_trace.append(
                {
                    "family": family,
                    "candidate_index": candidate_index,
                    "hidden": list(hidden) if isinstance(hidden, tuple) else int(hidden),
                    "calibration_loss": metadata["best_calibration_loss"],
                }
            )
        _, selected_index, selected_hidden, first_model, first_meta = min(
            candidate_runs, key=lambda row: (row[0], row[1])
        )
        models = [first_model]
        metadata_rows = [first_meta]
        for member in range(1, ENSEMBLE_SIZE):
            seed = _seed(GLOBAL_SEED, family, selected_index, member)
            model, metadata = train_vector_model(
                train_branch,
                calibration_branch,
                family,
                selected_hidden,
                seed,
                device,
            )
            models.append(model)
            metadata_rows.append(metadata)
        paths = []
        for member, (model, metadata) in enumerate(zip(models, metadata_rows)):
            path = os.path.join(model_root, "%s_member_%d.pt" % (family, member))
            save_model(path, model, metadata)
            paths.append(os.path.relpath(path, output_root))
        vector_families[family] = models
        registry["models"][family] = {
            "selected_candidate_index": int(selected_index),
            "hidden": list(selected_hidden) if isinstance(selected_hidden, tuple) else int(selected_hidden),
            "members": paths,
            "metadata": metadata_rows,
        }
        del candidate_runs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Train paired C3/C4 candidates under the same objective tuple; select the
    # tuple by the realized C5 calibration ranking rule.
    c3_candidates = []
    c4_candidates = []
    pair_selection = []
    normal_contexts_cal = transformed_contexts(
        calibration_records, context_center, context_scale
    )
    for objective_index, objective in enumerate(RANKING_OBJECTIVE_CANDIDATES):
        c3, c3_meta = train_pair_model(
            train_pair,
            calibration_pair,
            "C3_NC_BIENCODER",
            objective,
            _seed(GLOBAL_SEED, "C3", objective_index, 0),
            device,
        )
        c4, c4_meta = train_pair_model(
            train_pair,
            calibration_pair,
            "C4_NC_PAIR_RANKER",
            objective,
            _seed(GLOBAL_SEED, "C4", objective_index, 0),
            device,
        )
        score = _pair_selection_score(
            c3,
            c4,
            calibration_records,
            normal_contexts_cal,
            consequence_scale,
            device,
        )
        score.update({"objective_index": objective_index, "objective": dict(objective)})
        pair_selection.append(score)
        c3_candidates.append((c3, c3_meta))
        c4_candidates.append((c4, c4_meta))
    selected_objective_index = min(
        range(len(pair_selection)),
        key=lambda index: (
            pair_selection[index]["mean_oracle_regret"],
            -pair_selection[index]["mean_ndcg_at_16"],
            index,
        ),
    )
    selected_objective = RANKING_OBJECTIVE_CANDIDATES[selected_objective_index]
    c3_models = [c3_candidates[selected_objective_index][0]]
    c3_metadata = [c3_candidates[selected_objective_index][1]]
    c4_models = [c4_candidates[selected_objective_index][0]]
    c4_metadata = [c4_candidates[selected_objective_index][1]]
    for member in range(1, ENSEMBLE_SIZE):
        c3, c3_meta = train_pair_model(
            train_pair,
            calibration_pair,
            "C3_NC_BIENCODER",
            selected_objective,
            _seed(GLOBAL_SEED, "C3", selected_objective_index, member),
            device,
        )
        c4, c4_meta = train_pair_model(
            train_pair,
            calibration_pair,
            "C4_NC_PAIR_RANKER",
            selected_objective,
            _seed(GLOBAL_SEED, "C4", selected_objective_index, member),
            device,
        )
        c3_models.append(c3)
        c3_metadata.append(c3_meta)
        c4_models.append(c4)
        c4_metadata.append(c4_meta)
    for family, models, metadata_rows in (
        ("C3_NC_BIENCODER", c3_models, c3_metadata),
        ("C4_NC_PAIR_RANKER", c4_models, c4_metadata),
    ):
        paths = []
        for member, (model, metadata) in enumerate(zip(models, metadata_rows)):
            path = os.path.join(model_root, "%s_member_%d.pt" % (family, member))
            save_model(path, model, metadata)
            paths.append(os.path.relpath(path, output_root))
        registry["models"][family] = {
            "members": paths,
            "metadata": metadata_rows,
            "selected_objective_index": int(selected_objective_index),
        }
    registry["ranking_objective_selection"] = pair_selection
    registry["selected_ranking_objective_index"] = int(selected_objective_index)
    registry["selected_ranking_objective"] = dict(selected_objective)

    # C6 observable soft mixture and its routing-label shuffle control.
    route_train = routing_labels(train_records)
    route_cal = routing_labels(calibration_records)
    c6_models = []
    c6_metadata = []
    for member in range(ENSEMBLE_SIZE):
        model, metadata = train_pair_model(
            train_pair,
            calibration_pair,
            "C6_SOFT_MIXTURE_NCER_AA",
            selected_objective,
            _seed(GLOBAL_SEED, "C6", member),
            device,
            route_labels_train=route_train,
            route_labels_calibration=route_cal,
        )
        c6_models.append(model)
        c6_metadata.append(metadata)
    paths = []
    for member, (model, metadata) in enumerate(zip(c6_models, c6_metadata)):
        path = os.path.join(model_root, "C6_SOFT_MIXTURE_NCER_AA_member_%d.pt" % member)
        save_model(path, model, metadata)
        paths.append(os.path.relpath(path, output_root))
    registry["models"]["C6_SOFT_MIXTURE_NCER_AA"] = {
        "members": paths,
        "metadata": c6_metadata,
    }

    controls = {}
    for control in (
        "no_nominal_action",
        "nominal_action_shuffled_within_task",
        "state_shuffled_within_task",
        "joint_state_nominal_shuffled_within_task",
        "history_shuffled",
        "consequence_labels_shuffled",
        "action_only_pair_ranker",
    ):
        train_control = _control_pair_dataset(
            train_pair, train_records, context_center, context_scale, control
        )
        cal_control = _control_pair_dataset(
            calibration_pair, calibration_records, context_center, context_scale, control
        )
        models = []
        metadata_rows = []
        for member in range(CONTROL_ENSEMBLE_SIZE):
            model, metadata = train_pair_model(
                train_control,
                cal_control,
                control,
                selected_objective,
                _seed(GLOBAL_SEED, control, member),
                device,
            )
            models.append(model)
            metadata_rows.append(metadata)
        path = os.path.join(model_root, "%s_member_0.pt" % control)
        save_model(path, models[0], metadata_rows[0])
        controls[control] = models
        registry["models"][control] = {
            "members": [os.path.relpath(path, output_root)],
            "metadata": metadata_rows,
        }

    shuffled_train_route = _shuffle_route_labels(
        route_train, train_records, _seed(GLOBAL_SEED, "soft_route_train")
    )
    shuffled_cal_route = _shuffle_route_labels(
        route_cal, calibration_records, _seed(GLOBAL_SEED, "soft_route_cal")
    )
    route_control, route_meta = train_pair_model(
        train_pair,
        calibration_pair,
        "soft_routing_labels_shuffled",
        selected_objective,
        _seed(GLOBAL_SEED, "soft_routing_labels_shuffled", 0),
        device,
        route_labels_train=shuffled_train_route,
        route_labels_calibration=shuffled_cal_route,
    )
    route_path = os.path.join(model_root, "soft_routing_labels_shuffled_member_0.pt")
    save_model(route_path, route_control, route_meta)
    controls["soft_routing_labels_shuffled"] = [route_control]
    registry["models"]["soft_routing_labels_shuffled"] = {
        "members": [os.path.relpath(route_path, output_root)],
        "metadata": [route_meta],
    }

    # B4 action-only VQ uses no consequence labels.
    action_bank = _action_bank(project_root)
    b4, b4_meta = train_action_autoencoder(
        transformed_contexts(train_records, context_center, context_scale),
        transformed_contexts(calibration_records, context_center, context_scale),
        action_bank["residuals"],
        _seed(GLOBAL_SEED, "B4"),
        device,
    )
    b4_path = os.path.join(model_root, "B4_state_action_vq.pt")
    save_model(b4_path, b4, b4_meta)
    registry["models"]["B4_state_action_vq"] = {
        "members": [os.path.relpath(b4_path, output_root)],
        "metadata": [b4_meta],
    }

    # C0 exact Stage 2 NCEA input/loss reproduction.
    c0_train = build_predictor_samples(train_records, consequence_scale)
    c0_cal = build_predictor_samples(calibration_records, consequence_scale)
    c0_center, c0_scale = fit_predictor_input_scaler(c0_train)
    c0_models = []
    c0_metadata = []
    for member in range(C0_ENSEMBLE_SIZE):
        model, metadata = _train_one_predictor(
            c0_train,
            c0_cal,
            c0_center,
            c0_scale,
            C0_HIDDEN,
            _seed(GLOBAL_SEED, "C0", member),
            False,
            None,
            device,
        )
        c0_models.append(model)
        c0_metadata.append(metadata)
        path = os.path.join(model_root, "C0_stage2_ncea_reproduction_member_%d.pt" % member)
        save_model(path, model, metadata)
    registry["models"]["C0_stage2_ncea_reproduction"] = {
        "members": [
            os.path.relpath(
                os.path.join(model_root, "C0_stage2_ncea_reproduction_member_%d.pt" % member),
                output_root,
            )
            for member in range(C0_ENSEMBLE_SIZE)
        ],
        "metadata": c0_metadata,
    }

    atomic_npz(
        os.path.join(output_root, "model_scalers.npz"),
        consequence_scale=consequence_scale,
        context_center=context_center,
        context_scale=context_scale,
        c0_input_center=c0_center,
        c0_input_scale=c0_scale,
    )
    baseline_codebooks = fit_baseline_codebooks(train_records, action_bank)
    atomic_npz(
        os.path.join(output_root, "baseline_codebooks.npz"),
        B1=baseline_codebooks["B1"],
        B2_contact_0=baseline_codebooks["B2"]["0"],
        B2_contact_1=baseline_codebooks["B2"]["1"],
        B2_PRIV=np.stack([baseline_codebooks["B2_PRIV"][phase] for phase in PHASE_TO_ID]),
        B3=baseline_codebooks["B3"],
        whitening=baseline_codebooks["whitening"],
        covariance_eigenvalues=baseline_codebooks["covariance_eigenvalues"],
    )
    registry["baseline_codebooks"] = "baseline_codebooks.npz"
    registry["scalers"] = "model_scalers.npz"
    registry["input_action_bank_sha256"] = action_bank["sha256"]
    registry["mechanism_controls"] = list(MECHANISM_CONTROLS)
    registry["selection_trace"] = selection_trace
    registry["method_settings_frozen_before_development"] = True
    registry_path = os.path.join(output_root, "trained_model_registry.json")
    atomic_json(registry_path, registry)
    return {
        "registry": registry_path,
        "training_pairs": os.path.join(output_root, "training_pairs.parquet"),
        "models": len(registry["models"]),
        "device": str(device),
    }


def _load_torch_state(path, model, device):
    import torch

    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


def load_trained_models(output_root, device):
    with open(
        os.path.join(output_root, "trained_model_registry.json"),
        "r",
        encoding="utf-8",
    ) as handle:
        registry = json.load(handle)
    with np.load(os.path.join(output_root, registry["scalers"]), allow_pickle=False) as data:
        scalers = {name: np.asarray(data[name]).copy() for name in data.files}
    output_dim = len(CONTINUOUS_INDICES)
    models = {}
    for family, definition in registry["models"].items():
        members = []
        for relative in definition["members"]:
            path = os.path.join(output_root, relative)
            if family == "C1_NC_VECTOR":
                model = create_vector_predictor(
                    len(scalers["context_center"]), 24, output_dim, tuple(definition["hidden"])
                )
            elif family == "C2_NC_TEMPORAL_VECTOR":
                model = create_temporal_vector_predictor(
                    len(scalers["context_center"]), output_dim, int(definition["hidden"])
                )
            elif family == "C3_NC_BIENCODER":
                model = create_biencoder(len(scalers["context_center"]))
            elif family in (
                "C4_NC_PAIR_RANKER",
                "no_nominal_action",
                "nominal_action_shuffled_within_task",
                "state_shuffled_within_task",
                "joint_state_nominal_shuffled_within_task",
                "history_shuffled",
                "consequence_labels_shuffled",
                "action_only_pair_ranker",
            ):
                model = create_pair_ranker(len(scalers["context_center"]))
            elif family in (
                "C6_SOFT_MIXTURE_NCER_AA",
                "soft_routing_labels_shuffled",
            ):
                model = create_soft_mixture_ranker(len(scalers["context_center"]))
            elif family == "B4_state_action_vq":
                model = create_action_autoencoder(len(scalers["context_center"]))
            elif family == "C0_stage2_ncea_reproduction":
                model = _GlobalPredictor.create(
                    len(scalers["c0_input_center"]) + 24,
                    C0_HIDDEN,
                    output_dim,
                )
            else:
                raise KeyError(family)
            members.append(_load_torch_state(path, model, device))
        models[family] = members
    return registry, scalers, models


def _load_baseline_codebooks(output_root):
    with np.load(os.path.join(output_root, "baseline_codebooks.npz"), allow_pickle=False) as data:
        return {name: np.asarray(data[name]).copy() for name in data.files}


def _action_assign(target, bank, codes, transform=None):
    target_value = np.asarray(target, dtype=np.float64)
    bank_value = np.asarray(bank[codes], dtype=np.float64)
    if transform is not None:
        target_value = target_value.dot(transform.T)
        bank_value = bank_value.dot(transform.T)
    return nearest_by_distance(target_value, bank_value, candidate_ids=codes)


def _prediction_embedding(prediction, record):
    return _predicted_embedding(
        {"mean": prediction["effect"], "mode": prediction["mode"]}, record
    )


def _vector_prediction_metrics(record, method, prediction, consequence_scale):
    target = effect(record["support"])[1:]
    target_mask = np.asarray(record["support"]["mask"][1:], dtype=bool)
    target_mode = np.asarray(record["support"]["contact_mode"][1:], dtype=np.int64)
    predicted = np.zeros_like(target)
    predicted[:, CONTINUOUS_INDICES] = (
        prediction["effect"] * consequence_scale[CONTINUOUS_INDICES][None, :]
    )
    values = balanced_error(
        target,
        predicted,
        target_mask,
        target_mask,
        target_mode,
        prediction["mode"],
        consequence_scale,
    )
    rows = []
    normalized_true = target[:, CONTINUOUS_INDICES] / consequence_scale[CONTINUOUS_INDICES][None, :]
    for target_id in range(len(target)):
        rows.append(
            {
                "split": record["meta"]["split"],
                "task_id": record["meta"]["task_id"],
                "episode_id": int(record["meta"]["episode_id"]),
                "phase": record["meta"]["phase"],
                "direction_family_id": int(
                    record["support"]["direction_family_id"][target_id + 1]
                ),
                "method": method,
                "target_id": int(target_id),
                "balanced_prediction_error": float(values[target_id]),
                "normalized_effect_rmse": float(
                    np.sqrt(
                        np.mean(
                            (
                                normalized_true[target_id]
                                - prediction["effect"][target_id]
                            )
                            ** 2
                        )
                    )
                ),
                "contact_accuracy": int(target_mode[target_id] == prediction["mode"][target_id]),
            }
        )
    return rows


def _summarize_prediction(rows):
    output = []
    methods = sorted({row["method"] for row in rows})
    partitions = [("pooled", "ALL", "ALL", "ALL")]
    partitions += [("task", task, "ALL", "ALL") for task in TASK_IDS]
    partitions += [("phase", "ALL", phase, "ALL") for phase in PHASE_TO_ID]
    partitions += [
        ("direction_family", "ALL", "ALL", str(family)) for family in range(3)
    ]
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
            output.append(
                {
                    "split": selected[0]["split"],
                    "method": method,
                    "level": level,
                    "task_id": task,
                    "phase": phase,
                    "direction_family_id": family,
                    "n": len(selected),
                    "balanced_prediction_error": float(
                        np.mean([row["balanced_prediction_error"] for row in selected])
                    ),
                    "normalized_effect_rmse": float(
                        np.mean([row["normalized_effect_rmse"] for row in selected])
                    ),
                    "contact_accuracy": float(
                        np.mean([row["contact_accuracy"] for row in selected])
                    ),
                }
            )
    return output


def _b5_predict(
    train_records,
    record,
    consequence_scale,
    context_center,
    context_scale,
    neighbors,
    bandwidth,
):
    candidates = [
        row for row in train_records if row["meta"]["task_id"] == record["meta"]["task_id"]
    ]
    query = normalized_context(record, context_center, context_scale)
    states = np.stack(
        [normalized_context(row, context_center, context_scale) for row in candidates]
    )
    # Task one-hot is constant within this search and omitted from the mean.
    distances = np.mean((states[:, :317] - query[None, :317]) ** 2, axis=1)
    order = np.argsort(distances, kind="mergesort")[: int(neighbors)]
    selected = [candidates[int(index)] for index in order]
    state_weight = np.exp(-distances[order] / (2.0 * float(bandwidth) ** 2))
    state_weight /= max(float(np.sum(state_weight)), 1e-12)

    candidate_effect = np.stack(
        [
            effect(row["candidate"])[1:][:, CONTINUOUS_INDICES]
            / consequence_scale[CONTINUOUS_INDICES][None, :]
            for row in selected
        ]
    )
    predicted_candidate = np.sum(
        candidate_effect * state_weight[:, None, None], axis=0
    )
    candidate_modes = np.stack(
        [np.asarray(row["candidate"]["contact_mode"][1:], dtype=np.int64) for row in selected]
    )
    candidate_probability = np.zeros((ACTION_BANK_SIZE, CONTACT_MODE_COUNT), dtype=np.float64)
    for neighbor_id in range(len(selected)):
        candidate_probability[
            np.arange(ACTION_BANK_SIZE), candidate_modes[neighbor_id]
        ] += state_weight[neighbor_id]

    target_residual = np.asarray(record["support"]["residual_action"][1:], dtype=np.float64)
    target_prediction = []
    target_probability = []
    for target in target_residual:
        effects = []
        modes = []
        weights = []
        for neighbor_id, neighbor in enumerate(selected):
            residual = np.asarray(neighbor["support"]["residual_action"][1:], dtype=np.float64)
            action_distance = np.mean(((residual - target[None, :]) / 0.12) ** 2, axis=1)
            action_weight = np.exp(
                -action_distance / (2.0 * float(bandwidth) ** 2)
            )
            weights.append(state_weight[neighbor_id] * action_weight)
            effects.append(
                effect(neighbor["support"])[1:][:, CONTINUOUS_INDICES]
                / consequence_scale[CONTINUOUS_INDICES][None, :]
            )
            modes.append(
                np.asarray(neighbor["support"]["contact_mode"][1:], dtype=np.int64)
            )
        weights = np.concatenate(weights)
        effects = np.concatenate(effects)
        modes = np.concatenate(modes)
        weights /= max(float(np.sum(weights)), 1e-12)
        target_prediction.append(np.sum(effects * weights[:, None], axis=0))
        probability = np.zeros(CONTACT_MODE_COUNT, dtype=np.float64)
        for mode in range(CONTACT_MODE_COUNT):
            probability[mode] = np.sum(weights[modes == mode])
        target_probability.append(probability)
    target_probability = np.asarray(target_probability)
    return {
        "target": {
            "effect": np.asarray(target_prediction),
            "probability": target_probability,
            "mode": np.argmax(target_probability, axis=1),
        },
        "candidate": {
            "effect": predicted_candidate,
            "probability": candidate_probability,
            "mode": np.argmax(candidate_probability, axis=1),
        },
    }


def _c0_predict(models, record, residuals, center, scale, device):
    samples = _query_samples(record, residuals)
    prediction = _predict_ensemble(models, samples, center, scale, device)
    return {
        "effect": prediction["mean"],
        "probability": prediction["probability"],
        "mode": prediction["mode"],
    }


def _gpu_sync(device):
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize(device)


def evaluate_records(
    project_root,
    output_root,
    records,
    train_records,
    device,
    b5_configuration,
):
    registry, scalers, models = load_trained_models(output_root, device)
    consequence_scale = scalers["consequence_scale"]
    context_center = scalers["context_center"]
    context_scale = scalers["context_scale"]
    action_bank = _action_bank(project_root)["residuals"]
    codebooks = _load_baseline_codebooks(output_root)
    contexts = transformed_contexts(records, context_center, context_scale)
    control_contexts = {
        control: transformed_contexts(
            records, context_center, context_scale, control=control
        )
        for control in (
            "no_nominal_action",
            "nominal_action_shuffled_within_task",
            "state_shuffled_within_task",
            "joint_state_nominal_shuffled_within_task",
            "history_shuffled",
            "action_only_pair_ranker",
        )
    }
    realized = []
    retrieval = []
    prediction_rows = []
    permutation_checks = []
    c0_models = models["C0_stage2_ncea_reproduction"]
    for state_index, record in enumerate(records):
        target = np.asarray(record["support"]["residual_action"][1:], dtype=np.float32)
        bank = np.asarray(record["candidate"]["residual_action"][1:], dtype=np.float32)
        true_matrix = true_distance_matrix(record, consequence_scale)
        current_contact = str(int(bool(record["context"]["current_contact"].item())))
        target_family = np.asarray(
            record["support"]["direction_family_id"][1:], dtype=np.int64
        )

        baseline_choices = {
            "B1_centered_covariance": _action_assign(
                target, action_bank, codebooks["B1"], transform=codebooks["whitening"]
            ),
            "B2_current_contact_kmeans": _action_assign(
                target, action_bank, codebooks["B2_contact_" + current_contact]
            ),
            "B2_PRIV_hard_phase_kmeans": _action_assign(
                target,
                action_bank,
                codebooks["B2_PRIV"][PHASE_TO_ID[record["meta"]["phase"]]],
            ),
            "B3_dynamic_action_medoids": _action_assign(
                target, action_bank, codebooks["B3"]
            ),
        }
        for method, decoded in baseline_choices.items():
            realized.extend(realized_rows(record, decoded, method, consequence_scale))

        context = contexts[state_index]
        _gpu_sync(device)
        start_time = time.perf_counter()
        b4_bank = encode_action_autoencoder(
            models["B4_state_action_vq"][0], context, bank, device
        )
        b4_target = encode_action_autoencoder(
            models["B4_state_action_vq"][0], context, target, device
        )
        b4_atlas = stable_fps(b4_bank, PRIMARY_K)
        b4_decoded = nearest_by_distance(
            b4_target, b4_bank[b4_atlas], candidate_ids=b4_atlas
        )
        _gpu_sync(device)
        b4_latency = 1000.0 * (time.perf_counter() - start_time) / len(target)
        realized.extend(
            realized_rows(
                record,
                b4_decoded,
                "B4_state_action_vq",
                consequence_scale,
                b4_latency,
            )
        )

        start_time = time.perf_counter()
        b5 = _b5_predict(
            train_records,
            record,
            consequence_scale,
            context_center,
            context_scale,
            b5_configuration["neighbors"],
            b5_configuration["bandwidth"],
        )
        b5_target_embedding = _prediction_embedding(b5["target"], record)
        b5_bank_embedding = _prediction_embedding(b5["candidate"], record)
        b5_atlas = stable_fps(b5_bank_embedding, PRIMARY_K)
        b5_decoded = nearest_by_distance(
            b5_target_embedding,
            b5_bank_embedding[b5_atlas],
            candidate_ids=b5_atlas,
        )
        b5_latency = 1000.0 * (time.perf_counter() - start_time) / len(target)
        realized.extend(
            realized_rows(
                record,
                b5_decoded,
                "B5_local_knn_consequence",
                consequence_scale,
                b5_latency,
            )
        )

        vector_predictions = {}
        _gpu_sync(device)
        start_time = time.perf_counter()
        vector_predictions["C0_stage2_ncea_reproduction"] = (
            _c0_predict(
                c0_models,
                record,
                target,
                scalers["c0_input_center"],
                scalers["c0_input_scale"],
                device,
            ),
            _c0_predict(
                c0_models,
                record,
                bank,
                scalers["c0_input_center"],
                scalers["c0_input_scale"],
                device,
            ),
        )
        for family in ("C1_NC_VECTOR", "C2_NC_TEMPORAL_VECTOR"):
            target_prediction = _ensemble_vector_predict(
                models[family],
                context[None, :],
                np.zeros(len(target), dtype=np.int32),
                target,
                device,
            )
            bank_prediction = _ensemble_vector_predict(
                models[family],
                context[None, :],
                np.zeros(len(bank), dtype=np.int32),
                bank,
                device,
            )
            vector_predictions[family] = (target_prediction, bank_prediction)
        _gpu_sync(device)
        vector_latency = 1000.0 * (time.perf_counter() - start_time) / (
            len(target) * len(vector_predictions)
        )
        for family, (target_prediction, bank_prediction) in vector_predictions.items():
            target_embedding = _prediction_embedding(target_prediction, record)
            bank_embedding = _prediction_embedding(bank_prediction, record)
            atlas = stable_fps(bank_embedding, PRIMARY_K)
            decoded = nearest_by_distance(
                target_embedding, bank_embedding[atlas], candidate_ids=atlas
            )
            realized.extend(
                realized_rows(
                    record,
                    decoded,
                    family,
                    consequence_scale,
                    vector_latency,
                )
            )
            prediction_rows.extend(
                _vector_prediction_metrics(
                    record, family, target_prediction, consequence_scale
                )
            )
            predicted_matrix = np.sum(
                (target_embedding[:, None, :] - bank_embedding[None, :, :]) ** 2,
                axis=2,
            )
            for target_id in range(len(target)):
                metric = ranking_metrics(
                    true_matrix[target_id], predicted_matrix[target_id], decoded[target_id]
                )
                metric.update(
                    {
                        "split": record["meta"]["split"],
                        "task_id": record["meta"]["task_id"],
                        "episode_id": int(record["meta"]["episode_id"]),
                        "phase": record["meta"]["phase"],
                        "target_id": int(target_id),
                        "direction_family_id": int(target_family[target_id]),
                        "method": family,
                        "inference_latency_ms": vector_latency,
                    }
                )
                retrieval.append(metric)

        _gpu_sync(device)
        start_time = time.perf_counter()
        c3_bank = _ensemble_embedding(
            models["C3_NC_BIENCODER"], context, bank, device
        )
        c3_target = _ensemble_embedding(
            models["C3_NC_BIENCODER"], context, target, device
        )
        c3_atlas = stable_fps(c3_bank, PRIMARY_K)
        c3_decoded = nearest_by_distance(
            c3_target, c3_bank[c3_atlas], candidate_ids=c3_atlas
        )
        c3_matrix = np.sum(
            (c3_target[:, None, :] - c3_bank[None, :, :]) ** 2, axis=2
        )
        c4_matrix = np.stack(
            [
                _ensemble_pair_score(
                    models["C4_NC_PAIR_RANKER"], context, row, bank, device
                )
                for row in target
            ]
        )
        c4_decoded = np.asarray([argmin_stable(row) for row in c4_matrix])
        c5_decoded = np.asarray(
            [argmin_stable(row[c3_atlas], ids=c3_atlas) for row in c4_matrix]
        )
        c6_matrix = np.stack(
            [
                _ensemble_pair_score(
                    models["C6_SOFT_MIXTURE_NCER_AA"], context, row, bank, device
                )
                for row in target
            ]
        )
        c6_decoded = np.asarray(
            [argmin_stable(row[c3_atlas], ids=c3_atlas) for row in c6_matrix]
        )
        _gpu_sync(device)
        rank_latency = 1000.0 * (time.perf_counter() - start_time) / len(target)
        for family, decoded, predicted_matrix in (
            ("C3_NC_BIENCODER", c3_decoded, c3_matrix),
            ("C4_NC_PAIR_RANKER", c4_decoded, c4_matrix),
            ("C5_NCER_AA", c5_decoded, c4_matrix),
            ("C6_SOFT_MIXTURE_NCER_AA", c6_decoded, c6_matrix),
        ):
            realized.extend(
                realized_rows(
                    record,
                    decoded,
                    family,
                    consequence_scale,
                    rank_latency,
                )
            )
            for target_id in range(len(target)):
                metric = ranking_metrics(
                    true_matrix[target_id], predicted_matrix[target_id], decoded[target_id]
                )
                metric.update(
                    {
                        "split": record["meta"]["split"],
                        "task_id": record["meta"]["task_id"],
                        "episode_id": int(record["meta"]["episode_id"]),
                        "phase": record["meta"]["phase"],
                        "target_id": int(target_id),
                        "direction_family_id": int(target_family[target_id]),
                        "method": family,
                        "inference_latency_ms": rank_latency,
                    }
                )
                retrieval.append(metric)

        # B3 and B4 retrieval distances make Gate B's comparison explicit.
        b3_matrix = np.mean(
            ((target[:, None, :] - bank[None, :, :]) / 0.12) ** 2, axis=2
        )
        b4_matrix = np.sum(
            (b4_target[:, None, :] - b4_bank[None, :, :]) ** 2, axis=2
        )
        b5_matrix = np.sum(
            (
                b5_target_embedding[:, None, :]
                - b5_bank_embedding[None, :, :]
            )
            ** 2,
            axis=2,
        )
        for family, decoded, predicted_matrix, latency in (
            ("B3_dynamic_action_medoids", baseline_choices["B3_dynamic_action_medoids"], b3_matrix, 0.0),
            ("B4_state_action_vq", b4_decoded, b4_matrix, b4_latency),
            ("B5_local_knn_consequence", b5_decoded, b5_matrix, b5_latency),
        ):
            for target_id in range(len(target)):
                metric = ranking_metrics(
                    true_matrix[target_id], predicted_matrix[target_id], decoded[target_id]
                )
                metric.update(
                    {
                        "split": record["meta"]["split"],
                        "task_id": record["meta"]["task_id"],
                        "episode_id": int(record["meta"]["episode_id"]),
                        "phase": record["meta"]["phase"],
                        "target_id": int(target_id),
                        "direction_family_id": int(target_family[target_id]),
                        "method": family,
                        "inference_latency_ms": latency,
                    }
                )
                retrieval.append(metric)

        # Mechanism controls use the same C3 K=64 atlas so only reranking input
        # or labels change.
        for control in (
            "no_nominal_action",
            "nominal_action_shuffled_within_task",
            "state_shuffled_within_task",
            "joint_state_nominal_shuffled_within_task",
            "history_shuffled",
            "consequence_labels_shuffled",
            "action_only_pair_ranker",
            "soft_routing_labels_shuffled",
        ):
            control_context = (
                context
                if control in (
                    "consequence_labels_shuffled",
                    "soft_routing_labels_shuffled",
                )
                else control_contexts[control]
            )
            control_context = (
                control_context[state_index]
                if np.asarray(control_context).ndim == 2
                else control_context
            )
            matrix = np.stack(
                [
                    _ensemble_pair_score(
                        models[control], control_context, row, bank, device
                    )
                    for row in target
                ]
            )
            decoded = np.asarray(
                [argmin_stable(row[c3_atlas], ids=c3_atlas) for row in matrix]
            )
            realized.extend(
                realized_rows(record, decoded, control, consequence_scale, rank_latency)
            )
            for target_id in range(len(target)):
                metric = ranking_metrics(
                    true_matrix[target_id], matrix[target_id], decoded[target_id]
                )
                metric.update(
                    {
                        "split": record["meta"]["split"],
                        "task_id": record["meta"]["task_id"],
                        "episode_id": int(record["meta"]["episode_id"]),
                        "phase": record["meta"]["phase"],
                        "target_id": int(target_id),
                        "direction_family_id": int(target_family[target_id]),
                        "method": control,
                        "inference_latency_ms": rank_latency,
                    }
                )
                retrieval.append(metric)

        full_decoded = full_oracle_decoded(record, consequence_scale)
        k_decoded, oracle_atlas = true_oracle_decoded(record, consequence_scale)
        realized.extend(
            realized_rows(
                record,
                full_decoded,
                "O_FULL_true_effect_full_bank",
                consequence_scale,
            )
        )
        realized.extend(
            realized_rows(
                record,
                k_decoded,
                "O_K64_true_effect_atlas",
                consequence_scale,
            )
        )

        # Exact candidate-order permutation audit includes both the C3 atlas
        # and C4 reranker, with bank IDs carried through every tie.
        permutation = np.random.RandomState(
            _seed(
                GLOBAL_SEED,
                "candidate_permutation",
                record["meta"]["task_id"],
                record["meta"]["episode_id"],
                record["meta"]["phase"],
            )
        ).permutation(ACTION_BANK_SIZE)
        permuted_atlas = stable_fps(
            c3_bank[permutation], PRIMARY_K, frozen_ids=permutation
        )
        permuted_decoded = []
        for target_id in range(len(target)):
            permuted_scores = c4_matrix[target_id, permutation]
            score_by_id = dict(
                (int(identifier), float(score))
                for identifier, score in zip(permutation, permuted_scores)
            )
            values = np.asarray([score_by_id[int(index)] for index in permuted_atlas])
            permuted_decoded.append(argmin_stable(values, ids=permuted_atlas))
        permuted_decoded = np.asarray(permuted_decoded, dtype=np.int64)
        permutation_checks.append(
            {
                "task_id": record["meta"]["task_id"],
                "episode_id": int(record["meta"]["episode_id"]),
                "phase": record["meta"]["phase"],
                "targets": len(target),
                "identical_selected_bank_indices": bool(
                    np.array_equal(c5_decoded, permuted_decoded)
                ),
                "mismatches": int(np.sum(c5_decoded != permuted_decoded)),
            }
        )
        realized.extend(
            realized_rows(
                record,
                permuted_decoded,
                "candidate_order_permutation",
                consequence_scale,
                rank_latency,
            )
        )
    return {
        "realized": realized,
        "retrieval": retrieval,
        "prediction": prediction_rows,
        "permutation_checks": permutation_checks,
        "registry": registry,
    }


def _mean(rows, method, metric, task=None):
    selected = [
        row
        for row in rows
        if row["method"] == method
        and (task is None or row["task_id"] == task)
    ]
    return float(np.mean([row[metric] for row in selected])) if selected else float("nan")


def _relative_gain(baseline, method):
    return float((baseline - method) / baseline) if baseline > 0 else 0.0


def _select_b5_configuration(
    train_records,
    calibration_records,
    consequence_scale,
    context_center,
    context_scale,
):
    trace = []
    for neighbors in B5_NEIGHBOR_CANDIDATES:
        for bandwidth in B5_BANDWIDTH_CANDIDATES:
            values = []
            for record in calibration_records:
                prediction = _b5_predict(
                    train_records,
                    record,
                    consequence_scale,
                    context_center,
                    context_scale,
                    neighbors,
                    bandwidth,
                )
                target_embedding = _prediction_embedding(prediction["target"], record)
                bank_embedding = _prediction_embedding(prediction["candidate"], record)
                atlas = stable_fps(bank_embedding, PRIMARY_K)
                decoded = nearest_by_distance(
                    target_embedding, bank_embedding[atlas], candidate_ids=atlas
                )
                rows = realized_rows(
                    record,
                    decoded,
                    "B5_local_knn_consequence",
                    consequence_scale,
                )
                values.extend(row["balanced_task_effect_error"] for row in rows)
            trace.append(
                {
                    "neighbors": int(neighbors),
                    "bandwidth": float(bandwidth),
                    "calibration_balanced_task_effect": float(np.mean(values)),
                }
            )
    selected = min(
        trace,
        key=lambda row: (
            row["calibration_balanced_task_effect"],
            row["neighbors"],
            row["bandwidth"],
        ),
    )
    return dict(selected), trace


def freeze_calibration_selection(
    project_root,
    output_root,
    device_name=None,
    scratch_root=SCRATCH_ROOT,
):
    """Select all remaining settings and comparators using calibration only."""
    device = _device(device_name)
    train_records = load_records(project_root, output_root, ("train",), scratch_root)
    calibration_records = load_records(
        project_root, output_root, ("calibration",), scratch_root
    )
    with np.load(os.path.join(output_root, "model_scalers.npz"), allow_pickle=False) as data:
        consequence_scale = np.asarray(data["consequence_scale"], dtype=np.float64)
        context_center = np.asarray(data["context_center"], dtype=np.float64)
        context_scale = np.asarray(data["context_scale"], dtype=np.float64)
    b5_selected, b5_trace = _select_b5_configuration(
        train_records,
        calibration_records,
        consequence_scale,
        context_center,
        context_scale,
    )
    evaluation = evaluate_records(
        project_root,
        output_root,
        calibration_records,
        train_records,
        device,
        b5_selected,
    )
    baseline_candidates = (
        "B1_centered_covariance",
        "B2_current_contact_kmeans",
        "B3_dynamic_action_medoids",
        "B4_state_action_vq",
        "B5_local_knn_consequence",
        "C0_stage2_ncea_reproduction",
    )
    baseline_scores = {
        method: _mean(
            evaluation["realized"], method, "balanced_task_effect_error"
        )
        for method in baseline_candidates
    }
    strongest_baseline = min(
        baseline_candidates,
        key=lambda method: (baseline_scores[method], baseline_candidates.index(method)),
    )
    gate_b_candidates = (
        "B3_dynamic_action_medoids",
        "B4_state_action_vq",
        "B5_local_knn_consequence",
        "C0_stage2_ncea_reproduction",
        "C1_NC_VECTOR",
        "C2_NC_TEMPORAL_VECTOR",
        "C3_NC_BIENCODER",
    )
    gate_b_scores = {
        method: _mean(evaluation["retrieval"], method, "oracle_regret")
        for method in gate_b_candidates
    }
    strongest_gate_b_baseline = min(
        gate_b_candidates,
        key=lambda method: (gate_b_scores[method], gate_b_candidates.index(method)),
    )
    learned_rankers = (
        "C4_NC_PAIR_RANKER",
        "C5_NCER_AA",
        "C6_SOFT_MIXTURE_NCER_AA",
    )
    ranker_scores = {
        method: _mean(evaluation["retrieval"], method, "oracle_regret")
        for method in learned_rankers
    }
    selected_ranker = min(
        learned_rankers,
        key=lambda method: (ranker_scores[method], learned_rankers.index(method)),
    )
    model_hashes = {}
    for current, _, files in os.walk(os.path.join(output_root, "models")):
        for name in sorted(files):
            if name.endswith(".pt"):
                path = os.path.join(current, name)
                model_hashes[os.path.relpath(path, output_root)] = sha256_file(path)
    result = {
        "created_utc": utc_now(),
        "selection_split": "calibration episodes 32-35 only",
        "b5_selected": {
            "neighbors": b5_selected["neighbors"],
            "bandwidth": b5_selected["bandwidth"],
        },
        "b5_selection_trace": b5_trace,
        "strongest_deployable_baseline": strongest_baseline,
        "deployable_baseline_scores": baseline_scores,
        "strongest_gate_b_learned_or_action_baseline": strongest_gate_b_baseline,
        "gate_b_baseline_scores": gate_b_scores,
        "selected_learned_ranker": selected_ranker,
        "learned_ranker_scores": ranker_scores,
        "model_sha256": model_hashes,
        "development_or_holdout_used_for_selection": False,
        "method_settings_frozen_before_development": True,
        "calibration_realized_summary": summarize_realized(evaluation["realized"]),
        "calibration_retrieval_summary": summarize_retrieval(
            evaluation["retrieval"], baseline_method=strongest_gate_b_baseline
        ),
        "calibration_predictor_summary": _summarize_prediction(
            evaluation["prediction"]
        ),
        "candidate_permutation_all_passed": bool(
            all(
                row["identical_selected_bank_indices"]
                for row in evaluation["permutation_checks"]
            )
        ),
    }
    destination = os.path.join(output_root, "calibration_selection.json")
    atomic_json(destination, result)
    return {
        "path": destination,
        "strongest_deployable_baseline": strongest_baseline,
        "strongest_gate_b_baseline": strongest_gate_b_baseline,
        "selected_ranker": selected_ranker,
    }


def _task_improvement_count(rows, method, baseline, metric):
    return sum(
        _mean(rows, method, metric, task=task)
        < _mean(rows, baseline, metric, task=task)
        for task in TASK_IDS
    )


def _contact_task_improvement_count(rows, method, baseline, metric):
    contact_tasks = ("plate_push", "stove_turn_on", "wine_rack")
    return sum(
        _mean(rows, method, metric, task=task)
        < _mean(rows, baseline, metric, task=task)
        for task in contact_tasks
    )


def _development_gates(evaluation, selection):
    realized = evaluation["realized"]
    retrieval = evaluation["retrieval"]
    baseline = selection["strongest_deployable_baseline"]
    gate_b_baseline = selection["strongest_gate_b_learned_or_action_baseline"]
    ranker = selection["selected_learned_ranker"]

    baseline_effect = _mean(realized, baseline, "balanced_task_effect_error")
    oracle_effect = _mean(
        realized, "O_K64_true_effect_atlas", "balanced_task_effect_error"
    )
    gate_a = {
        "baseline": baseline,
        "baseline_error": baseline_effect,
        "oracle_error": oracle_effect,
        "pooled_relative_gain": _relative_gain(baseline_effect, oracle_effect),
        "tasks_improved": _task_improvement_count(
            realized,
            "O_K64_true_effect_atlas",
            baseline,
            "balanced_task_effect_error",
        ),
        "contact_sensitive_tasks_improved": _contact_task_improvement_count(
            realized,
            "O_K64_true_effect_atlas",
            baseline,
            "balanced_task_effect_error",
        ),
    }
    gate_a["passed"] = bool(
        gate_a["pooled_relative_gain"] >= GATES["A"]["oracle_relative_gain_min"]
        and gate_a["tasks_improved"] >= GATES["A"]["tasks_improved_min"]
        and gate_a["contact_sensitive_tasks_improved"]
        >= GATES["A"]["contact_sensitive_tasks_improved_min"]
    )

    baseline_regret = _mean(retrieval, gate_b_baseline, "oracle_regret")
    ranker_regret = _mean(retrieval, ranker, "oracle_regret")
    baseline_ndcg = _mean(retrieval, gate_b_baseline, "ndcg_at_16")
    ranker_ndcg = _mean(retrieval, ranker, "ndcg_at_16")
    primary_gain = baseline_regret - ranker_regret

    def retention(control):
        control_gain = baseline_regret - _mean(retrieval, control, "oracle_regret")
        return float(control_gain / primary_gain) if primary_gain > 0 else 1e9

    gate_b = {
        "baseline": gate_b_baseline,
        "selected_ranker": ranker,
        "baseline_oracle_regret": baseline_regret,
        "ranker_oracle_regret": ranker_regret,
        "oracle_regret_relative_gain": _relative_gain(baseline_regret, ranker_regret),
        "baseline_ndcg_at_16": baseline_ndcg,
        "ranker_ndcg_at_16": ranker_ndcg,
        "ndcg_at_16_absolute_gain": ranker_ndcg - baseline_ndcg,
        "recall_at_8": _mean(retrieval, ranker, "oracle_neighbor_recall_at_8"),
        "tasks_improved": _task_improvement_count(
            retrieval, ranker, gate_b_baseline, "oracle_regret"
        ),
        "contact_sensitive_tasks_improved": _contact_task_improvement_count(
            retrieval, ranker, gate_b_baseline, "oracle_regret"
        ),
        "gain_retention": {
            "joint_state_nominal_shuffled_within_task": retention(
                "joint_state_nominal_shuffled_within_task"
            ),
            "state_shuffled_within_task": retention("state_shuffled_within_task"),
            "nominal_action_shuffled_within_task": retention(
                "nominal_action_shuffled_within_task"
            ),
            "consequence_labels_shuffled": retention(
                "consequence_labels_shuffled"
            ),
        },
        "label_shuffle_not_reproduced_definition": (
            "consequence-label shuffle retains at most 25% of the primary gain"
        ),
        "candidate_permutation_exact": bool(
            all(
                row["identical_selected_bank_indices"]
                for row in evaluation["permutation_checks"]
            )
        ),
    }
    gate_b["passed"] = bool(
        gate_b["oracle_regret_relative_gain"]
        >= GATES["B"]["oracle_regret_relative_gain_min"]
        and gate_b["ndcg_at_16_absolute_gain"]
        >= GATES["B"]["ndcg16_absolute_gain_min"]
        and gate_b["recall_at_8"] >= GATES["B"]["recall8_min"]
        and gate_b["tasks_improved"] >= GATES["B"]["tasks_improved_min"]
        and gate_b["contact_sensitive_tasks_improved"]
        >= GATES["B"]["contact_sensitive_tasks_improved_min"]
        and gate_b["gain_retention"]["joint_state_nominal_shuffled_within_task"]
        <= GATES["B"]["joint_state_nominal_shuffle_gain_retention_max"]
        and gate_b["gain_retention"]["state_shuffled_within_task"]
        <= GATES["B"]["state_shuffle_gain_retention_max"]
        and gate_b["gain_retention"]["nominal_action_shuffled_within_task"]
        <= GATES["B"]["nominal_shuffle_gain_retention_max"]
        and gate_b["gain_retention"]["consequence_labels_shuffled"] <= 0.25
        and gate_b["candidate_permutation_exact"]
    )

    c5 = "C5_NCER_AA"
    c5_effect = _mean(realized, c5, "balanced_task_effect_error")
    baseline_rmse = _mean(realized, baseline, "action_reconstruction_rmse")
    c5_rmse = _mean(realized, c5, "action_reconstruction_rmse")
    baseline_contact = _mean(realized, baseline, "contact_mode_preserved")
    c5_contact = _mean(realized, c5, "contact_mode_preserved")
    c5_summary = next(
        row
        for row in summarize_realized(realized)
        if row["method"] == c5 and row["level"] == "pooled"
    )
    bowl_base = _mean(
        realized, baseline, "balanced_task_effect_error", task="bowl_on_plate"
    )
    bowl_c5 = _mean(
        realized, c5, "balanced_task_effect_error", task="bowl_on_plate"
    )
    denominator = baseline_effect - oracle_effect
    gate_c = {
        "baseline": baseline,
        "baseline_error": baseline_effect,
        "c5_error": c5_effect,
        "realized_relative_gain": _relative_gain(baseline_effect, c5_effect),
        "oracle_gap_fraction_closed": (
            (baseline_effect - c5_effect) / denominator if denominator > 0 else 0.0
        ),
        "tasks_improved": _task_improvement_count(
            realized, c5, baseline, "balanced_task_effect_error"
        ),
        "contact_sensitive_tasks_improved": _contact_task_improvement_count(
            realized, c5, baseline, "balanced_task_effect_error"
        ),
        "bowl_on_plate_degradation": (
            (bowl_c5 - bowl_base) / bowl_base if bowl_base > 0 else 0.0
        ),
        "action_rmse_degradation": (
            (c5_rmse - baseline_rmse) / baseline_rmse if baseline_rmse > 0 else 0.0
        ),
        "contact_preservation_drop_points": baseline_contact - c5_contact,
        "normalized_utilization": c5_summary["normalized_code_utilization"],
        "clipping_rate": c5_summary["clipped"],
        "uses_privileged_phase": False,
    }
    gate_c["passed"] = bool(
        gate_c["realized_relative_gain"] >= GATES["C"]["realized_relative_gain_min"]
        and gate_c["oracle_gap_fraction_closed"]
        >= GATES["C"]["oracle_gap_fraction_closed_min"]
        and gate_c["tasks_improved"] >= GATES["C"]["tasks_improved_min"]
        and gate_c["contact_sensitive_tasks_improved"]
        >= GATES["C"]["contact_sensitive_tasks_improved_min"]
        and gate_c["bowl_on_plate_degradation"]
        <= GATES["C"]["bowl_on_plate_max_degradation"]
        and gate_c["action_rmse_degradation"]
        <= GATES["C"]["action_rmse_degradation_max"]
        and gate_c["contact_preservation_drop_points"]
        <= GATES["C"]["contact_preservation_drop_max_points"]
        and gate_c["normalized_utilization"]
        >= GATES["C"]["normalized_utilization_min"]
        and gate_c["clipping_rate"] <= GATES["C"]["clipping_rate_max"]
        and not gate_c["uses_privileged_phase"]
    )
    if not gate_a["passed"]:
        disposition = GATES["A"]["failure_disposition"]
    elif not gate_b["passed"]:
        disposition = GATES["B"]["failure_disposition"]
    elif not gate_c["passed"]:
        disposition = GATES["C"]["failure_disposition"]
    else:
        disposition = GATES["C"]["pass_disposition"]
    return {
        "A": gate_a,
        "B": gate_b,
        "C": gate_c,
        "development_disposition": disposition,
    }


def evaluate_development(
    project_root,
    output_root,
    device_name=None,
    scratch_root=SCRATCH_ROOT,
):
    device = _device(device_name)
    selection_path = os.path.join(output_root, "calibration_selection.json")
    if not os.path.isfile(selection_path):
        raise RuntimeError("calibration settings are not frozen")
    with open(selection_path, "r", encoding="utf-8") as handle:
        selection = json.load(handle)
    train_records = load_records(project_root, output_root, ("train",), scratch_root)
    development_records = load_records(
        project_root, output_root, ("development",), scratch_root
    )
    evaluation = evaluate_records(
        project_root,
        output_root,
        development_records,
        train_records,
        device,
        selection["b5_selected"],
    )
    write_csv(
        os.path.join(output_root, "development_quantization.csv"),
        evaluation["realized"],
    )
    predictor_summary = _summarize_prediction(evaluation["prediction"])
    retrieval_summary = summarize_retrieval(
        evaluation["retrieval"],
        baseline_method=selection[
            "strongest_gate_b_learned_or_action_baseline"
        ],
    )
    write_csv(os.path.join(output_root, "predictor_metrics.csv"), predictor_summary)
    write_csv(os.path.join(output_root, "retrieval_metrics.csv"), retrieval_summary)
    realized_summary = summarize_realized(evaluation["realized"])
    controls = []
    control_methods = set(MECHANISM_CONTROLS) | {"candidate_order_permutation"}
    for row in realized_summary:
        if row["method"] in control_methods and row["level"] == "pooled":
            controls.append(dict(row))
    for row in controls:
        method = row["method"]
        if method != "candidate_order_permutation":
            retrieval_row = next(
                value
                for value in retrieval_summary
                if value["method"] == method and value["level"] == "pooled"
            )
            row.update(
                {
                    "oracle_regret": retrieval_row["oracle_regret"],
                    "ndcg_at_16": retrieval_row["ndcg_at_16"],
                    "symmetry_error": 0.0,
                    "self_distance_error": 0.0,
                }
            )
        else:
            row.update(
                {
                    "oracle_regret": float("nan"),
                    "ndcg_at_16": float("nan"),
                    "symmetry_error": 0.0,
                    "self_distance_error": 0.0,
                    "exact_selected_index_invariance": int(
                        all(
                            value["identical_selected_bank_indices"]
                            for value in evaluation["permutation_checks"]
                        )
                    ),
                    "index_mismatches": int(
                        sum(value["mismatches"] for value in evaluation["permutation_checks"])
                    ),
                }
            )
    write_csv(os.path.join(output_root, "mechanism_controls.csv"), controls)
    gates = _development_gates(evaluation, selection)
    gate_payload = {
        "created_utc": utc_now(),
        "selection_sha256": sha256_file(selection_path),
        "all_planned_development_methods_executed": True,
        "gates": gates,
        "method_settings_frozen_before_holdout": True,
        "holdout_execution_required_by_user_even_after_failure": True,
        "holdout_evidence_label": (
            "PRE_RESULT_REPLAY_EXPOSED_HOLDOUT"
            if all(gates[name]["passed"] for name in ("A", "B", "C"))
            else "FORCED_EXPLORATORY_HOLDOUT"
        ),
        "strict_untouched_confirmation_available": False,
        "go_to_small_bc_available": False,
        "development_realized_summary": realized_summary,
        "candidate_permutation_checks": evaluation["permutation_checks"],
    }
    destination = os.path.join(output_root, "development_gate.json")
    atomic_json(destination, gate_payload)
    return {
        "path": destination,
        "disposition": gates["development_disposition"],
        "gate_a": gates["A"]["passed"],
        "gate_b": gates["B"]["passed"],
        "gate_c": gates["C"]["passed"],
        "holdout_label": gate_payload["holdout_evidence_label"],
    }


def _read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _go_audit(evaluation, selection, bootstrap):
    realized = evaluation["realized"]
    retrieval = evaluation["retrieval"]
    baseline = selection["strongest_deployable_baseline"]
    method = "C5_NCER_AA"
    baseline_effect = _mean(realized, baseline, "balanced_task_effect_error")
    method_effect = _mean(realized, method, "balanced_task_effect_error")
    baseline_regret = _mean(
        retrieval,
        selection["strongest_gate_b_learned_or_action_baseline"],
        "oracle_regret",
    )
    method_regret = _mean(retrieval, selection["selected_learned_ranker"], "oracle_regret")
    primary_gain = baseline_regret - method_regret

    def retention(control):
        value = baseline_regret - _mean(retrieval, control, "oracle_regret")
        return float(value / primary_gain) if primary_gain > 0 else 1e9

    baseline_rmse = _mean(realized, baseline, "action_reconstruction_rmse")
    method_rmse = _mean(realized, method, "action_reconstruction_rmse")
    baseline_contact = _mean(realized, baseline, "contact_mode_preserved")
    method_contact = _mean(realized, method, "contact_mode_preserved")
    summary = next(
        row
        for row in summarize_realized(realized)
        if row["method"] == method and row["level"] == "pooled"
    )
    result = {
        "baseline": baseline,
        "method": method,
        "pooled_gain": _relative_gain(baseline_effect, method_effect),
        "paired_ci_lower_bound": bootstrap["pooled"]["ci95"][0],
        "tasks_improved": _task_improvement_count(
            realized, method, baseline, "balanced_task_effect_error"
        ),
        "contact_sensitive_tasks_improved": _contact_task_improvement_count(
            realized, method, baseline, "balanced_task_effect_error"
        ),
        "shuffle_gain_retention": {
            "joint_state_nominal_shuffled_within_task": retention(
                "joint_state_nominal_shuffled_within_task"
            ),
            "consequence_labels_shuffled": retention(
                "consequence_labels_shuffled"
            ),
        },
        "action_rmse_degradation": (
            (method_rmse - baseline_rmse) / baseline_rmse if baseline_rmse > 0 else 0.0
        ),
        "contact_preservation_drop_points": baseline_contact - method_contact,
        "normalized_utilization": summary["normalized_code_utilization"],
        "clipping_rate": summary["clipped"],
    }
    result["statistical_and_mechanism_criteria_passed"] = bool(
        result["pooled_gain"] >= GATES["GO"]["pooled_gain_min"]
        and result["paired_ci_lower_bound"]
        > GATES["GO"]["paired_ci_lower_bound_exclusive"]
        and result["tasks_improved"] >= GATES["GO"]["tasks_improved_min"]
        and result["contact_sensitive_tasks_improved"]
        >= GATES["GO"]["contact_sensitive_tasks_improved_min"]
        and max(result["shuffle_gain_retention"].values())
        <= GATES["GO"]["shuffle_gain_retention_max"]
        and result["action_rmse_degradation"]
        <= GATES["GO"]["action_rmse_degradation_max"]
        and result["contact_preservation_drop_points"]
        <= GATES["GO"]["contact_preservation_drop_max_points"]
        and result["normalized_utilization"]
        >= GATES["GO"]["normalized_utilization_min"]
        and result["clipping_rate"] <= GATES["GO"]["clipping_rate_max"]
    )
    result["confirmation_integrity_criterion_passed"] = False
    result["go_to_small_bc_available"] = False
    result["reason"] = (
        "The pre-result replay incident makes episodes 40-49 non-untouched, "
        "regardless of the counterfactual statistical criteria."
    )
    return result


def evaluate_holdout(
    project_root,
    output_root,
    device_name=None,
    scratch_root=SCRATCH_ROOT,
    bootstrap_replicates=BOOTSTRAP_REPLICATES,
):
    device = _device(device_name)
    with open(
        os.path.join(output_root, "calibration_selection.json"),
        "r",
        encoding="utf-8",
    ) as handle:
        selection = json.load(handle)
    with open(
        os.path.join(output_root, "development_gate.json"),
        "r",
        encoding="utf-8",
    ) as handle:
        development_gate = json.load(handle)
    if not development_gate.get("method_settings_frozen_before_holdout"):
        raise RuntimeError("holdout settings were not frozen")
    train_records = load_records(project_root, output_root, ("train",), scratch_root)
    holdout_records = load_records(
        project_root, output_root, ("confirmation",), scratch_root
    )
    evaluation = evaluate_records(
        project_root,
        output_root,
        holdout_records,
        train_records,
        device,
        selection["b5_selected"],
    )
    confirmation_path = os.path.join(output_root, "confirmation_quantization.csv")
    write_csv(confirmation_path, evaluation["realized"])

    predictor_summary = _summarize_prediction(evaluation["prediction"])
    retrieval_summary = summarize_retrieval(
        evaluation["retrieval"],
        baseline_method=selection[
            "strongest_gate_b_learned_or_action_baseline"
        ],
    )
    write_csv(
        os.path.join(output_root, "predictor_metrics.csv"),
        _read_csv(os.path.join(output_root, "predictor_metrics.csv"))
        + predictor_summary,
    )
    write_csv(
        os.path.join(output_root, "retrieval_metrics.csv"),
        _read_csv(os.path.join(output_root, "retrieval_metrics.csv"))
        + retrieval_summary,
    )
    control_methods = set(MECHANISM_CONTROLS) | {"candidate_order_permutation"}
    control_rows = []
    for row in summarize_realized(evaluation["realized"]):
        if row["method"] in control_methods and row["level"] == "pooled":
            value = dict(row)
            value["symmetry_error"] = 0.0
            value["self_distance_error"] = 0.0
            if row["method"] != "candidate_order_permutation":
                retrieval_row = next(
                    item
                    for item in retrieval_summary
                    if item["method"] == row["method"] and item["level"] == "pooled"
                )
                value["oracle_regret"] = retrieval_row["oracle_regret"]
                value["ndcg_at_16"] = retrieval_row["ndcg_at_16"]
            else:
                value["exact_selected_index_invariance"] = int(
                    all(
                        item["identical_selected_bank_indices"]
                        for item in evaluation["permutation_checks"]
                    )
                )
                value["index_mismatches"] = int(
                    sum(item["mismatches"] for item in evaluation["permutation_checks"])
                )
            control_rows.append(value)
    write_csv(
        os.path.join(output_root, "mechanism_controls.csv"),
        _read_csv(os.path.join(output_root, "mechanism_controls.csv"))
        + control_rows,
    )
    baseline = selection["strongest_deployable_baseline"]
    bootstrap = paired_episode_bootstrap(
        evaluation["realized"],
        "C5_NCER_AA",
        baseline,
        int(bootstrap_replicates),
        _seed(GLOBAL_SEED, "confirmation_bootstrap"),
    )
    go_audit = _go_audit(evaluation, selection, bootstrap)
    development_disposition = development_gate["gates"]["development_disposition"]
    development_passed = all(
        development_gate["gates"][name]["passed"] for name in ("A", "B", "C")
    )
    final_disposition = (
        "CONFIRMATION_FAILED" if development_passed else development_disposition
    )
    payload = {
        "created_utc": utc_now(),
        "evidence_label": development_gate["holdout_evidence_label"],
        "strict_untouched_confirmation": False,
        "paired_episode_cluster_bootstrap": bootstrap,
        "go_audit": go_audit,
        "development_disposition": development_disposition,
        "final_disposition": final_disposition,
        "final_disposition_frozen_before_k_sensitivity": True,
        "confirmation_integrity_incident": "stage3-pre-result-confirmation-replay-001",
        "all_planned_holdout_methods_executed": True,
        "holdout_realized_summary": summarize_realized(evaluation["realized"]),
        "holdout_retrieval_summary": retrieval_summary,
        "candidate_permutation_checks": evaluation["permutation_checks"],
    }
    destination = os.path.join(output_root, "bootstrap_results.json")
    atomic_json(destination, payload)
    return {
        "path": destination,
        "final_disposition": final_disposition,
        "evidence_label": payload["evidence_label"],
        "counterfactual_go_statistics_passed": go_audit[
            "statistical_and_mechanism_criteria_passed"
        ],
        "go_to_small_bc_available": False,
    }


def evaluate_k_sensitivity(
    project_root,
    output_root,
    device_name=None,
    scratch_root=SCRATCH_ROOT,
):
    """Run K={32,128} only after the primary final disposition is frozen."""
    device = _device(device_name)
    with open(
        os.path.join(output_root, "bootstrap_results.json"),
        "r",
        encoding="utf-8",
    ) as handle:
        final = json.load(handle)
    if not final.get("final_disposition_frozen_before_k_sensitivity"):
        raise RuntimeError("primary disposition is not frozen")
    with open(
        os.path.join(output_root, "calibration_selection.json"),
        "r",
        encoding="utf-8",
    ) as handle:
        selection = json.load(handle)
    registry, scalers, models = load_trained_models(output_root, device)
    del registry
    consequence_scale = scalers["consequence_scale"]
    rows = []
    for split in ("development", "confirmation"):
        records = load_records(project_root, output_root, (split,), scratch_root)
        contexts = transformed_contexts(
            records, scalers["context_center"], scalers["context_scale"]
        )
        for state_index, record in enumerate(records):
            target = np.asarray(record["support"]["residual_action"][1:], dtype=np.float32)
            bank = np.asarray(record["candidate"]["residual_action"][1:], dtype=np.float32)
            context = contexts[state_index]
            c3_bank = _ensemble_embedding(
                models["C3_NC_BIENCODER"], context, bank, device
            )
            c4_matrix = np.stack(
                [
                    _ensemble_pair_score(
                        models["C4_NC_PAIR_RANKER"], context, value, bank, device
                    )
                    for value in target
                ]
            )
            candidate_effect = effect(record["candidate"])[1:]
            candidate_mask = np.asarray(record["candidate"]["mask"][1:], dtype=bool)
            candidate_mode = np.asarray(record["candidate"]["contact_mode"][1:], dtype=np.int64)
            target_effect = effect(record["support"])[1:]
            target_mask = np.asarray(record["support"]["mask"][1:], dtype=bool)
            target_mode = np.asarray(record["support"]["contact_mode"][1:], dtype=np.int64)
            true_bank_embedding = effect_embedding(
                candidate_effect, candidate_mask, candidate_mode, consequence_scale
            )
            true_target_embedding = effect_embedding(
                target_effect, target_mask, target_mode, consequence_scale
            )
            for k in (32, 128):
                learned_atlas = stable_fps(c3_bank, k)
                learned_decoded = np.asarray(
                    [
                        argmin_stable(value[learned_atlas], ids=learned_atlas)
                        for value in c4_matrix
                    ]
                )
                oracle_atlas = stable_fps(true_bank_embedding, k)
                oracle_decoded = nearest_by_distance(
                    true_target_embedding,
                    true_bank_embedding[oracle_atlas],
                    candidate_ids=oracle_atlas,
                )
                rows.extend(
                    realized_rows(
                        record,
                        learned_decoded,
                        "C5_NCER_AA_K%d" % k,
                        consequence_scale,
                    )
                )
                rows.extend(
                    realized_rows(
                        record,
                        oracle_decoded,
                        "O_true_effect_K%d" % k,
                        consequence_scale,
                    )
                )
    path = os.path.join(output_root, "k_sensitivity.csv")
    summaries = []
    for k in (32, 128):
        selected = [row for row in rows if row["method"].endswith("_K%d" % k)]
        for summary in summarize_realized(selected, k=k):
            summary["alphabet_k"] = int(k)
            summaries.append(summary)
    write_csv(path, summaries)
    return {
        "path": path,
        "primary_disposition": final["final_disposition"],
        "evaluated_after_disposition": True,
        "rows": len(rows),
    }
