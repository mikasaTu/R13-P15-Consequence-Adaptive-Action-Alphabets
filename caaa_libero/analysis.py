"""Pure-numpy fitting and quantization-plan construction for Stage 1.

This module deliberately has no simulator imports.  Formal jobs can therefore
run simulation with the frozen Python 3.8 LIBERO environment and run artifact
export with the separate analysis environment that provides Zarr / Parquet.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import os
import shutil

import numpy as np

from . import config
from .env_adapter import FEATURE_NAMES
from .math_utils import (
    covariance_whitener,
    farthest_point_codebook,
    kmeans,
    metric_effective_rank,
    pca_fit,
    ridge_jacobian,
    r2_score,
    robust_center_scale,
    spearmanr,
    truncated_pinv,
)
from .pipeline import CONTACT_MODE_TO_ID, utc_now
from .storage import atomic_json, atomic_npz, atomic_text, mark_complete, validate_complete


RAW_METHODS = {
    "euclidean_farthest",
    "global_kmeans",
    "phase_conditioned_kmeans",
}
BASELINE_METHODS = (
    "euclidean_farthest",
    "covariance_mahalanobis",
    "global_kmeans",
    "phase_conditioned_kmeans",
    "pca_kmeans",
    "old_diagonal_sensitivity",
)


def snapshot_key(record):
    return "%s__e%02d__%s" % (
        record["task_id"],
        int(record["episode_id"]),
        record["phase"],
    )


def _scalar(data, name):
    value = data[name]
    return value.item() if value.shape == () else value.tolist()


def load_branch_records(output_root):
    pattern = os.path.join(output_root, "work", "branch_shards", "*", "*.npz")
    records = []
    for path in sorted(glob.glob(pattern)):
        valid, evidence = validate_complete(path)
        if not valid:
            raise RuntimeError("incomplete branch shard %s: %s" % (path, evidence))
        with np.load(path, allow_pickle=False) as data:
            record = {name: np.asarray(data[name]).copy() for name in data.files}
        for name in ("task_id", "task_name", "split", "phase"):
            record[name] = str(_scalar(record, name))
        for name in ("episode_id", "snapshot_index"):
            record[name] = int(_scalar(record, name))
        record["path"] = path
        record["key"] = snapshot_key(record)
        records.append(record)
    expected = len(config.TASKS) * config.N_EPISODES * config.N_PHASES
    if len(records) != expected:
        raise RuntimeError("expected %d branch shards, found %d" % (expected, len(records)))
    return records


def _delta_consequences(record, settled=True):
    values = record["settled"] if settled else record["immediate"]
    return np.asarray(values - values[[0]], dtype=np.float64)


def fit_consequence_scaler(records):
    values = []
    masks = []
    for record in records:
        if record["split"] != "train":
            continue
        delta = _delta_consequences(record, settled=True)[1:]
        values.append(delta)
        masks.append(np.asarray(record["mask"][1:], dtype=bool))
    center, scale = robust_center_scale(np.concatenate(values), np.concatenate(masks))
    # Consequence regression is performed on deltas, so zero is the physically
    # meaningful origin.  The robust center is retained only as audit evidence.
    return {
        "observed_center": center,
        "delta_center": np.zeros_like(center),
        "scale": scale,
    }


def _fit_one(record, scale, ridge):
    x = np.asarray(record["delta_action"], dtype=np.float64)
    y = _delta_consequences(record, settled=True) / scale[None, :]
    mask = np.asarray(record["mask"][0], dtype=bool)
    y[:, ~mask] = 0.0
    fit = np.isclose(record["radius"], 0.05)
    evaluate = np.isclose(record["radius"], 0.10)
    j = ridge_jacobian(x[fit], y[fit], ridge)
    prediction = x[evaluate].dot(j.T)
    truth = y[evaluate]
    residual = prediction - truth
    denom = max(float(np.mean(np.sum(truth * truth, axis=1))), 1e-12)
    normalized_rmse = math.sqrt(float(np.mean(np.sum(residual * residual, axis=1))) / denom)
    return j, {
        "local_r2": r2_score(truth, prediction, mask=np.broadcast_to(mask, truth.shape)),
        "local_normalized_rmse": normalized_rmse,
        "predicted_to_realized_norm_spearman": spearmanr(
            np.linalg.norm(prediction, axis=1), np.linalg.norm(truth, axis=1)
        ),
    }


def select_ridge(records, scale):
    trace = []
    for ridge in config.RIDGE_GRID:
        scores = []
        for record in records:
            if record["split"] != "calibration":
                continue
            _, metrics = _fit_one(record, scale, ridge)
            scores.append(metrics)
        row = {
            "ridge": float(ridge),
            "median_spearman": float(np.nanmedian([x["predicted_to_realized_norm_spearman"] for x in scores])),
            "median_r2": float(np.nanmedian([x["local_r2"] for x in scores])),
            "median_normalized_rmse": float(np.nanmedian([x["local_normalized_rmse"] for x in scores])),
        }
        trace.append(row)
    chosen = max(
        trace,
        key=lambda row: (
            row["median_spearman"],
            row["median_r2"],
            -row["median_normalized_rmse"],
            -row["ridge"],
        ),
    )
    return chosen["ridge"], trace


def fit_all_jacobians(records, scale, ridge):
    models = {}
    for record in records:
        j, metrics = _fit_one(record, scale, ridge)
        models[record["key"]] = {"j": j, "fit_metrics": metrics}
    return models


def select_singular_cutoff(records, models, scale):
    trace = []
    for cutoff in config.SINGULAR_CUTOFF_GRID:
        errors, conditions, ranks = [], [], []
        for record in records:
            if record["split"] != "calibration":
                continue
            j = models[record["key"]]["j"]
            pinv, singular, rank, condition = truncated_pinv(j, cutoff)
            p = pinv.dot(j)
            keep = np.isclose(record["radius"], 0.10)
            x = record["delta_action"][keep]
            truth = _delta_consequences(record, True)[keep] / scale[None, :]
            predicted = (x.dot(p.T)).dot(j.T)
            denom = max(float(np.mean(np.sum(truth * truth, axis=1))), 1e-12)
            errors.append(math.sqrt(float(np.mean(np.sum((predicted - truth) ** 2, axis=1))) / denom))
            conditions.append(condition)
            ranks.append(rank)
        row = {
            "singular_cutoff": float(cutoff),
            "median_projected_normalized_rmse": float(np.nanmedian(errors)),
            "median_condition_number": float(np.nanmedian(conditions)),
            "median_rank": float(np.nanmedian(ranks)),
        }
        trace.append(row)
    # Projection error is primary; condition number breaks practically tied
    # choices (within 1e-6), making selection entirely calibration-only.
    best_error = min(row["median_projected_normalized_rmse"] for row in trace)
    eligible = [row for row in trace if row["median_projected_normalized_rmse"] <= best_error + 1e-6]
    chosen = min(eligible, key=lambda row: (row["median_condition_number"], -row["median_rank"]))
    return chosen["singular_cutoff"], trace


def attach_geometry(records, models, cutoff):
    for record in records:
        model = models[record["key"]]
        pinv, singular, rank, condition = truncated_pinv(model["j"], cutoff)
        model.update(
            {
                "pinv": pinv,
                "projector": pinv.dot(model["j"]),
                "singular_values": singular,
                "rank": rank,
                "condition_number": condition,
                "effective_rank": metric_effective_rank(singular),
            }
        )


def build_permutation_map(records):
    groups = {}
    for record in records:
        group = (record["task_id"], record["split"], record["phase"])
        groups.setdefault(group, []).append(record)
    mapping = {}
    for values in groups.values():
        values = sorted(values, key=lambda row: row["episode_id"])
        for index, record in enumerate(values):
            mapping[record["key"]] = values[(index + 1) % len(values)]["key"]
    return mapping


def _stable_seed(value):
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def random_matched_transform(record, model, metric_regularization):
    singular = np.asarray(model["singular_values"], dtype=np.float64)
    spectrum = np.zeros(config.CHUNK_CONTINUOUS_DIM, dtype=np.float64)
    spectrum[: min(len(singular), len(spectrum))] = singular[: len(spectrum)]
    spectrum = np.sqrt(spectrum * spectrum + float(metric_regularization))
    rng = np.random.RandomState(_stable_seed("random-spd:" + record["key"]))
    q, r = np.linalg.qr(rng.normal(size=(config.CHUNK_CONTINUOUS_DIM,) * 2))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q *= signs[None, :]
    return np.diag(spectrum).dot(q.T)


def state_transform(method, record, models, parameters):
    model = models[record["key"]]
    regularization = float(parameters["metric_regularization"])
    if method == "old_diagonal_sensitivity":
        diagonal = np.sum(model["j"] * model["j"], axis=0) + regularization
        return np.diag(np.sqrt(diagonal))
    if method == "random_spd":
        return random_matched_transform(record, model, regularization)
    if method == "permuted_j":
        other = models[parameters["permutation_map"][record["key"]]]["j"]
        return np.vstack([other, math.sqrt(regularization) * np.eye(config.CHUNK_CONTINUOUS_DIM)])
    if method == "caaa_v2":
        return np.vstack([model["j"], math.sqrt(regularization) * np.eye(config.CHUNK_CONTINUOUS_DIM)])
    raise KeyError(method)


def transformed_action(method, sensitive_action, record, models, parameters):
    x = np.asarray(sensitive_action, dtype=np.float64)
    if method in RAW_METHODS:
        return x
    if method == "covariance_mahalanobis":
        return parameters["covariance_whitener"].dot(x - parameters["covariance_mean"])
    if method == "pca_kmeans":
        return parameters["pca_components"].dot(x - parameters["pca_mean"])
    return state_transform(method, record, models, parameters).dot(x)


def inverse_transformed_action(method, transformed, record, models, parameters):
    z = np.asarray(transformed, dtype=np.float64)
    if method in RAW_METHODS:
        return z
    if method == "covariance_mahalanobis":
        return parameters["covariance_mean"] + parameters["covariance_dewhitener"].dot(z)
    if method == "pca_kmeans":
        return parameters["pca_mean"] + parameters["pca_components"].T.dot(z)
    return np.linalg.pinv(state_transform(method, record, models, parameters), rcond=1e-10).dot(z)


def _candidate_rows(record):
    return np.flatnonzero(np.isclose(record["radius"], 0.10))


def _sensitive_samples(records, models, split):
    samples = []
    for record in records:
        if record["split"] != split:
            continue
        p = models[record["key"]]["projector"]
        for row in _candidate_rows(record):
            action = np.asarray(record["action_cont"][row], dtype=np.float64)
            samples.append({"record": record, "row": int(row), "x": p.dot(action)})
    return samples


def _linear_decoding_error(samples, decoded):
    errors = []
    for sample, value in zip(samples, decoded):
        record = sample["record"]
        # Consequence error predicted by the independently fit local Jacobian.
        delta = value - sample["x"]
        errors.append(float(np.linalg.norm(sample["model_j"].dot(delta))))
    return float(np.mean(errors))


def select_covariance_and_pca(train_samples, calibration_samples, models):
    train_x = np.stack([sample["x"] for sample in train_samples])
    for sample in calibration_samples:
        sample["model_j"] = models[sample["record"]["key"]]["j"]
    calibration_x = np.stack([sample["x"] for sample in calibration_samples])

    covariance_trace = []
    covariance_candidates = {}
    for regularization in config.METRIC_REGULARIZATION_GRID:
        mean, white, dewhite, spectrum = covariance_whitener(train_x, regularization)
        train_z = (train_x - mean).dot(white.T)
        centers, _, _ = kmeans(train_z, config.PRIMARY_K, config.GLOBAL_SEED + 101, max_iter=25)
        calibration_z = (calibration_x - mean).dot(white.T)
        labels = np.argmin(
            np.sum((calibration_z[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1
        )
        decoded = mean[None, :] + centers[labels].dot(dewhite.T)
        error = _linear_decoding_error(calibration_samples, decoded)
        row = {"regularization": float(regularization), "calibration_linear_effect_error": error}
        covariance_trace.append(row)
        covariance_candidates[float(regularization)] = (mean, white, dewhite, spectrum)
    chosen_cov = min(covariance_trace, key=lambda row: (row["calibration_linear_effect_error"], row["regularization"]))

    pca_trace = []
    pca_candidates = {}
    for rank in config.PCA_RANK_GRID:
        mean, components, singular = pca_fit(train_x, rank)
        train_z = (train_x - mean).dot(components.T)
        centers, _, _ = kmeans(train_z, config.PRIMARY_K, config.GLOBAL_SEED + 211, max_iter=25)
        calibration_z = (calibration_x - mean).dot(components.T)
        labels = np.argmin(
            np.sum((calibration_z[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1
        )
        decoded = mean[None, :] + centers[labels].dot(components)
        error = _linear_decoding_error(calibration_samples, decoded)
        row = {"rank": int(rank), "calibration_linear_effect_error": error}
        pca_trace.append(row)
        pca_candidates[int(rank)] = (mean, components, singular)
    chosen_pca = min(pca_trace, key=lambda row: (row["calibration_linear_effect_error"], row["rank"]))
    return {
        "covariance": covariance_candidates[chosen_cov["regularization"]],
        "covariance_regularization": chosen_cov["regularization"],
        "covariance_trace": covariance_trace,
        "pca": pca_candidates[chosen_pca["rank"]],
        "pca_rank": chosen_pca["rank"],
        "pca_trace": pca_trace,
    }


def select_metric_regularization(records, models):
    trace = []
    for regularization in config.METRIC_REGULARIZATION_GRID:
        correlations = []
        for record in records:
            if record["split"] != "calibration":
                continue
            model = models[record["key"]]
            keep = _candidate_rows(record)
            delta = record["delta_action"][keep].dot(model["projector"].T)
            metric_norm = np.sqrt(
                np.sum(delta.dot(model["j"].T) ** 2, axis=1)
                + float(regularization) * np.sum(delta * delta, axis=1)
            )
            realized = _delta_consequences(record, True)[keep]
            realized_norm = np.linalg.norm(realized, axis=1)
            correlations.append(spearmanr(metric_norm, realized_norm))
        trace.append(
            {
                "regularization": float(regularization),
                "median_metric_to_unscaled_consequence_spearman": float(np.nanmedian(correlations)),
            }
        )
    chosen = max(trace, key=lambda row: (row["median_metric_to_unscaled_consequence_spearman"], -row["regularization"]))
    return chosen["regularization"], trace


def fit_codebooks(train_samples, models, parameters, output_root):
    manifest = []
    for method_index, method in enumerate(config.METHODS):
        for k in config.K_VALUES:
            destination = os.path.join(output_root, "alphabet_codebooks", "%s_k%d.npz" % (method, k))
            if method == "phase_conditioned_kmeans":
                phase_centers = []
                phase_inertia = []
                for phase_index, phase in enumerate(config.PHASES):
                    subset = [sample for sample in train_samples if sample["record"]["phase"] == phase]
                    z = np.stack(
                        [transformed_action(method, sample["x"], sample["record"], models, parameters) for sample in subset]
                    )
                    centers, labels, inertia = kmeans(
                        z, k, config.GLOBAL_SEED + 1009 * method_index + 17 * k + phase_index, max_iter=35
                    )
                    phase_centers.append(centers)
                    phase_inertia.append(inertia)
                centers = np.stack(phase_centers)
                train_utilization = float("nan")
                atomic_npz(
                    destination,
                    method=np.asarray(method),
                    k=np.asarray(k, dtype=np.int32),
                    phases=np.asarray(config.PHASES),
                    centers=centers,
                    inertia=np.asarray(phase_inertia),
                )
            else:
                z = np.stack(
                    [transformed_action(method, sample["x"], sample["record"], models, parameters) for sample in train_samples]
                )
                if method == "euclidean_farthest":
                    centers = farthest_point_codebook(z, k, config.GLOBAL_SEED + 17 * k)
                    labels = np.argmin(np.sum((z[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)
                    inertia = float(np.sum((z - centers[labels]) ** 2))
                else:
                    centers, labels, inertia = kmeans(
                        z, k, config.GLOBAL_SEED + 1009 * method_index + 17 * k, max_iter=35
                    )
                train_utilization = float(len(np.unique(labels))) / float(k)
                atomic_npz(
                    destination,
                    method=np.asarray(method),
                    k=np.asarray(k, dtype=np.int32),
                    centers=centers,
                    inertia=np.asarray(inertia),
                    train_utilization=np.asarray(train_utilization),
                )
            marker = mark_complete(destination, {"method": method, "k": int(k), "created_utc": utc_now()})
            manifest.append(
                {
                    "method": method,
                    "k": int(k),
                    "path": destination,
                    "marker": marker,
                    "train_utilization": train_utilization,
                }
            )
    atomic_json(
        os.path.join(output_root, "alphabet_codebooks", "manifest.json"),
        {"created_utc": utc_now(), "codebooks": manifest},
    )
    return manifest


def load_codebooks(output_root):
    result = {}
    for method in config.METHODS:
        for k in config.K_VALUES:
            path = os.path.join(output_root, "alphabet_codebooks", "%s_k%d.npz" % (method, k))
            valid, evidence = validate_complete(path)
            if not valid:
                raise RuntimeError("invalid codebook %s: %s" % (path, evidence))
            with np.load(path, allow_pickle=False) as data:
                result[(method, k)] = np.asarray(data["centers"], dtype=np.float64).copy()
    return result


def _nearest_center(value, centers):
    distances = np.sum((centers - value[None, :]) ** 2, axis=1)
    index = int(np.argmin(distances))
    return index, centers[index]


def write_quantization_plans(records, models, parameters, output_root):
    codebooks = load_codebooks(output_root)
    plans = []
    for record in records:
        if record["split"] not in ("calibration", "test"):
            continue
        model = models[record["key"]]
        p = model["projector"]
        identity_minus_p = np.eye(config.CHUNK_CONTINUOUS_DIM) - p
        decoded_actions = []
        original_actions = []
        methods = []
        ks = []
        candidate_rows = []
        code_indices = []
        clipped_coordinates = []
        for row in _candidate_rows(record):
            direction = int(record["direction"][row])
            requested_ks = [config.PRIMARY_K]
            if record["split"] == "test" and direction < 6:
                requested_ks.extend(config.SENSITIVITY_K)
            action = np.asarray(record["action_cont"][row], dtype=np.float64)
            sensitive = p.dot(action)
            null_residual = identity_minus_p.dot(action)
            for k in requested_ks:
                for method in config.METHODS:
                    transformed = transformed_action(method, sensitive, record, models, parameters)
                    centers = codebooks[(method, int(k))]
                    if method == "phase_conditioned_kmeans":
                        centers = centers[config.PHASES.index(record["phase"])]
                    code_index, center = _nearest_center(transformed, centers)
                    decoded_sensitive = inverse_transformed_action(
                        method, center, record, models, parameters
                    )
                    decoded = p.dot(decoded_sensitive) + null_residual
                    clipped = np.clip(decoded, -1.0, 1.0)
                    clipped_coordinates.append(int(np.sum(np.abs(decoded - clipped) > 1e-12)))
                    full = np.asarray(record["base_actions"], dtype=np.float64).copy()
                    full[:, config.CONTINUOUS_ACTION_INDICES] = clipped.reshape(
                        config.CHUNK_HORIZON, len(config.CONTINUOUS_ACTION_INDICES)
                    )
                    decoded_actions.append(full)
                    original_actions.append(np.asarray(record["action_full"][row], dtype=np.float64))
                    methods.append(method)
                    ks.append(int(k))
                    candidate_rows.append(int(row))
                    code_indices.append(code_index)
        destination = os.path.join(output_root, "work", "quantization_plans", record["task_id"], record["key"] + ".npz")
        row_array = np.asarray(candidate_rows, dtype=np.int16)
        atomic_npz(
            destination,
            task_id=np.asarray(record["task_id"]),
            task_name=np.asarray(record["task_name"]),
            episode_id=np.asarray(record["episode_id"], dtype=np.int16),
            split=np.asarray(record["split"]),
            phase=np.asarray(record["phase"]),
            snapshot_index=np.asarray(record["snapshot_index"], dtype=np.int32),
            methods=np.asarray(methods),
            k=np.asarray(ks, dtype=np.int16),
            candidate_row=row_array,
            direction=np.asarray(record["direction"][row_array], dtype=np.int16),
            sign=np.asarray(record["sign"][row_array], dtype=np.int8),
            radius=np.asarray(record["radius"][row_array], dtype=np.float64),
            code_index=np.asarray(code_indices, dtype=np.int16),
            clipped_coordinates=np.asarray(clipped_coordinates, dtype=np.int16),
            decoded_actions=np.asarray(decoded_actions, dtype=np.float64),
            original_actions=np.asarray(original_actions, dtype=np.float64),
            original_immediate=np.asarray(record["immediate"][row_array], dtype=np.float64),
            original_settled=np.asarray(record["settled"][row_array], dtype=np.float64),
            original_mask=np.asarray(record["mask"][row_array], dtype=np.uint8),
            original_contact_mode=np.asarray(record["contact_mode"][row_array], dtype=np.int8),
            original_settled_success=np.asarray(record["settled_success"][row_array], dtype=np.uint8),
            original_immediate_progress=np.asarray(record["immediate_progress"][row_array], dtype=np.float64),
            original_settled_progress=np.asarray(record["settled_progress"][row_array], dtype=np.float64),
        )
        marker = mark_complete(
            destination,
            {"snapshot_key": record["key"], "rows": len(methods), "created_utc": utc_now()},
        )
        plans.append({"path": destination, "marker": marker, "rows": len(methods)})
    atomic_json(
        os.path.join(output_root, "work", "quantization_plan_manifest.json"),
        {"created_utc": utc_now(), "plans": plans, "count": len(plans)},
    )
    return plans


def metric_correlations(records, models, parameters):
    rows = []
    for record in records:
        model = models[record["key"]]
        keep = _candidate_rows(record)
        delta = np.asarray(record["delta_action"][keep], dtype=np.float64)
        sensitive_delta = delta.dot(model["projector"].T)
        realized = _delta_consequences(record, True)[keep] / parameters["consequence_scale"][None, :]
        realized_norm = np.linalg.norm(realized, axis=1)
        for method in config.METHODS:
            if method in RAW_METHODS:
                metric_norm = np.linalg.norm(sensitive_delta, axis=1)
            elif method == "covariance_mahalanobis":
                metric_norm = np.linalg.norm(sensitive_delta.dot(parameters["covariance_whitener"].T), axis=1)
            elif method == "pca_kmeans":
                metric_norm = np.linalg.norm(sensitive_delta.dot(parameters["pca_components"].T), axis=1)
            else:
                transform = state_transform(method, record, models, parameters)
                metric_norm = np.linalg.norm(sensitive_delta.dot(transform.T), axis=1)
            row = {
                "task_id": record["task_id"],
                "episode_id": record["episode_id"],
                "split": record["split"],
                "phase": record["phase"],
                "method": method,
                "ridge": float(parameters["ridge"]),
                "singular_cutoff": float(parameters["singular_cutoff"]),
                "local_r2": float(model["fit_metrics"]["local_r2"]),
                "local_normalized_rmse": float(model["fit_metrics"]["local_normalized_rmse"]),
                "local_predicted_to_realized_spearman": float(
                    model["fit_metrics"]["predicted_to_realized_norm_spearman"]
                ),
                "effective_rank": float(model["effective_rank"]),
                "truncated_rank": int(model["rank"]),
                "condition_number": float(model["condition_number"]),
                "metric_to_consequence_spearman": float(spearmanr(metric_norm, realized_norm)),
            }
            rows.append(row)
    return rows


def export_jacobian_metrics(rows, output_root):
    jsonl = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    atomic_text(os.path.join(output_root, "work", "jacobian_metrics.jsonl"), jsonl)
    try:
        import pandas as pd

        destination = os.path.join(output_root, "jacobian_metrics.parquet")
        temporary = destination + ".incomplete"
        pd.DataFrame(rows).to_parquet(temporary, index=False)
        os.replace(temporary, destination)
    except Exception as error:
        raise RuntimeError("Parquet export requires pandas+pyarrow: %s" % (error,))


def consolidate_zarr(records, output_root):
    try:
        import zarr
    except Exception as error:
        raise RuntimeError("Zarr export requires zarr: %s" % (error,))
    destination = os.path.join(output_root, "branch_rollouts.zarr")
    temporary = destination + ".incomplete"
    if os.path.isdir(temporary):
        shutil.rmtree(temporary)
    root = zarr.open_group(temporary, mode="w")
    root.attrs.update(
        {
            "created_utc": utc_now(),
            "schema": "caaa-libero-branch-rollouts-v1",
            "tasks": [task["task_id"] for task in config.TASKS],
            "feature_names": list(FEATURE_NAMES),
        }
    )
    array_names = (
        "snapshot_state",
        "base_actions",
        "action_cont",
        "delta_action",
        "action_full",
        "direction",
        "sign",
        "radius",
        "initial",
        "immediate",
        "settled",
        "mask",
        "initial_success",
        "immediate_success",
        "settled_success",
        "initial_progress",
        "immediate_progress",
        "settled_progress",
        "contact_mode",
        "contact_sequence",
        "final_state",
    )
    for record in records:
        group = root.require_group(
            "%s/episode_%02d/%s" % (record["task_id"], record["episode_id"], record["phase"])
        )
        group.attrs.update(
            {
                "task_name": record["task_name"],
                "split": record["split"],
                "snapshot_index": record["snapshot_index"],
                "source_npz": record["path"],
            }
        )
        for name in array_names:
            value = np.asarray(record[name])
            chunks = value.shape if value.ndim == 0 else (min(value.shape[0], 32),) + value.shape[1:]
            group.create_dataset(name, data=value, chunks=chunks, overwrite=True)
    if os.path.isdir(destination):
        shutil.rmtree(destination)
    os.replace(temporary, destination)


def save_models(records, models, parameters, output_root):
    directory = os.path.join(output_root, "work", "jacobians")
    os.makedirs(directory, exist_ok=True)
    for record in records:
        model = models[record["key"]]
        path = os.path.join(directory, record["key"] + ".npz")
        atomic_npz(
            path,
            j=model["j"],
            pinv=model["pinv"],
            projector=model["projector"],
            singular_values=model["singular_values"],
            rank=np.asarray(model["rank"], dtype=np.int16),
            condition_number=np.asarray(model["condition_number"]),
            effective_rank=np.asarray(model["effective_rank"]),
        )
        mark_complete(path, {"snapshot_key": record["key"], "created_utc": utc_now()})
    atomic_npz(
        os.path.join(output_root, "work", "analysis_parameters.npz"),
        consequence_scale=parameters["consequence_scale"],
        observed_consequence_center=parameters["observed_consequence_center"],
        covariance_mean=parameters["covariance_mean"],
        covariance_whitener=parameters["covariance_whitener"],
        covariance_dewhitener=parameters["covariance_dewhitener"],
        pca_mean=parameters["pca_mean"],
        pca_components=parameters["pca_components"],
        ridge=np.asarray(parameters["ridge"]),
        singular_cutoff=np.asarray(parameters["singular_cutoff"]),
        metric_regularization=np.asarray(parameters["metric_regularization"]),
    )


def prepare_analysis(output_root):
    records = load_branch_records(output_root)
    scaler = fit_consequence_scaler(records)
    ridge, ridge_trace = select_ridge(records, scaler["scale"])
    models = fit_all_jacobians(records, scaler["scale"], ridge)
    cutoff, cutoff_trace = select_singular_cutoff(records, models, scaler["scale"])
    attach_geometry(records, models, cutoff)
    train_samples = _sensitive_samples(records, models, "train")
    calibration_samples = _sensitive_samples(records, models, "calibration")
    projection = select_covariance_and_pca(train_samples, calibration_samples, models)
    metric_regularization, metric_trace = select_metric_regularization(records, models)
    covariance_mean, covariance_white, covariance_dewhite, covariance_spectrum = projection["covariance"]
    pca_mean, pca_components, pca_singular = projection["pca"]
    parameters = {
        "ridge": ridge,
        "singular_cutoff": cutoff,
        "metric_regularization": metric_regularization,
        "consequence_scale": scaler["scale"],
        "observed_consequence_center": scaler["observed_center"],
        "covariance_mean": covariance_mean,
        "covariance_whitener": covariance_white,
        "covariance_dewhitener": covariance_dewhite,
        "pca_mean": pca_mean,
        "pca_components": pca_components,
        "permutation_map": build_permutation_map(records),
    }
    selection = {
        "created_utc": utc_now(),
        "selection_split": "calibration_only",
        "ridge": ridge,
        "singular_cutoff": cutoff,
        "metric_regularization": metric_regularization,
        "covariance_regularization": projection["covariance_regularization"],
        "pca_rank": projection["pca_rank"],
        "ridge_trace": ridge_trace,
        "singular_cutoff_trace": cutoff_trace,
        "metric_regularization_trace": metric_trace,
        "covariance_trace": projection["covariance_trace"],
        "pca_trace": projection["pca_trace"],
        "train_sample_count": len(train_samples),
        "calibration_sample_count": len(calibration_samples),
    }
    atomic_json(os.path.join(output_root, "work", "model_selection.json"), selection)
    save_models(records, models, parameters, output_root)
    fit_codebooks(train_samples, models, parameters, output_root)
    rows = metric_correlations(records, models, parameters)
    export_jacobian_metrics(rows, output_root)
    consolidate_zarr(records, output_root)
    plans = write_quantization_plans(records, models, parameters, output_root)
    result = {
        "created_utc": utc_now(),
        "selection": selection,
        "jacobian_metric_rows": len(rows),
        "quantization_plans": len(plans),
        "status": "READY_FOR_REALIZED_QUANTIZATION_REPLAY",
    }
    atomic_json(os.path.join(output_root, "work", "analysis_prepare_manifest.json"), result)
    return result
