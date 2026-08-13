"""Stage 1.5 failure localization and residual/effect-alphabet rescue audit.

The module is deliberately split into pure analysis, simulator collection and
reporting commands.  Pure analysis runs in the Python 3.11 analysis
environment; collection runs in the frozen Python 3.8 LIBERO environment.
Stage 1 inputs are read-only throughout.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from . import config
from .analysis import load_branch_records
from .env_adapter import FEATURE_NAMES, LiberoTaskRuntime
from .math_utils import (
    covariance_whitener,
    kmeans,
    metric_effective_rank,
    r2_score,
    spearmanr,
)
from .pipeline import CONTACT_MODE_TO_ID, _pack_rollouts, utc_now
from .storage import (
    atomic_json,
    atomic_npz,
    atomic_text,
    mark_complete,
    sha256_file,
    sha256_tree,
    validate_complete,
)


PRIMARY_K = 64
EPS = 1e-12
METHODS_REVISED = (
    "M2_centered_covariance_residual",
    "M3_cara",
    "M4_reca",
    "M5_phase_residual_kmeans",
    "M6_permuted_j_reca",
    "M7_random_spd_constrained",
)
METHODS_DEPLOYABLE = METHODS_REVISED[:4]
METHOD_M0 = "M0_frozen_caaa_v2"
METHOD_M1 = "M1_covariance_mahalanobis"
ORACLE_O1 = "O1_true_effect_oracle"
ORACLE_O2 = "O2_linear_j_oracle"

CONSEQUENCE_GROUPS = {
    "object_pose": tuple(range(0, 9)) + tuple(range(27, 36)),
    "tcp_object_relative_pose": tuple(range(9, 27)),
    "contact_and_force": tuple(range(41, 44)),
    "gripper_and_articulation": tuple(range(36, 40)),
    "task_progress": (40,),
    "constraint_violations": (44, 45),
}

RECA_BETA_GRID = (1e-6, 1e-4, 1e-2, 1.0)
RECA_RADIUS_GRID = (0.10, 0.20, 0.40)
RECA_FEASIBILITY_QUANTILES = (0.90, 0.95, 0.99)


def _write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for name in row:
                if name not in seen:
                    fieldnames.append(name)
                    seen.add(name)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".incomplete"
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _stable_seed(label):
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:4], "little")


def _snapshot_key(task_id, episode_id, phase):
    return "%s__e%02d__%s" % (task_id, int(episode_id), phase)


def _scalar(array):
    value = np.asarray(array)
    return value.item() if value.shape == () else value.tolist()


def _candidate_rows(record, radius=0.10):
    return np.flatnonzero(np.isclose(np.asarray(record["radius"], dtype=np.float64), radius))


def _normalized_delta(record, scale, settled=True):
    values = np.asarray(record["settled"] if settled else record["immediate"], dtype=np.float64)
    delta = (values - values[[0]]) / np.asarray(scale, dtype=np.float64)[None, :]
    mask = np.asarray(record["mask"], dtype=bool) & np.asarray(record["mask"][[0]], dtype=bool)
    delta[~mask] = 0.0
    return delta


def _load_parameters(stage1_root):
    path = os.path.join(stage1_root, "work", "analysis_parameters.npz")
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]).copy() for name in data.files}


def _load_models(stage1_root, records):
    result = {}
    for record in records:
        path = os.path.join(stage1_root, "work", "jacobians", record["key"] + ".npz")
        with np.load(path, allow_pickle=False) as data:
            result[record["key"]] = {name: np.asarray(data[name]).copy() for name in data.files}
    return result


def _caaa_transform(model, regularization):
    return np.vstack(
        [
            np.asarray(model["j"], dtype=np.float64),
            math.sqrt(float(regularization)) * np.eye(config.CHUNK_CONTINUOUS_DIM),
        ]
    )


def _random_transform(record, model, regularization):
    singular = np.asarray(model["singular_values"], dtype=np.float64)
    spectrum = np.zeros(config.CHUNK_CONTINUOUS_DIM, dtype=np.float64)
    spectrum[: min(len(singular), len(spectrum))] = singular[: len(spectrum)]
    spectrum = np.sqrt(spectrum * spectrum + float(regularization))
    rng = np.random.RandomState(_stable_seed("stage1.5-random-spd:" + record["key"]))
    q, r = np.linalg.qr(rng.normal(size=(config.CHUNK_CONTINUOUS_DIM,) * 2))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q *= signs[None, :]
    return np.diag(spectrum).dot(q.T)


def build_permutation_map(records):
    grouped = {}
    for record in records:
        grouped.setdefault((record["task_id"], record["split"], record["phase"]), []).append(record)
    mapping = {}
    for values in grouped.values():
        values = sorted(values, key=lambda row: int(row["episode_id"]))
        for index, record in enumerate(values):
            mapping[record["key"]] = values[(index + 1) % len(values)]["key"]
    return mapping


def project_box_ball(value, lower, upper, radius):
    """Euclidean projection onto a coordinate box intersected with an L2 ball."""
    value = np.asarray(value, dtype=np.float64)
    return _project_box_ball_rows(value[None, :], lower, upper, radius)[0]


def _project_box_ball_rows(values, lower, upper, radius):
    """Vectorized exact projection for rows sharing the same box and radius."""
    values = np.asarray(values, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    clipped = np.clip(values, lower[None, :], upper[None, :])
    radius = float(radius)
    active = np.linalg.norm(clipped, axis=1) > radius + 1e-12
    if not np.any(active):
        return clipped
    selected = values[active]
    low = np.zeros(len(selected), dtype=np.float64)
    high = np.ones(len(selected), dtype=np.float64)
    for _ in range(60):
        candidate = np.clip(
            selected / (1.0 + high[:, None]), lower[None, :], upper[None, :]
        )
        expand = np.linalg.norm(candidate, axis=1) > radius
        if not np.any(expand):
            break
        high[expand] *= 2.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        candidate = np.clip(
            selected / (1.0 + middle[:, None]), lower[None, :], upper[None, :]
        )
        outside = np.linalg.norm(candidate, axis=1) > radius
        low[outside] = middle[outside]
        high[~outside] = middle[~outside]
    clipped[active] = np.clip(
        selected / (1.0 + high[:, None]), lower[None, :], upper[None, :]
    )
    return clipped


def solve_constrained_ridge(matrix, target, beta, base_action, radius, max_iter=300):
    """Solve the convex bounded/radius-constrained ridge decoder with FISTA."""
    result = _solve_constrained_ridge_batch(
        matrix,
        np.asarray(target, dtype=np.float64)[None, :],
        beta,
        base_action,
        radius,
        max_iter=max_iter,
    )
    return (
        result["decoded"][0],
        float(result["residual"][0]),
        float(result["objective"][0]),
        int(result["iterations"]),
    )


def _solve_constrained_ridge_batch(matrix, targets, beta, base_action, radius, max_iter=300):
    matrix = np.asarray(matrix, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    base_action = np.asarray(base_action, dtype=np.float64)
    beta = float(beta)
    lower = -1.0 - base_action
    upper = 1.0 - base_action
    ata = matrix.T.dot(matrix)
    target_times_matrix = targets.dot(matrix)
    lhs = ata + beta * np.eye(matrix.shape[1])
    try:
        initial = np.linalg.solve(lhs, target_times_matrix.T).T
    except np.linalg.LinAlgError:
        initial = target_times_matrix.dot(np.linalg.pinv(lhs, rcond=1e-12).T)
    spectral = float(np.linalg.norm(matrix, ord=2))
    lipschitz = max(2.0 * (spectral * spectral + beta), 1e-12)
    x = _project_box_ball_rows(initial, lower, upper, radius)
    y = x.copy()
    acceleration = 1.0
    iterations = 0
    for iterations in range(1, int(max_iter) + 1):
        gradient = 2.0 * (y.dot(ata) - target_times_matrix + beta * y)
        updated = _project_box_ball_rows(y - gradient / lipschitz, lower, upper, radius)
        movement = np.linalg.norm(updated - x, axis=1)
        reference = np.maximum(1.0, np.linalg.norm(x, axis=1))
        if float(np.max(movement / reference)) <= 1e-10:
            x = updated
            break
        next_acceleration = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * acceleration * acceleration))
        y = updated + ((acceleration - 1.0) / next_acceleration) * (updated - x)
        x = updated
        acceleration = next_acceleration
    error = x.dot(matrix.T) - targets
    residual = np.linalg.norm(error, axis=1)
    objective = residual * residual + beta * np.sum(x * x, axis=1)
    return {
        "decoded": x,
        "residual": residual,
        "objective": objective,
        "iterations": iterations,
    }


def decode_prototypes(matrix, prototypes, beta, base_action, radius):
    start = time.perf_counter()
    solved = _solve_constrained_ridge_batch(
        matrix, np.asarray(prototypes, dtype=np.float64), beta, base_action, radius
    )
    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    return {
        "decoded": solved["decoded"],
        "residual": solved["residual"],
        "objective": solved["objective"],
        "iterations": np.full(len(prototypes), solved["iterations"], dtype=np.int32),
        "latency_ms_per_token": elapsed_ms / float(max(len(prototypes), 1)),
    }


def _entropy_and_utilization(labels, capacity=PRIMARY_K):
    labels = np.asarray(labels, dtype=np.int64)
    labels = labels[labels >= 0]
    if labels.size == 0:
        return 0.0, 0.0, 1.0, 0.0
    counts = np.bincount(labels, minlength=int(capacity)).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    normalized_entropy = entropy / math.log(float(capacity)) if capacity > 1 else 1.0
    utilization = float(np.count_nonzero(counts)) / float(capacity)
    perplexity = math.exp(entropy) / float(capacity)
    return normalized_entropy, utilization, 1.0 - utilization, perplexity


def _nearest(value, centers, feasible=None):
    centers = np.asarray(centers, dtype=np.float64)
    distance = np.sum((centers - np.asarray(value)[None, :]) ** 2, axis=1)
    if feasible is not None:
        feasible = np.asarray(feasible, dtype=bool)
        if not np.any(feasible):
            return -1
        distance[~feasible] = np.inf
    return int(np.argmin(distance))


def _load_stage1_caaa_state_results(stage1_root, scale):
    results = {}
    pattern = os.path.join(stage1_root, "work", "quantized_shards", "*", "*.npz")
    for path in sorted(glob.glob(pattern)):
        with np.load(path, allow_pickle=False) as data:
            method = np.asarray(data["methods"]).astype(str)
            keep = (method == "caaa_v2") & (np.asarray(data["k"]) == PRIMARY_K)
            if not np.any(keep):
                continue
            task_id = str(_scalar(data["task_id"]))
            episode_id = int(_scalar(data["episode_id"]))
            phase = str(_scalar(data["phase"]))
            key = _snapshot_key(task_id, episode_id, phase)
            mask = np.asarray(data["original_mask"][keep], dtype=bool)
            settled = (
                np.asarray(data["settled"][keep], dtype=np.float64)
                - np.asarray(data["original_settled"][keep], dtype=np.float64)
            ) / scale[None, :]
            settled[~mask] = 0.0
            group_mse = {}
            for name, indices in CONSEQUENCE_GROUPS.items():
                values = settled[:, list(indices)]
                group_mse[name] = float(np.mean(np.sum(values * values, axis=1)))
            decoded = np.asarray(data["decoded_actions"][keep, :, :6], dtype=np.float64).reshape(-1, 24)
            original = np.asarray(data["original_actions"][keep, :, :6], dtype=np.float64).reshape(-1, 24)
            results[key] = {
                "realized_effect_error": float(np.mean(np.linalg.norm(settled, axis=1))),
                "action_reconstruction_error": float(np.mean(np.linalg.norm(decoded - original, axis=1) / math.sqrt(24.0))),
                "clipped_coordinate_fraction": float(np.sum(np.asarray(data["clipped_coordinates"][keep]))) / float(np.sum(keep) * 24),
                "labels": np.asarray(data["code_index"][keep], dtype=np.int64),
                "group_mse": group_mse,
            }
    return results


def retrospective_diagnostics(records, models, parameters, stage1_root, stage1_5_root):
    scale = np.asarray(parameters["consequence_scale"], dtype=np.float64)
    regularization = float(parameters["metric_regularization"])
    with np.load(
        os.path.join(stage1_root, "alphabet_codebooks", "caaa_v2_k64.npz"),
        allow_pickle=False,
    ) as data:
        caaa_centers = np.asarray(data["centers"], dtype=np.float64)
    realized = _load_stage1_caaa_state_results(stage1_root, scale)
    rows = []
    for record in records:
        model = models[record["key"]]
        j = np.asarray(model["j"], dtype=np.float64)
        dy = _normalized_delta(record, scale, settled=True)
        evaluate = _candidate_rows(record, 0.10)
        truth = dy[evaluate]
        prediction = np.asarray(record["delta_action"][evaluate], dtype=np.float64).dot(j.T)
        mask = np.broadcast_to(np.asarray(record["mask"][0], dtype=bool), truth.shape)
        denominator = max(float(np.mean(np.sum(truth * truth, axis=1))), EPS)
        nrmse = math.sqrt(float(np.mean(np.sum((prediction - truth) ** 2, axis=1))) / denominator)

        lookup = {}
        for index in range(1, len(record["radius"])):
            lookup[(int(record["direction"][index]), float(record["radius"][index]), int(record["sign"][index]))] = index
        antithetic, derivative_drift = [], []
        for direction in range(config.PERTURBATION_DIRECTIONS):
            for radius in config.PERTURBATION_RADII:
                minus = dy[lookup[(direction, float(radius), -1)]]
                plus = dy[lookup[(direction, float(radius), 1)]]
                antithetic.append(
                    float(np.linalg.norm(plus + minus))
                    / max(float(np.linalg.norm(plus) + np.linalg.norm(minus)), EPS)
                )
            for sign in config.PERTURBATION_SIGNS:
                small = dy[lookup[(direction, 0.05, int(sign))]] / 0.05
                large = dy[lookup[(direction, 0.10, int(sign))]] / 0.10
                derivative_drift.append(
                    float(np.linalg.norm(large - small))
                    / max(0.5 * float(np.linalg.norm(large) + np.linalg.norm(small)), EPS)
                )

        transform = _caaa_transform(model, regularization)
        transform_pinv = np.linalg.pinv(transform, rcond=1e-10)
        projector = np.asarray(model["projector"], dtype=np.float64)
        labels, residuals, preclip_norms, clipped = [], [], [], []
        for index in evaluate:
            action = np.asarray(record["action_cont"][index], dtype=np.float64)
            sensitive = projector.dot(action)
            null = action - sensitive
            label = _nearest(transform.dot(sensitive), caaa_centers)
            center = caaa_centers[label]
            decoded_sensitive = transform_pinv.dot(center)
            decoded = projector.dot(decoded_sensitive) + null
            clipped_value = np.clip(decoded, -1.0, 1.0)
            labels.append(label)
            residuals.append(
                float(np.linalg.norm(transform.dot(transform_pinv.dot(center)) - center))
                / max(float(np.linalg.norm(center)), EPS)
            )
            preclip_norms.append(float(np.linalg.norm(decoded)))
            clipped.append(float(np.mean(np.abs(decoded - clipped_value) > 1e-12)))
        entropy, utilization, dead, perplexity = _entropy_and_utilization(labels)
        singular = np.asarray(model["singular_values"], dtype=np.float64)
        retained = singular[singular >= float(parameters["singular_cutoff"]) * max(singular[0], EPS)]
        pinv_norm = 1.0 / max(float(retained[-1]), EPS) if len(retained) else float("inf")
        contact_base = int(record["contact_mode"][0])
        row = {
            "task_id": record["task_id"],
            "episode_id": int(record["episode_id"]),
            "split": record["split"],
            "phase": record["phase"],
            "snapshot_index": int(record["snapshot_index"]),
            "local_r2": float(r2_score(truth, prediction, mask=mask)),
            "local_normalized_rmse": float(nrmse),
            "local_predicted_to_realized_spearman": float(
                spearmanr(np.linalg.norm(prediction, axis=1), np.linalg.norm(truth, axis=1))
            ),
            "antithetic_nonlinearity_mean": float(np.mean(antithetic)),
            "antithetic_nonlinearity_max": float(np.max(antithetic)),
            "radius_derivative_drift_mean": float(np.mean(derivative_drift)),
            "radius_derivative_drift_max": float(np.max(derivative_drift)),
            "contact_mode_switch_rate": float(np.mean(np.asarray(record["contact_mode"])[evaluate] != contact_base)),
            "truncated_rank": int(_scalar(model["rank"])),
            "effective_rank": float(_scalar(model["effective_rank"])),
            "condition_number": float(_scalar(model["condition_number"])),
            "jacobian_singular_values_json": json.dumps([float(value) for value in singular]),
            "transform_singular_values_json": json.dumps([float(value) for value in np.linalg.svd(transform, compute_uv=False)]),
            "pseudoinverse_operator_norm": float(pinv_norm),
            "selected_center_reachable_residual_mean": float(np.mean(residuals)),
            "decoded_preclip_action_norm_mean": float(np.mean(preclip_norms)),
            "predicted_clipped_coordinate_fraction": float(np.mean(clipped)),
            "assignment_normalized_entropy": entropy,
            "assignment_utilization": utilization,
            "assignment_dead_code_ratio": dead,
            "assignment_normalized_perplexity": perplexity,
        }
        if record["key"] in realized:
            frozen = realized[record["key"]]
            row.update(
                {
                    "realized_effect_error": frozen["realized_effect_error"],
                    "realized_action_reconstruction_error": frozen["action_reconstruction_error"],
                    "realized_clipped_coordinate_fraction": frozen["clipped_coordinate_fraction"],
                }
            )
            for name, value in frozen["group_mse"].items():
                row["group_mse_" + name] = value
        else:
            row.update(
                {
                    "realized_effect_error": float("nan"),
                    "realized_action_reconstruction_error": float("nan"),
                    "realized_clipped_coordinate_fraction": float("nan"),
                }
            )
            for name in CONSEQUENCE_GROUPS:
                row["group_mse_" + name] = float("nan")
        rows.append(row)

    import pandas as pd

    destination = os.path.join(stage1_5_root, "retrospective_diagnostics.parquet")
    temporary = destination + ".incomplete"
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    decomposition = _error_decomposition(rows)
    _write_csv(os.path.join(stage1_5_root, "error_decomposition.csv"), decomposition)
    return rows, decomposition


def _error_decomposition(diagnostics):
    output = []
    available = [row for row in diagnostics if np.isfinite(row["realized_effect_error"])]
    scopes = [("pooled", available)]
    scopes.extend(
        ("task:" + task["task_id"], [row for row in available if row["task_id"] == task["task_id"]])
        for task in config.TASKS
    )
    scopes.extend(
        ("phase:" + phase, [row for row in available if row["phase"] == phase])
        for phase in config.PHASES
    )
    for scope, values in scopes:
        total = sum(float(np.mean([row["group_mse_" + name] for row in values])) for name in CONSEQUENCE_GROUPS)
        for name in CONSEQUENCE_GROUPS:
            mse = float(np.mean([row["group_mse_" + name] for row in values]))
            output.append(
                {
                    "row_type": "consequence_group_error",
                    "scope": scope,
                    "consequence_group": name,
                    "n_states": len(values),
                    "mean_squared_normalized_error": mse,
                    "share_of_total_squared_error": mse / max(total, EPS),
                    "regression_term": "",
                    "standardized_coefficient": "",
                    "regression_r2": "",
                }
            )

    feature_names = (
        "local_normalized_rmse",
        "selected_center_reachable_residual_mean",
        "pseudoinverse_operator_norm",
        "realized_clipped_coordinate_fraction",
        "assignment_dead_code_ratio",
    )
    matrix = np.asarray([[row[name] for name in feature_names] for row in available], dtype=np.float64)
    matrix = np.nan_to_num(matrix, nan=np.nanmedian(matrix, axis=0), posinf=0.0, neginf=0.0)
    means = np.mean(matrix, axis=0)
    standard = np.std(matrix, axis=0)
    standard[standard < 1e-12] = 1.0
    matrix = (matrix - means) / standard
    task_dummies = np.asarray(
        [[float(row["task_id"] == task["task_id"]) for task in config.TASKS[1:]] for row in available]
    )
    phase_dummies = np.asarray(
        [[float(row["phase"] == phase) for phase in config.PHASES[1:]] for row in available]
    )
    design = np.column_stack([np.ones(len(available)), matrix, task_dummies, phase_dummies])
    response = np.log1p(np.asarray([row["realized_effect_error"] for row in available], dtype=np.float64))
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    prediction = design.dot(coefficients)
    total = float(np.sum((response - response.mean()) ** 2))
    regression_r2 = 1.0 - float(np.sum((response - prediction) ** 2)) / max(total, EPS)
    terms = (
        ("local_model_error", coefficients[1]),
        ("center_infeasibility", coefficients[2]),
        ("inverse_amplification", coefficients[3]),
        ("clipping", coefficients[4]),
        ("codebook_collapse", coefficients[5]),
    )
    for name, coefficient in terms:
        output.append(
            {
                "row_type": "descriptive_standardized_regression",
                "scope": "pooled_with_task_and_phase_indicators",
                "consequence_group": "",
                "n_states": len(available),
                "mean_squared_normalized_error": "",
                "share_of_total_squared_error": "",
                "regression_term": name,
                "standardized_coefficient": float(coefficient),
                "regression_r2": regression_r2,
            }
        )
    return output


def _fit_codebooks(records, models, parameters):
    scale = np.asarray(parameters["consequence_scale"], dtype=np.float64)
    regularization = float(parameters["metric_regularization"])
    train = [record for record in records if record["split"] == "train"]
    delta_samples, cara_samples, effect_samples, random_samples = [], [], [], []
    phase_samples = {phase: [] for phase in config.PHASES}
    for record in train:
        rows = _candidate_rows(record)
        delta = np.asarray(record["delta_action"][rows], dtype=np.float64)
        dy = _normalized_delta(record, scale, settled=True)[rows]
        transform = _caaa_transform(models[record["key"]], regularization)
        random_transform = _random_transform(record, models[record["key"]], regularization)
        delta_samples.append(delta)
        cara_samples.append(delta.dot(transform.T))
        effect_samples.append(dy)
        random_samples.append(delta.dot(random_transform.T))
        phase_samples[record["phase"]].append(delta)
    delta_train = np.concatenate(delta_samples)
    covariance_mean, covariance_white, covariance_dewhite, covariance_spectrum = covariance_whitener(
        delta_train, float(parameters["covariance_regularization"])
    )
    covariance_z = (delta_train - covariance_mean).dot(covariance_white.T)
    m2_centers, _, m2_inertia = kmeans(covariance_z, PRIMARY_K, config.GLOBAL_SEED + 1502, max_iter=35)
    m3_centers, _, m3_inertia = kmeans(
        np.concatenate(cara_samples), PRIMARY_K, config.GLOBAL_SEED + 1503, max_iter=35
    )
    m4_centers, _, m4_inertia = kmeans(
        np.concatenate(effect_samples), PRIMARY_K, config.GLOBAL_SEED + 1504, max_iter=35
    )
    m5_centers, m5_inertia = [], []
    for phase_index, phase in enumerate(config.PHASES):
        centers, _, inertia = kmeans(
            np.concatenate(phase_samples[phase]),
            PRIMARY_K,
            config.GLOBAL_SEED + 1505 + phase_index,
            max_iter=35,
        )
        m5_centers.append(centers)
        m5_inertia.append(inertia)
    m7_centers, _, m7_inertia = kmeans(
        np.concatenate(random_samples), PRIMARY_K, config.GLOBAL_SEED + 1507, max_iter=35
    )
    return {
        "covariance_mean": covariance_mean,
        "covariance_whitener": covariance_white,
        "covariance_dewhitener": covariance_dewhite,
        "covariance_spectrum": covariance_spectrum,
        "m2_centers": m2_centers,
        "m3_centers": m3_centers,
        "m4_centers": m4_centers,
        "m5_centers": np.asarray(m5_centers),
        "m7_centers": m7_centers,
        "m2_inertia": np.asarray(m2_inertia),
        "m3_inertia": np.asarray(m3_inertia),
        "m4_inertia": np.asarray(m4_inertia),
        "m5_inertia": np.asarray(m5_inertia),
        "m7_inertia": np.asarray(m7_inertia),
    }


def _calibrate_reca(records, models, parameters, codebooks, permutation_map):
    scale = np.asarray(parameters["consequence_scale"], dtype=np.float64)
    calibration = [record for record in records if record["split"] == "calibration"]
    centers = np.asarray(codebooks["m4_centers"], dtype=np.float64)
    trace = []
    cache = {}
    for beta in RECA_BETA_GRID:
        for radius in RECA_RADIUS_GRID:
            residual_pool = []
            for record in calibration:
                base = np.asarray(record["base_actions"][:, :6], dtype=np.float64).reshape(-1)
                decoded = decode_prototypes(models[record["key"]]["j"], centers, beta, base, radius)
                cache[(record["key"], beta, radius)] = decoded
                residual_pool.extend(decoded["residual"].tolist())
            residual_pool = np.asarray(residual_pool, dtype=np.float64)
            for quantile in RECA_FEASIBILITY_QUANTILES:
                threshold = float(np.quantile(residual_pool, quantile))
                effect_error, action_error, missing = [], [], 0
                for record in calibration:
                    decoded = cache[(record["key"], beta, radius)]
                    feasible = decoded["residual"] <= threshold + 1e-12
                    rows = _candidate_rows(record)
                    target_effect = np.asarray(record["delta_action"][rows], dtype=np.float64).dot(
                        np.asarray(models[record["key"]]["j"], dtype=np.float64).T
                    )
                    truth = _normalized_delta(record, scale, settled=True)[rows]
                    for target_delta, target_prediction, target_truth in zip(
                        np.asarray(record["delta_action"][rows], dtype=np.float64), target_effect, truth
                    ):
                        label = _nearest(target_prediction, centers, feasible)
                        if label < 0:
                            missing += 1
                            value = np.zeros(24, dtype=np.float64)
                        else:
                            value = decoded["decoded"][label]
                        effect_error.append(
                            float(np.linalg.norm(models[record["key"]]["j"].dot(value) - target_truth))
                        )
                        action_error.append(float(np.linalg.norm(value - target_delta) / math.sqrt(24.0)))
                trace.append(
                    {
                        "beta": float(beta),
                        "radius_cap": float(radius),
                        "feasibility_quantile": float(quantile),
                        "feasibility_threshold": threshold,
                        "calibration_linear_effect_error": float(np.mean(effect_error)),
                        "calibration_action_reconstruction_error": float(np.mean(action_error)),
                        "infeasible_token_rate": float(np.mean(residual_pool > threshold)),
                        "states_without_feasible_token": int(missing),
                    }
                )
    selected = min(
        trace,
        key=lambda row: (
            row["calibration_linear_effect_error"],
            row["infeasible_token_rate"],
            row["calibration_action_reconstruction_error"],
            row["beta"],
            row["radius_cap"],
        ),
    )
    beta = selected["beta"]
    radius = selected["radius_cap"]
    quantile = selected["feasibility_quantile"]

    control_thresholds = {}
    for name in ("M4_reca", "M6_permuted_j_reca", "M7_random_spd_constrained"):
        residuals = []
        for record in calibration:
            base = np.asarray(record["base_actions"][:, :6], dtype=np.float64).reshape(-1)
            if name == "M4_reca":
                matrix = np.asarray(models[record["key"]]["j"], dtype=np.float64)
                prototypes = codebooks["m4_centers"]
            elif name == "M6_permuted_j_reca":
                matrix = np.asarray(models[permutation_map[record["key"]]]["j"], dtype=np.float64)
                prototypes = codebooks["m4_centers"]
            else:
                matrix = _random_transform(record, models[record["key"]], float(parameters["metric_regularization"]))
                prototypes = codebooks["m7_centers"]
            decoded = decode_prototypes(matrix, prototypes, beta, base, radius)
            residuals.extend(decoded["residual"].tolist())
        control_thresholds[name] = float(np.quantile(np.asarray(residuals), quantile))
    selected = dict(selected)
    selected["control_thresholds"] = control_thresholds
    return selected, trace


def _decode_state(record, models, parameters, codebooks, selection, permutation_map):
    base = np.asarray(record["base_actions"][:, :6], dtype=np.float64).reshape(-1)
    target_rows = _candidate_rows(record)
    target_delta = np.asarray(record["delta_action"][target_rows], dtype=np.float64)
    regularization = float(parameters["metric_regularization"])
    beta = float(selection["beta"])
    radius_cap = float(selection["radius_cap"])
    output = {}

    z = (target_delta - codebooks["covariance_mean"][None, :]).dot(
        codebooks["covariance_whitener"].T
    )
    labels = np.asarray([_nearest(value, codebooks["m2_centers"]) for value in z], dtype=np.int16)
    decoded = codebooks["covariance_mean"][None, :] + codebooks["m2_centers"][labels].dot(
        codebooks["covariance_dewhitener"].T
    )
    output[METHODS_REVISED[0]] = (labels, decoded, None, None, 0.0)

    transform = _caaa_transform(models[record["key"]], regularization)
    transformed = target_delta.dot(transform.T)
    labels = np.asarray([_nearest(value, codebooks["m3_centers"]) for value in transformed], dtype=np.int16)
    decoded = codebooks["m3_centers"][labels].dot(np.linalg.pinv(transform, rcond=1e-10).T)
    output[METHODS_REVISED[1]] = (labels, decoded, None, None, 0.0)

    j = np.asarray(models[record["key"]]["j"], dtype=np.float64)
    token = decode_prototypes(j, codebooks["m4_centers"], beta, base, radius_cap)
    threshold = float(selection["control_thresholds"]["M4_reca"])
    feasible = token["residual"] <= threshold + 1e-12
    predicted = target_delta.dot(j.T)
    labels = np.asarray([_nearest(value, codebooks["m4_centers"], feasible) for value in predicted], dtype=np.int16)
    decoded = np.asarray(
        [token["decoded"][label] if label >= 0 else np.zeros(24) for label in labels], dtype=np.float64
    )
    chosen_residual = np.asarray(
        [token["residual"][label] if label >= 0 else float("inf") for label in labels]
    )
    output[METHODS_REVISED[2]] = (
        labels,
        decoded,
        chosen_residual,
        float(np.mean(~feasible)),
        token["latency_ms_per_token"],
    )

    phase_centers = codebooks["m5_centers"][config.PHASES.index(record["phase"])]
    labels = np.asarray([_nearest(value, phase_centers) for value in target_delta], dtype=np.int16)
    output[METHODS_REVISED[3]] = (labels, phase_centers[labels], None, None, 0.0)

    permuted_j = np.asarray(models[permutation_map[record["key"]]]["j"], dtype=np.float64)
    token = decode_prototypes(permuted_j, codebooks["m4_centers"], beta, base, radius_cap)
    threshold = float(selection["control_thresholds"]["M6_permuted_j_reca"])
    feasible = token["residual"] <= threshold + 1e-12
    predicted = target_delta.dot(permuted_j.T)
    labels = np.asarray([_nearest(value, codebooks["m4_centers"], feasible) for value in predicted], dtype=np.int16)
    decoded = np.asarray(
        [token["decoded"][label] if label >= 0 else np.zeros(24) for label in labels], dtype=np.float64
    )
    chosen_residual = np.asarray(
        [token["residual"][label] if label >= 0 else float("inf") for label in labels]
    )
    output[METHODS_REVISED[4]] = (
        labels,
        decoded,
        chosen_residual,
        float(np.mean(~feasible)),
        token["latency_ms_per_token"],
    )

    random_transform = _random_transform(record, models[record["key"]], regularization)
    token = decode_prototypes(random_transform, codebooks["m7_centers"], beta, base, radius_cap)
    threshold = float(selection["control_thresholds"]["M7_random_spd_constrained"])
    feasible = token["residual"] <= threshold + 1e-12
    transformed = target_delta.dot(random_transform.T)
    labels = np.asarray([_nearest(value, codebooks["m7_centers"], feasible) for value in transformed], dtype=np.int16)
    decoded = np.asarray(
        [token["decoded"][label] if label >= 0 else np.zeros(24) for label in labels], dtype=np.float64
    )
    chosen_residual = np.asarray(
        [token["residual"][label] if label >= 0 else float("inf") for label in labels]
    )
    output[METHODS_REVISED[5]] = (
        labels,
        decoded,
        chosen_residual,
        float(np.mean(~feasible)),
        token["latency_ms_per_token"],
    )
    return target_rows, target_delta, output


def _write_old_test_plans(records, models, parameters, codebooks, selection, permutation_map, stage1_5_root):
    manifest = []
    for record in records:
        if record["split"] != "test":
            continue
        target_rows, target_delta, decoded_by_method = _decode_state(
            record, models, parameters, codebooks, selection, permutation_map
        )
        base = np.asarray(record["base_actions"][:, :6], dtype=np.float64).reshape(-1)
        methods, candidate_rows, code_indices = [], [], []
        decoded_actions, original_actions = [], []
        clipped_coordinates, preclip_norm = [], []
        feasibility_residual, infeasible_token_rate, solver_latency = [], [], []
        for method in METHODS_REVISED:
            labels, decoded_delta, residual, state_infeasible, latency = decoded_by_method[method]
            for offset, row in enumerate(target_rows):
                preclip = base + decoded_delta[offset]
                clipped = np.clip(preclip, -1.0, 1.0)
                full = np.asarray(record["base_actions"], dtype=np.float64).copy()
                full[:, :6] = clipped.reshape(config.CHUNK_HORIZON, 6)
                methods.append(method)
                candidate_rows.append(int(row))
                code_indices.append(int(labels[offset]))
                decoded_actions.append(full)
                original_actions.append(np.asarray(record["action_full"][row], dtype=np.float64))
                clipped_coordinates.append(int(np.sum(np.abs(preclip - clipped) > 1e-12)))
                preclip_norm.append(float(np.linalg.norm(preclip)))
                feasibility_residual.append(
                    float(residual[offset]) if residual is not None else float("nan")
                )
                infeasible_token_rate.append(
                    float(state_infeasible) if state_infeasible is not None else 0.0
                )
                solver_latency.append(float(latency))
        row_array = np.asarray(candidate_rows, dtype=np.int16)
        destination = os.path.join(
            stage1_5_root,
            "work",
            "old_test_plans",
            record["task_id"],
            record["key"] + ".npz",
        )
        atomic_npz(
            destination,
            evidence_set=np.asarray("old_test_internal_screen"),
            task_id=np.asarray(record["task_id"]),
            task_name=np.asarray(record["task_name"]),
            episode_id=np.asarray(record["episode_id"], dtype=np.int16),
            phase=np.asarray(record["phase"]),
            snapshot_index=np.asarray(record["snapshot_index"], dtype=np.int32),
            methods=np.asarray(methods),
            k=np.full(len(methods), PRIMARY_K, dtype=np.int16),
            candidate_row=row_array,
            direction=np.asarray(record["direction"][row_array], dtype=np.int16),
            sign=np.asarray(record["sign"][row_array], dtype=np.int8),
            radius=np.asarray(record["radius"][row_array], dtype=np.float64),
            code_index=np.asarray(code_indices, dtype=np.int16),
            clipped_coordinates=np.asarray(clipped_coordinates, dtype=np.int16),
            preclip_action_norm=np.asarray(preclip_norm, dtype=np.float64),
            feasibility_residual=np.asarray(feasibility_residual, dtype=np.float64),
            state_infeasible_token_rate=np.asarray(infeasible_token_rate, dtype=np.float64),
            solver_latency_ms_per_token=np.asarray(solver_latency, dtype=np.float64),
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
            {
                "created_utc": utc_now(),
                "snapshot_key": record["key"],
                "rows": len(methods),
                "methods": list(METHODS_REVISED),
            },
        )
        manifest.append({"path": destination, "marker": marker, "rows": len(methods)})
    atomic_json(
        os.path.join(stage1_5_root, "work", "old_test_plan_manifest.json"),
        {"created_utc": utc_now(), "count": len(manifest), "plans": manifest},
    )
    return manifest


def prepare_old_test(stage1_root, stage1_5_root):
    records = load_branch_records(stage1_root)
    models = _load_models(stage1_root, records)
    parameters = _load_parameters(stage1_root)
    with open(os.path.join(stage1_root, "work", "model_selection.json"), "r", encoding="utf-8") as handle:
        frozen_selection = json.load(handle)
    parameters["covariance_regularization"] = np.asarray(
        frozen_selection["covariance_regularization"], dtype=np.float64
    )
    diagnostics, decomposition = retrospective_diagnostics(
        records, models, parameters, stage1_root, stage1_5_root
    )
    codebooks = _fit_codebooks(records, models, parameters)
    permutation_map = build_permutation_map(records)
    selection, calibration_trace = _calibrate_reca(
        records, models, parameters, codebooks, permutation_map
    )
    codebook_path = os.path.join(stage1_5_root, "work", "stage1_5_codebooks.npz")
    atomic_npz(codebook_path, **codebooks)
    codebook_marker = mark_complete(
        codebook_path, {"created_utc": utc_now(), "primary_k": PRIMARY_K, "fit_split": "stage1_train"}
    )
    plans = _write_old_test_plans(
        records, models, parameters, codebooks, selection, permutation_map, stage1_5_root
    )
    method_definitions = {
        "created_utc": utc_now(),
        "schema_version": "r13-p15-stage1.5-methods-v1",
        "primary_k": PRIMARY_K,
        "k32_or_k128_inspected": False,
        "frozen_stage1_parameters": {
            "ridge": float(parameters["ridge"]),
            "singular_cutoff": float(parameters["singular_cutoff"]),
            "metric_regularization": float(parameters["metric_regularization"]),
            "covariance_regularization": float(parameters["covariance_regularization"]),
            "consequence_scale_sha256": hashlib.sha256(
                np.asarray(parameters["consequence_scale"], dtype=np.float64).tobytes()
            ).hexdigest(),
        },
        "matched_contract": {
            "target_rows": "all 48 signed radius-0.10 branches at every state",
            "train_episodes": list(config.TRAIN_EPISODES),
            "calibration_episodes": list(config.CALIBRATION_EPISODES),
            "old_test_episodes": list(config.TEST_EPISODES),
            "gripper": "copied unchanged",
            "action_bounds": [-1.0, 1.0],
            "consequence_schema": "frozen Stage 1 caaa-libero-consequence-v1",
        },
        "methods": {
            METHOD_M0: "byte-frozen Stage 1 CAAA-v2 K=64 realized rows",
            METHOD_M1: "byte-frozen Stage 1 covariance-Mahalanobis K=64 realized rows",
            METHODS_REVISED[0]: "global covariance-whitened K=64 k-means on train radius-0.10 residual actions; add state a0",
            METHODS_REVISED[1]: "global K=64 k-means on T_s delta_a; state pseudoinverse then add a0",
            METHODS_REVISED[2]: "global K=64 train realized-effect prototypes; bounded/radius-constrained local ridge decode",
            METHODS_REVISED[3]: "four phase-specific K=64 raw residual-action codebooks",
            METHODS_REVISED[4]: "M4 with Jacobians cyclically permuted within task/split/phase",
            METHODS_REVISED[5]: "matched-spectrum deterministic random transform with the M4 constrained decoder",
            ORACLE_O1: "same-state radius-0.05 dictionary selected by true settled consequence",
            ORACLE_O2: "same dictionary selected by frozen local linear J delta_a",
        },
        "constrained_solver": {
            "algorithm": "FISTA with exact Euclidean projection onto box intersect L2 ball",
            "max_iterations": 300,
            "beta_grid": list(RECA_BETA_GRID),
            "radius_grid": list(RECA_RADIUS_GRID),
            "feasibility_quantiles": list(RECA_FEASIBILITY_QUANTILES),
            "selection_split": "stage1_calibration_only",
            "selected": selection,
            "calibration_trace": calibration_trace,
        },
        "codebooks": {
            "path": "work/stage1_5_codebooks.npz",
            "sha256": sha256_file(codebook_path),
            "complete_marker": codebook_marker,
        },
        "internal_screen": {
            "confirmatory": False,
            "minimum_pooled_improvement_vs_m1": 0.05,
            "minimum_tasks_improved": 2,
            "minimum_clipping_reduction_vs_m0": 0.80,
            "maximum_action_reconstruction_degradation_vs_m1": 0.15,
            "control_reproduction_rule": "M6 and M7 may each retain at most 25% of a positive candidate gain",
        },
    }
    atomic_json(os.path.join(stage1_5_root, "method_definitions.json"), method_definitions)
    result = {
        "created_utc": utc_now(),
        "status": "READY_FOR_OLD_TEST_INTERNAL_SCREEN_COLLECTION",
        "retrospective_states": len(diagnostics),
        "decomposition_rows": len(decomposition),
        "plans": len(plans),
        "rows_planned": int(sum(row["rows"] for row in plans)),
        "method_definitions": os.path.join(stage1_5_root, "method_definitions.json"),
        "codebook_sha256": sha256_file(codebook_path),
    }
    atomic_json(os.path.join(stage1_5_root, "work", "prepare_manifest.json"), result)
    return result


def _plan_paths(stage1_5_root, task_id=None, evidence_set="old_test"):
    directory = "old_test_plans" if evidence_set == "old_test" else "fresh_plans"
    return sorted(
        glob.glob(
            os.path.join(
                stage1_5_root,
                "work",
                directory,
                task_id if task_id else "*",
                "*.npz",
            )
        )
    )


def collect_old_test(paths, stage1_5_root, task_id=None, plan_limit=None):
    tasks = [task for task in config.TASKS if task_id is None or task["task_id"] == task_id]
    if not tasks:
        raise KeyError(task_id)
    completed = []
    for task in tasks:
        runtime = LiberoTaskRuntime(task, paths["libero_source"], paths["dataset_root"])
        try:
            plans = _plan_paths(stage1_5_root, task["task_id"], "old_test")
            if plan_limit is not None:
                plans = plans[: int(plan_limit)]
            for plan_path in plans:
                destination = os.path.join(
                    stage1_5_root,
                    "work",
                    "old_test_quantized_shards",
                    task["task_id"],
                    os.path.basename(plan_path),
                )
                valid, evidence = validate_complete(destination)
                if valid:
                    completed.append({"path": destination, "status": "resumed", "evidence": evidence})
                    continue
                valid, evidence = validate_complete(plan_path)
                if not valid:
                    raise RuntimeError("invalid plan %s: %s" % (plan_path, evidence))
                with np.load(plan_path, allow_pickle=False) as plan:
                    episode_id = int(_scalar(plan["episode_id"]))
                    snapshot_index = int(_scalar(plan["snapshot_index"]))
                    decoded_actions = np.asarray(plan["decoded_actions"], dtype=np.float64)
                    copied = {
                        name: np.asarray(plan[name]).copy()
                        for name in plan.files
                        if name != "decoded_actions"
                    }
                episode = runtime.load_episode(episode_id)
                runtime.initialize_episode_model(episode)
                snapshot = runtime.snapshot_from_recorded_state(
                    episode["states"][snapshot_index], episode["actions"][:snapshot_index]
                )
                rollouts = []
                for row, action in enumerate(decoded_actions):
                    rollouts.append(runtime.execute_chunk(snapshot, action))
                    if (row + 1) % 96 == 0:
                        print(
                            "STAGE1_5_OLD_TEST_PROGRESS task=%s plan=%s rows=%d/%d"
                            % (task["task_id"], os.path.basename(plan_path), row + 1, len(decoded_actions)),
                            flush=True,
                        )
                packed = _pack_rollouts(rollouts)
                atomic_npz(destination, decoded_actions=decoded_actions, **copied, **packed)
                marker = mark_complete(
                    destination,
                    {
                        "created_utc": utc_now(),
                        "task_id": task["task_id"],
                        "episode_id": episode_id,
                        "snapshot_index": snapshot_index,
                        "rows": len(decoded_actions),
                        "source_plan": plan_path,
                        "identical_restore_per_row": True,
                        "renderer": False,
                        "offscreen_renderer": False,
                    },
                )
                completed.append({"path": destination, "marker": marker, "status": "created"})
                print(
                    "STAGE1_5_OLD_TEST_SHARD_COMPLETE task=%s plan=%s rows=%d"
                    % (task["task_id"], os.path.basename(plan_path), len(decoded_actions)),
                    flush=True,
                )
        finally:
            runtime.close()
    manifest = {
        "created_utc": utc_now(),
        "task_id": task_id,
        "count": len(completed),
        "shards": completed,
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
    }
    suffix = task_id or "all"
    atomic_json(
        os.path.join(stage1_5_root, "work", "old_test_collection_%s.json" % suffix), manifest
    )
    return completed


def _effect_rows_from_shards(paths, scale, evidence_set):
    rows = []
    for path in sorted(paths):
        valid, evidence = validate_complete(path)
        if not valid:
            raise RuntimeError("invalid realized shard %s: %s" % (path, evidence))
        with np.load(path, allow_pickle=False) as data:
            count = len(data["methods"])
            mask = np.asarray(data["original_mask"], dtype=bool)
            immediate_delta = (
                np.asarray(data["immediate"], dtype=np.float64)
                - np.asarray(data["original_immediate"], dtype=np.float64)
            ) / scale[None, :]
            settled_delta = (
                np.asarray(data["settled"], dtype=np.float64)
                - np.asarray(data["original_settled"], dtype=np.float64)
            ) / scale[None, :]
            immediate_delta[~mask] = 0.0
            settled_delta[~mask] = 0.0
            decoded = np.asarray(data["decoded_actions"][:, :, :6], dtype=np.float64).reshape(count, 24)
            original = np.asarray(data["original_actions"][:, :, :6], dtype=np.float64).reshape(count, 24)
            progress_error = np.abs(
                np.asarray(data["settled_progress"], dtype=np.float64)
                - np.asarray(data["original_settled_progress"], dtype=np.float64)
            )
            optional = {
                name: np.asarray(data[name]) if name in data.files else None
                for name in (
                    "state_infeasible_token_rate",
                    "feasibility_residual",
                    "solver_latency_ms_per_token",
                    "preclip_action_norm",
                )
            }
            task_id = str(_scalar(data["task_id"]))
            episode_id = int(_scalar(data["episode_id"]))
            phase = str(_scalar(data["phase"]))
            for index in range(count):
                method = str(data["methods"][index])
                if method == "caaa_v2":
                    method = METHOD_M0
                elif method == "covariance_mahalanobis":
                    method = METHOD_M1
                rows.append(
                    {
                        "evidence_set": evidence_set,
                        "task_id": task_id,
                        "episode_id": episode_id,
                        "phase": phase,
                        "method": method,
                        "k": int(data["k"][index]),
                        "direction": int(data["direction"][index]),
                        "sign": int(data["sign"][index]),
                        "radius": float(data["radius"][index]),
                        "code_index": int(data["code_index"][index]),
                        "settled_effect_error": float(np.linalg.norm(settled_delta[index])),
                        "immediate_effect_error": float(np.linalg.norm(immediate_delta[index])),
                        "action_reconstruction_error": float(
                            np.linalg.norm(decoded[index] - original[index]) / math.sqrt(24.0)
                        ),
                        "contact_preserved": bool(
                            int(data["contact_mode"][index])
                            == int(data["original_contact_mode"][index])
                        ),
                        "progress_absolute_error": float(progress_error[index]),
                        "progress_preserved_005": bool(progress_error[index] <= 0.05),
                        "success_preserved": bool(
                            int(data["settled_success"][index])
                            == int(data["original_settled_success"][index])
                        ),
                        "clipped_coordinates": int(data["clipped_coordinates"][index]),
                        "infeasible_assignment": bool(int(data["code_index"][index]) < 0),
                        "state_infeasible_token_rate": float(optional["state_infeasible_token_rate"][index])
                        if optional["state_infeasible_token_rate"] is not None
                        else 0.0,
                        "feasibility_residual": float(optional["feasibility_residual"][index])
                        if optional["feasibility_residual"] is not None
                        else float("nan"),
                        "solver_latency_ms": float(optional["solver_latency_ms_per_token"][index])
                        if optional["solver_latency_ms_per_token"] is not None
                        else 0.0,
                        "preclip_action_norm": float(optional["preclip_action_norm"][index])
                        if optional["preclip_action_norm"] is not None
                        else float(np.linalg.norm(decoded[index])),
                        "source_shard": path,
                    }
                )
    return rows


def _load_frozen_m0_m1_rows(stage1_root, scale):
    paths = sorted(glob.glob(os.path.join(stage1_root, "work", "quantized_shards", "*", "*.npz")))
    rows = _effect_rows_from_shards(paths, scale, "old_test_internal_screen")
    return [
        row
        for row in rows
        if row["episode_id"] in config.TEST_EPISODES
        and row["k"] == PRIMARY_K
        and row["method"] in (METHOD_M0, METHOD_M1)
    ]


def oracle_rows(records, models, scale, evidence_set="old_test_internal_screen"):
    output = []
    for record in records:
        if record["split"] != "test":
            continue
        dictionary_rows = _candidate_rows(record, 0.05)
        target_rows = _candidate_rows(record, 0.10)
        settled = _normalized_delta(record, scale, settled=True)
        immediate = _normalized_delta(record, scale, settled=False)
        dictionary_linear = np.asarray(record["delta_action"][dictionary_rows], dtype=np.float64).dot(
            np.asarray(models[record["key"]]["j"], dtype=np.float64).T
        )
        target_linear = np.asarray(record["delta_action"][target_rows], dtype=np.float64).dot(
            np.asarray(models[record["key"]]["j"], dtype=np.float64).T
        )
        for target_offset, target_row in enumerate(target_rows):
            true_distances = np.sum(
                (settled[dictionary_rows] - settled[target_row][None, :]) ** 2, axis=1
            )
            linear_distances = np.sum(
                (dictionary_linear - target_linear[target_offset][None, :]) ** 2, axis=1
            )
            for method, dictionary_offset in (
                (ORACLE_O1, int(np.argmin(true_distances))),
                (ORACLE_O2, int(np.argmin(linear_distances))),
            ):
                selected_row = int(dictionary_rows[dictionary_offset])
                settled_error = settled[selected_row] - settled[target_row]
                immediate_error = immediate[selected_row] - immediate[target_row]
                selected_action = np.asarray(record["action_cont"][selected_row], dtype=np.float64)
                target_action = np.asarray(record["action_cont"][target_row], dtype=np.float64)
                progress_error = abs(
                    float(record["settled_progress"][selected_row])
                    - float(record["settled_progress"][target_row])
                )
                output.append(
                    {
                        "evidence_set": evidence_set,
                        "task_id": record["task_id"],
                        "episode_id": int(record["episode_id"]),
                        "phase": record["phase"],
                        "method": method,
                        "k": 48,
                        "direction": int(record["direction"][target_row]),
                        "sign": int(record["sign"][target_row]),
                        "radius": float(record["radius"][target_row]),
                        "code_index": dictionary_offset,
                        "settled_effect_error": float(np.linalg.norm(settled_error)),
                        "immediate_effect_error": float(np.linalg.norm(immediate_error)),
                        "action_reconstruction_error": float(
                            np.linalg.norm(selected_action - target_action) / math.sqrt(24.0)
                        ),
                        "contact_preserved": bool(
                            int(record["contact_mode"][selected_row])
                            == int(record["contact_mode"][target_row])
                        ),
                        "progress_absolute_error": progress_error,
                        "progress_preserved_005": bool(progress_error <= 0.05),
                        "success_preserved": bool(
                            int(record["settled_success"][selected_row])
                            == int(record["settled_success"][target_row])
                        ),
                        "clipped_coordinates": 0,
                        "infeasible_assignment": False,
                        "state_infeasible_token_rate": 0.0,
                        "feasibility_residual": 0.0,
                        "solver_latency_ms": 0.0,
                        "preclip_action_norm": float(np.linalg.norm(selected_action)),
                        "source_shard": record["path"],
                    }
                )
    return output


def _group_rows(rows, fields):
    grouped = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        grouped.setdefault(key, []).append(row)
    return grouped


def aggregate_effect_rows(rows, group_fields):
    output = []
    for key, values in sorted(_group_rows(rows, group_fields).items()):
        row = dict(zip(group_fields, key))
        method = values[0]["method"]
        capacity = 48 if method in (ORACLE_O1, ORACLE_O2) else PRIMARY_K
        entropy, utilization, dead, perplexity = _entropy_and_utilization(
            [value["code_index"] for value in values], capacity
        )
        row.update(
            {
                "n": len(values),
                "settled_effect_error_mean": float(
                    np.mean([value["settled_effect_error"] for value in values])
                ),
                "immediate_effect_error_mean": float(
                    np.mean([value["immediate_effect_error"] for value in values])
                ),
                "contact_mode_preservation": float(
                    np.mean([value["contact_preserved"] for value in values])
                ),
                "task_progress_preservation_005": float(
                    np.mean([value["progress_preserved_005"] for value in values])
                ),
                "task_progress_mae": float(
                    np.mean([value["progress_absolute_error"] for value in values])
                ),
                "action_reconstruction_error_mean": float(
                    np.mean([value["action_reconstruction_error"] for value in values])
                ),
                "codebook_utilization": utilization,
                "assignment_normalized_entropy": entropy,
                "normalized_codebook_perplexity": perplexity,
                "dead_code_ratio": dead,
                "clipped_coordinate_rate": float(
                    np.sum([value["clipped_coordinates"] for value in values])
                )
                / float(len(values) * 24),
                "infeasible_assignment_rate": float(
                    np.mean([value["infeasible_assignment"] for value in values])
                ),
                "infeasible_token_rate": float(
                    np.mean([value["state_infeasible_token_rate"] for value in values])
                ),
                "solver_latency_ms_mean": float(
                    np.mean([value["solver_latency_ms"] for value in values])
                ),
            }
        )
        output.append(row)
    return output


def paired_episode_bootstrap(rows, baseline=METHOD_M1, replicates=10000):
    methods = sorted(set(row["method"] for row in rows))
    grouped = _group_rows(rows, ("task_id", "episode_id", "method"))
    episode_means = {
        key: float(np.mean([row["settled_effect_error"] for row in values]))
        for key, values in grouped.items()
    }
    tasks = [task["task_id"] for task in config.TASKS]
    episode_ids = {
        task: sorted(
            {
                int(episode_id)
                for task_id, episode_id, method in episode_means
                if task_id == task and method == baseline
            }
        )
        for task in tasks
    }
    rng = np.random.RandomState(config.GLOBAL_SEED + 1515)
    result = {
        "created_utc": utc_now(),
        "evidence_status": "retrospective_internal_screen_not_confirmatory",
        "baseline": baseline,
        "cluster_unit": "episode",
        "resampling": "paired within-task episode clusters with replacement",
        "replicates": int(replicates),
        "comparisons": {},
    }
    for method in methods:
        if method == baseline:
            continue
        complete = all(
            all((task, episode, method) in episode_means for episode in episode_ids[task])
            for task in tasks
        )
        if not complete:
            continue

        def statistic(task_subset, draws):
            baseline_values, method_values = [], []
            for task in task_subset:
                for episode in draws[task]:
                    baseline_values.append(episode_means[(task, int(episode), baseline)])
                    method_values.append(episode_means[(task, int(episode), method)])
            base = float(np.mean(baseline_values))
            candidate = float(np.mean(method_values))
            return (base - candidate) / max(base, EPS)

        identity = {task: episode_ids[task] for task in tasks}
        pooled_point = statistic(tasks, identity)
        pooled_boot = np.empty(int(replicates), dtype=np.float64)
        task_boot = {task: np.empty(int(replicates), dtype=np.float64) for task in tasks}
        for index in range(int(replicates)):
            draws = {
                task: rng.choice(episode_ids[task], size=len(episode_ids[task]), replace=True).tolist()
                for task in tasks
            }
            pooled_boot[index] = statistic(tasks, draws)
            for task in tasks:
                task_boot[task][index] = statistic([task], draws)
        result["comparisons"][method] = {
            "pooled_relative_improvement": {
                "estimate": pooled_point,
                "ci95": [
                    float(np.percentile(pooled_boot, 2.5)),
                    float(np.percentile(pooled_boot, 97.5)),
                ],
            },
            "per_task_relative_improvement": {
                task: {
                    "estimate": statistic([task], identity),
                    "ci95": [
                        float(np.percentile(task_boot[task], 2.5)),
                        float(np.percentile(task_boot[task], 97.5)),
                    ],
                }
                for task in tasks
            },
        }
    return result


def internal_screen(rows):
    pooled = aggregate_effect_rows(rows, ("evidence_set", "method"))
    by_method = {row["method"]: row for row in pooled}
    baseline = by_method[METHOD_M1]
    m0 = by_method[METHOD_M0]
    task_rows = aggregate_effect_rows(rows, ("evidence_set", "task_id", "method"))
    by_task = {(row["task_id"], row["method"]): row for row in task_rows}
    base_error = baseline["settled_effect_error_mean"]
    control_gain = {
        method: base_error - by_method[method]["settled_effect_error_mean"]
        for method in (METHODS_REVISED[4], METHODS_REVISED[5])
    }
    candidates = []
    for method in METHODS_DEPLOYABLE:
        value = by_method[method]
        gain = base_error - value["settled_effect_error_mean"]
        improvement = gain / max(base_error, EPS)
        tasks_improved = sum(
            by_task[(task["task_id"], method)]["settled_effect_error_mean"]
            < by_task[(task["task_id"], METHOD_M1)]["settled_effect_error_mean"]
            for task in config.TASKS
        )
        clipping_reduction = 1.0 - value["clipped_coordinate_rate"] / max(
            m0["clipped_coordinate_rate"], EPS
        )
        action_degradation = (
            value["action_reconstruction_error_mean"]
            - baseline["action_reconstruction_error_mean"]
        ) / max(baseline["action_reconstruction_error_mean"], EPS)
        retention = {
            control: max(0.0, control_gain[control]) / max(gain, EPS)
            if gain > 0
            else None
            for control in control_gain
        }
        control_not_reproduced = gain > 0 and all(
            item is not None and item <= 0.25 for item in retention.values()
        )
        gates = {
            "pooled_improvement_at_least_005": bool(improvement >= 0.05),
            "at_least_two_tasks_improve": bool(tasks_improved >= 2),
            "clipping_reduction_at_least_080": bool(clipping_reduction >= 0.80),
            "controls_do_not_reproduce": bool(control_not_reproduced),
            "action_reconstruction_degradation_at_most_015": bool(action_degradation <= 0.15),
        }
        candidates.append(
            {
                "method": method,
                "settled_effect_error": value["settled_effect_error_mean"],
                "relative_improvement_vs_m1": improvement,
                "tasks_improved": int(tasks_improved),
                "clipping_reduction_vs_m0": clipping_reduction,
                "action_reconstruction_degradation_vs_m1": action_degradation,
                "control_gain_retention": retention,
                "gates": gates,
                "passes": bool(all(gates.values())),
            }
        )
    passing = [row["method"] for row in candidates if row["passes"]]
    o1_improvement = (
        base_error - by_method[ORACLE_O1]["settled_effect_error_mean"]
    ) / max(base_error, EPS)
    return {
        "evidence_status": "retrospective_internal_screen_not_confirmatory",
        "baseline": METHOD_M1,
        "o1_relative_improvement_vs_m1": o1_improvement,
        "o1_upper_bound_gate_at_least_010": bool(o1_improvement >= 0.10),
        "candidates": candidates,
        "passing_methods": passing,
        "decision": "ADVANCE_TO_FRESH_HOLDOUT" if passing else "REJECT_P15_FAMILY",
    }


def _mechanism_control_rows(rows, screen):
    scopes = [("pooled", rows)]
    scopes.extend(
        (task["task_id"], [row for row in rows if row["task_id"] == task["task_id"]])
        for task in config.TASKS
    )
    output = []
    screen_lookup = {row["method"]: row for row in screen["candidates"]}
    for scope, values in scopes:
        aggregated = aggregate_effect_rows(values, ("method",))
        lookup = {row["method"]: row for row in aggregated}
        baseline_error = lookup[METHOD_M1]["settled_effect_error_mean"]
        reca_gain = baseline_error - lookup[METHODS_REVISED[2]]["settled_effect_error_mean"]
        for method in (
            METHOD_M0,
            METHODS_REVISED[0],
            METHODS_REVISED[1],
            METHODS_REVISED[2],
            METHODS_REVISED[3],
            METHODS_REVISED[4],
            METHODS_REVISED[5],
            ORACLE_O1,
            ORACLE_O2,
        ):
            error = lookup[method]["settled_effect_error_mean"]
            gain = baseline_error - error
            candidate = screen_lookup.get(method)
            output.append(
                {
                    "evidence_set": "old_test_internal_screen",
                    "scope": scope,
                    "method": method,
                    "baseline": METHOD_M1,
                    "baseline_error": baseline_error,
                    "method_error": error,
                    "relative_improvement": gain / max(baseline_error, EPS),
                    "gain_retention_vs_reca": gain / reca_gain if abs(reca_gain) > EPS else float("nan"),
                    "clipped_coordinate_rate": lookup[method]["clipped_coordinate_rate"],
                    "action_reconstruction_error": lookup[method]["action_reconstruction_error_mean"],
                    "infeasible_token_rate": lookup[method]["infeasible_token_rate"],
                    "internal_screen_pass": candidate["passes"] if candidate is not None else "",
                }
            )
    return output


def materialize_not_collected(stage1_5_root, reason):
    split = {
        "created_utc": utc_now(),
        "status": "NOT_COLLECTED_INTERNAL_SCREEN_FAILED",
        "reason": reason,
        "preferred_episode_ids": list(range(16, 24)),
        "records": [],
    }
    atomic_json(os.path.join(stage1_5_root, "fresh_holdout_split.json"), split)
    import zarr

    destination = os.path.join(stage1_5_root, "fresh_branch_rollouts.zarr")
    root = zarr.open_group(destination, mode="w")
    root.attrs.update(
        {
            "schema": "r13-p15-stage1.5-fresh-rollouts-v1",
            "status": "NOT_COLLECTED_INTERNAL_SCREEN_FAILED",
            "reason": reason,
            "created_utc": utc_now(),
            "states": 0,
        }
    )


def screen_old_test(stage1_root, stage1_5_root, replicates=10000):
    parameters = _load_parameters(stage1_root)
    scale = np.asarray(parameters["consequence_scale"], dtype=np.float64)
    records = load_branch_records(stage1_root)
    models = _load_models(stage1_root, records)
    revised_paths = sorted(
        glob.glob(os.path.join(stage1_5_root, "work", "old_test_quantized_shards", "*", "*.npz"))
    )
    if len(revised_paths) != len(config.TASKS) * len(config.TEST_EPISODES) * len(config.PHASES):
        raise RuntimeError("expected 64 revised old-test shards, found %d" % len(revised_paths))
    rows = _load_frozen_m0_m1_rows(stage1_root, scale)
    rows.extend(_effect_rows_from_shards(revised_paths, scale, "old_test_internal_screen"))
    rows.extend(oracle_rows(records, models, scale))
    expected = (2 + len(METHODS_REVISED) + 2) * 64 * 48
    if len(rows) != expected:
        raise RuntimeError("expected %d old-test rows, found %d" % (expected, len(rows)))

    task_results = aggregate_effect_rows(rows, ("evidence_set", "task_id", "method"))
    phase_results = aggregate_effect_rows(
        rows, ("evidence_set", "task_id", "phase", "method")
    )
    _write_csv(
        os.path.join(stage1_5_root, "quantization_results_by_task.csv"), task_results
    )
    _write_csv(
        os.path.join(stage1_5_root, "quantization_results_by_phase.csv"), phase_results
    )
    bootstrap = paired_episode_bootstrap(rows, replicates=int(replicates))
    atomic_json(os.path.join(stage1_5_root, "bootstrap_results.json"), bootstrap)
    screen = internal_screen(rows)
    controls = _mechanism_control_rows(rows, screen)
    _write_csv(os.path.join(stage1_5_root, "mechanism_controls.csv"), controls)
    screen.update(
        {
            "created_utc": utc_now(),
            "row_count": len(rows),
            "shard_count": len(revised_paths),
            "bootstrap_replicates": int(replicates),
        }
    )
    atomic_json(os.path.join(stage1_5_root, "work", "internal_screen.json"), screen)
    if screen["decision"] == "REJECT_P15_FAMILY":
        materialize_not_collected(stage1_5_root, "no revised deployable method passed every Part E gate")
    return screen


def _fmt(value, digits=6):
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "NA"
    if number == 0.0:
        return "0"
    if abs(number) >= 10000 or abs(number) < 1e-4:
        return ("%.*e" % (digits, number)).rstrip("0").rstrip(".")
    return ("%.*f" % (digits, number)).rstrip("0").rstrip(".")


def _markdown_table(headers, rows):
    rendered = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    rendered.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(rendered)


def _weighted_summary(frame, fields):
    result = {}
    weights = np.asarray(frame["n"], dtype=np.float64)
    for field in fields:
        result[field] = float(np.average(np.asarray(frame[field], dtype=np.float64), weights=weights))
    result["n"] = int(np.sum(weights))
    return result


def finalize_stage1_5(stage1_root, stage1_5_root):
    import pandas as pd

    with open(os.path.join(stage1_5_root, "work", "internal_screen.json"), "r", encoding="utf-8") as handle:
        screen = json.load(handle)
    if screen["decision"] != "REJECT_P15_FAMILY":
        raise RuntimeError("fresh-holdout path is required before finalization")
    with open(os.path.join(stage1_5_root, "bootstrap_results.json"), "r", encoding="utf-8") as handle:
        bootstrap = json.load(handle)
    with open(os.path.join(stage1_5_root, "STAGE1_INPUT_BINDING.json"), "r", encoding="utf-8") as handle:
        binding = json.load(handle)
    with open(os.path.join(stage1_5_root, "method_definitions.json"), "r", encoding="utf-8") as handle:
        definitions = json.load(handle)
    with open(os.path.join(stage1_5_root, "fresh_holdout_split.json"), "r", encoding="utf-8") as handle:
        fresh = json.load(handle)
    with open(os.path.join(stage1_5_root, "work", "development_runs.json"), "r", encoding="utf-8") as handle:
        development = json.load(handle)

    task = pd.read_csv(os.path.join(stage1_5_root, "quantization_results_by_task.csv"))
    phase = pd.read_csv(os.path.join(stage1_5_root, "quantization_results_by_phase.csv"))
    diagnostics = pd.read_parquet(os.path.join(stage1_5_root, "retrospective_diagnostics.parquet"))
    decomposition = pd.read_csv(os.path.join(stage1_5_root, "error_decomposition.csv"))
    controls = pd.read_csv(os.path.join(stage1_5_root, "mechanism_controls.csv"))

    metric_fields = (
        "settled_effect_error_mean",
        "immediate_effect_error_mean",
        "contact_mode_preservation",
        "task_progress_preservation_005",
        "action_reconstruction_error_mean",
        "clipped_coordinate_rate",
        "infeasible_token_rate",
        "normalized_codebook_perplexity",
        "dead_code_ratio",
    )
    pooled = {
        method: _weighted_summary(values, metric_fields)
        for method, values in task.groupby("method", sort=True)
    }
    phase_pooled = {
        (phase_name, method): _weighted_summary(values, metric_fields)
        for (phase_name, method), values in phase.groupby(["phase", "method"], sort=True)
    }
    task_lookup = {
        (str(row.task_id), str(row.method)): row
        for row in task.itertuples(index=False)
    }
    method_order = [
        METHOD_M0,
        METHOD_M1,
        METHODS_REVISED[0],
        METHODS_REVISED[1],
        METHODS_REVISED[2],
        METHODS_REVISED[3],
        METHODS_REVISED[4],
        METHODS_REVISED[5],
        ORACLE_O1,
        ORACLE_O2,
    ]
    short = {
        METHOD_M0: "M0 CAAA",
        METHOD_M1: "M1 covariance",
        METHODS_REVISED[0]: "M2 centered covariance",
        METHODS_REVISED[1]: "M3 CARA",
        METHODS_REVISED[2]: "M4 RECA",
        METHODS_REVISED[3]: "M5 phase residual",
        METHODS_REVISED[4]: "M6 permuted-J RECA",
        METHODS_REVISED[5]: "M7 random-SPD",
        ORACLE_O1: "O1 true-effect oracle",
        ORACLE_O2: "O2 linear-J oracle",
    }
    baseline_error = pooled[METHOD_M1]["settled_effect_error_mean"]

    diag_fields = (
        "local_r2",
        "local_normalized_rmse",
        "antithetic_nonlinearity_mean",
        "radius_derivative_drift_mean",
        "contact_mode_switch_rate",
        "effective_rank",
        "condition_number",
        "pseudoinverse_operator_norm",
        "selected_center_reachable_residual_mean",
        "realized_clipped_coordinate_fraction",
        "assignment_utilization",
    )
    diag_median = diagnostics[list(diag_fields)].median(numeric_only=True)
    diag_by_phase = diagnostics.groupby("phase")[list(diag_fields)].median(numeric_only=True)
    regression = decomposition[decomposition["row_type"] == "descriptive_standardized_regression"]
    group_error = decomposition[
        (decomposition["row_type"] == "consequence_group_error")
        & (decomposition["scope"] == "pooled")
    ]

    required = (
        "PREREGISTRATION.md",
        "STAGE1_INPUT_BINDING.json",
        "retrospective_diagnostics.parquet",
        "error_decomposition.csv",
        "fresh_holdout_split.json",
        "fresh_branch_rollouts.zarr",
        "method_definitions.json",
        "quantization_results_by_task.csv",
        "quantization_results_by_phase.csv",
        "mechanism_controls.csv",
        "bootstrap_results.json",
    )
    artifact_hashes = {}
    for name in required:
        path = os.path.join(stage1_5_root, name)
        artifact_hashes[name] = sha256_tree(path) if os.path.isdir(path) else sha256_file(path)
    atomic_json(
        os.path.join(stage1_5_root, "work", "artifact_hashes.json"),
        {
            "created_utc": utc_now(),
            "algorithm": "sha256_file or sorted relative-path tree hash",
            "artifacts": artifact_hashes,
        },
    )

    selected = definitions["constrained_solver"]["selected"]
    lines = []
    lines.extend(
        [
            "# R13-P15 Stage 1.5 Report — Failure Localization and Rescue Audit",
            "",
            "## Executive result",
            "",
            "Stage 1 remains rejected, and no Stage 1.5 revised deployable method passed the preregistered old-test internal screen. The stopping rule therefore prohibited collection of a fresh holdout. The exact Stage 1.5 disposition is given at the end of this report.",
            "",
            "The nominal old-test gains of M2 and M5 are not evidence for consequence geometry: both reconstruct the globally repeated deterministic perturbation actions to floating-point precision, while the permuted-J and random-SPD controls retain 52%–60% of those gains. For M4 RECA, permuted-J retains 114.4% of the gain and random-SPD retains 99.98%.",
            "",
            "This is a diagnostic-only experiment. No ACT, Diffusion Policy, SmolVLA, pi0.5, DINO-WM, behavior cloning, policy training or fresh-holdout collection was started.",
            "",
            "## Evidence boundary and preregistration",
            "",
            "- Preregistration/input-binding commit: `9a3ac1a4c774103fe618bd283909c2793ed581ec`.",
            "- Frozen-method/old-test-plan commit: `aa82d46c5e0828956aef15918c2aa7656844472f`.",
            "- `PREREGISTRATION.md` and `STAGE1_INPUT_BINDING.json` were committed before any revised-method result was computed or inspected.",
            "- Primary K was 64. K=32 and K=128 were not inspected.",
            "- Stage 1 test episodes 12–15 are retrospective/internal-screen evidence only, never confirmatory evidence for a revised method.",
            "- Simulation ran locally on CPU with `CUDA_VISIBLE_DEVICES` empty, `MUJOCO_GL=glx`, renderer and offscreen renderer disabled. Four tasks ran in task-level CPU parallelism; no GPU was used.",
            "",
            "## Frozen Stage 1 inputs and byte identity",
            "",
            "Stage 1 remains `REJECT_CORE_HYPOTHESIS`; nothing in `experiments/r13_p15_caaa_v2/stage1/` was modified.",
            "",
            _markdown_table(
                ["input", "value"],
                [
                    ["repository input commit", binding["repository"]["input_commit"]],
                    ["repository input tree", binding["repository"]["input_tree"]],
                    ["Stage 1 formal commit", binding["repository"]["stage1_formal_commit"]],
                    ["Stage 1 formal Git tree", binding["repository"]["stage1_formal_git_tree"]],
                    ["LIBERO commit", binding["simulator"]["libero_upstream_commit"]],
                    ["LIBERO source tree SHA-256", binding["simulator"]["libero_source_tree_sha256"]],
                    ["environment lock SHA-256", binding["simulator"]["environment_lock_sha256"]],
                    ["complete Stage 1 tree SHA-256", binding["stage1"]["directory_tree_sha256"]],
                    ["branch rollout tree SHA-256", binding["stage1"]["artifacts"]["branch_rollouts_zarr_tree_sha256"]],
                    ["Jacobian metrics SHA-256", binding["stage1"]["artifacts"]["jacobian_metrics_parquet_sha256"]],
                    ["codebook tree SHA-256", binding["stage1"]["artifacts"]["alphabet_codebooks_tree_sha256"]],
                    ["quantization JSONL SHA-256", binding["stage1"]["artifacts"]["quantization_results_jsonl_sha256"]],
                    ["quantized shard tree SHA-256", binding["stage1"]["artifacts"]["quantized_shards_tree_sha256"]],
                    ["Stage 1 report SHA-256", binding["stage1"]["report_sha256"]],
                ],
            ),
            "",
            "The pre-run Stage 1 release verifier passed all 256 replay tests, all 256 branch shards, 256 Jacobians, 128 quantization plans, 128 realized shards and all published hashes. The final repository verifier repeats the path-level Git identity check and bound-hash checks.",
            "",
            "## Why Stage 1 remains rejected",
            "",
            "The frozen Stage 1 CAAA result was 39.62% worse than covariance on pooled test error (frozen Stage 1 95% CI -168.41% to -5.50%). Stage 1 also showed severe action amplification, clipping and collapse. Stage 1.5 does not reinterpret that evidence.",
            "",
            "## Retrospective failure localization",
            "",
            "Across all 256 frozen states, the medians were:",
            "",
            _markdown_table(
                ["diagnostic", "median"],
                [[name, _fmt(diag_median[name])] for name in diag_fields],
            ),
            "",
            "Per-phase medians:",
            "",
            _markdown_table(
                ["phase", "R2", "NRMSE", "antithetic", "radius drift", "eff. rank", "condition", "center residual", "clip fraction"],
                [
                    [
                        phase_name,
                        _fmt(diag_by_phase.loc[phase_name, "local_r2"]),
                        _fmt(diag_by_phase.loc[phase_name, "local_normalized_rmse"]),
                        _fmt(diag_by_phase.loc[phase_name, "antithetic_nonlinearity_mean"]),
                        _fmt(diag_by_phase.loc[phase_name, "radius_derivative_drift_mean"]),
                        _fmt(diag_by_phase.loc[phase_name, "effective_rank"]),
                        _fmt(diag_by_phase.loc[phase_name, "condition_number"]),
                        _fmt(diag_by_phase.loc[phase_name, "selected_center_reachable_residual_mean"]),
                        _fmt(diag_by_phase.loc[phase_name, "realized_clipped_coordinate_fraction"]),
                    ]
                    for phase_name in config.PHASES
                ],
            ),
            "",
            "Free-space Jacobians were locally accurate (median R2 0.941, NRMSE 0.167), but contact-onset, pre-contact and post-contact were substantially nonlinear. The median effective rank was 1.79 and condition number 5,622. M0 assigned only one of 64 codes at the median state (utilization 0.015625) and clipped 63.20% of continuous coordinates when pooled over realized old-test rows.",
            "",
            "The frozen normalized metric was overwhelmingly driven by contact/force dimensions:",
            "",
            _markdown_table(
                ["consequence group", "mean squared normalized error", "share"],
                [
                    [row.consequence_group, _fmt(row.mean_squared_normalized_error), _fmt(row.share_of_total_squared_error)]
                    for row in group_error.itertuples(index=False)
                ],
            ),
            "",
            "The descriptive standardized regression (128 states with realized M0 rows, task and phase indicators; R2 " + _fmt(regression["regression_r2"].iloc[0]) + ") was:",
            "",
            _markdown_table(
                ["diagnostic term", "standardized coefficient"],
                [[row.regression_term, _fmt(row.standardized_coefficient)] for row in regression.itertuples(index=False)],
            ),
            "",
            "These associations are descriptive, not causal. Signs are unstable under strong collinearity—for example clipping has a negative conditional coefficient even though M0 clipping is severe—so the mechanism interventions below receive more weight than this regression.",
            "",
            "### Failure localization conclusion",
            "",
            "The dominant M0 implementation failure was applying local perturbation geometry to uncentered full actions, followed by inverse amplification, clipping and one-code collapse. Centering/raw residual methods remove that execution pathology. However, CARA does not rescue CAAA, and RECA is reproduced by geometry-destroying controls. Local-model error is also material in contact phases: O1 gains 96.63% over M1 while O2 gains only 52.01%. Prototype infeasibility is not dominant (M4 token infeasibility 1.03%, clipping 0%). Thus the Stage 1 failure is not a single fixable clipping bug; the consequence-J geometry lacks mechanism specificity under this audit.",
            "",
            "## Frozen methods and calibration",
            "",
            "All deployable methods used the same Stage 1 train/calibration episodes, K=64, target actions, consequence schema/scales, action bounds, snapshots, simulator semantics and unchanged gripper commands.",
            "",
        ]
    )
    for method in method_order:
        lines.append("- **%s:** %s" % (short[method], definitions["methods"][method]))
    lines.extend(
        [
            "",
            "RECA calibration selected beta=" + _fmt(selected["beta"]) + ", residual radius cap=" + _fmt(selected["radius_cap"]) + ", feasibility quantile=" + _fmt(selected["feasibility_quantile"]) + " and threshold=" + _fmt(selected["feasibility_threshold"]) + ". Selection used only episodes 8–11. The realized old-test execution comprised 64 states and 18,432 revised-method branches; every completion marker, plan binding and finite-value check passed.",
            "",
            "## Old-test internal-screen results",
            "",
            "Pooled results (all intervals below remain retrospective):",
            "",
        ]
    )
    pooled_rows = []
    for method in method_order:
        value = pooled[method]
        if method == METHOD_M1:
            improvement, ci = 0.0, [0.0, 0.0]
        else:
            comparison = bootstrap["comparisons"][method]["pooled_relative_improvement"]
            improvement, ci = comparison["estimate"], comparison["ci95"]
        pooled_rows.append(
            [
                short[method],
                _fmt(value["settled_effect_error_mean"]),
                _fmt(improvement),
                "[%s, %s]" % (_fmt(ci[0]), _fmt(ci[1])),
                _fmt(value["action_reconstruction_error_mean"]),
                _fmt(value["clipped_coordinate_rate"]),
                _fmt(value["contact_mode_preservation"]),
                _fmt(value["task_progress_preservation_005"]),
                _fmt(value["infeasible_token_rate"]),
            ]
        )
    lines.extend(
        [
            _markdown_table(
                ["method", "settled error", "rel. gain vs M1", "paired 95% CI", "action error", "clip", "contact", "progress", "infeasible"],
                pooled_rows,
            ),
            "",
            "Per-task settled error:",
            "",
            _markdown_table(
                ["task"] + [short[method] for method in method_order],
                [
                    [task_info["task_id"]]
                    + [_fmt(task_lookup[(task_info["task_id"], method)].settled_effect_error_mean) for method in method_order]
                    for task_info in config.TASKS
                ],
            ),
            "",
            "Pooled-across-task per-phase settled error:",
            "",
            _markdown_table(
                ["phase"] + [short[method] for method in method_order],
                [
                    [phase_name]
                    + [_fmt(phase_pooled[(phase_name, method)]["settled_effect_error_mean"]) for method in method_order]
                    for phase_name in config.PHASES
                ],
            ),
            "",
            "Candidate screen gates:",
            "",
            _markdown_table(
                ["candidate", "pooled gain", "tasks", "clip reduction", "action degradation", "M6 retention", "M7 retention", "pass"],
                [
                    [
                        short[row["method"]],
                        _fmt(row["relative_improvement_vs_m1"]),
                        row["tasks_improved"],
                        _fmt(row["clipping_reduction_vs_m0"]),
                        _fmt(row["action_reconstruction_degradation_vs_m1"]),
                        _fmt(row["control_gain_retention"][METHODS_REVISED[4]]),
                        _fmt(row["control_gain_retention"][METHODS_REVISED[5]]),
                        str(row["passes"]),
                    ]
                    for row in screen["candidates"]
                ],
            ),
            "",
            "No candidate passed. M2/M5 exploit the fact that the same 48 signed radius-0.10 perturbations appear in train and old test: their action errors are approximately 5.36e-17 and 1.07e-17. The remaining nonzero effect error despite nearly identical actions is concentrated in highly scaled contact/force channels, exposing numerical/contact sensitivity of this retrospective design rather than action-alphabet generalization.",
            "",
            "## Pooled and per-task confidence intervals",
            "",
            "All 10,000-replicate intervals use paired episode clusters resampled within task. They characterize the old-test internal screen only.",
            "",
        ]
    )
    ci_rows = []
    for method in method_order:
        if method == METHOD_M1:
            continue
        comparison = bootstrap["comparisons"][method]
        pooled_ci = comparison["pooled_relative_improvement"]
        row = [
            short[method],
            "%s [%s, %s]" % (
                _fmt(pooled_ci["estimate"]),
                _fmt(pooled_ci["ci95"][0]),
                _fmt(pooled_ci["ci95"][1]),
            ),
        ]
        for task_info in config.TASKS:
            value = comparison["per_task_relative_improvement"][task_info["task_id"]]
            row.append(
                "%s [%s, %s]" % (
                    _fmt(value["estimate"]),
                    _fmt(value["ci95"][0]),
                    _fmt(value["ci95"][1]),
                )
            )
        ci_rows.append(row)
    lines.extend(
        [
            _markdown_table(
                ["method", "pooled"] + [task["task_id"] for task in config.TASKS],
                ci_rows,
            ),
            "",
            "## Mechanism and oracle gaps",
            "",
            "- O1 versus M1: 96.63% lower error, establishing a local dictionary upper bound on this old perturbation support.",
            "- O1 versus O2: O1 error " + _fmt(pooled[ORACLE_O1]["settled_effect_error_mean"]) + " versus O2 " + _fmt(pooled[ORACLE_O2]["settled_effect_error_mean"]) + "; the 44.62 percentage-point gain gap identifies substantial local-model loss.",
            "- O2 versus M4: O2 error " + _fmt(pooled[ORACLE_O2]["settled_effect_error_mean"]) + " versus M4 " + _fmt(pooled[METHODS_REVISED[2]]["settled_effect_error_mean"]) + ", leaving a small global-prototype/decoder gap relative to the much larger model gap.",
            "- M3 versus M0: CARA reduces pooled clipping from " + _fmt(pooled[METHOD_M0]["clipped_coordinate_rate"]) + " to " + _fmt(pooled[METHODS_REVISED[1]]["clipped_coordinate_rate"]) + " but increases settled error from " + _fmt(pooled[METHOD_M0]["settled_effect_error_mean"]) + " to " + _fmt(pooled[METHODS_REVISED[1]]["settled_effect_error_mean"]) + ". Centering fixes clipping but not CAAA geometry.",
            "- M4 versus M6: permuted-J is better (" + _fmt(pooled[METHODS_REVISED[4]]["settled_effect_error_mean"]) + " versus " + _fmt(pooled[METHODS_REVISED[2]]["settled_effect_error_mean"]) + ") and retains 114.37% of the M4 gain.",
            "- M4 versus M7: random-SPD is effectively identical (" + _fmt(pooled[METHODS_REVISED[5]]["settled_effect_error_mean"]) + " versus " + _fmt(pooled[METHODS_REVISED[2]]["settled_effect_error_mean"]) + ") and retains 99.98% of the M4 gain.",
            "",
            "## Fresh holdout",
            "",
            "Fresh IDs: none. Preferred IDs 16–23 were never read, validated, selected or executed because the Part E internal screen failed. `fresh_holdout_split.json` and `fresh_branch_rollouts.zarr` both carry status `" + fresh["status"] + "` with zero records/states. This is a stopping manifest, not confirmatory data.",
            "",
            "## Failed and negative runs",
            "",
        ]
    )
    for run in development["runs"]:
        lines.append(
            "- `%s`: %s (exit %s). %s"
            % (
                run["command"],
                run["outcome"],
                run["exit_code"],
                run.get("reason", "Completed as specified."),
            )
        )
    lines.extend(
        [
            "- Simulator collection: 64/64 shards and 18,432/18,432 revised branches completed; zero marker, plan-binding or finite-value failures.",
            "- M3 CARA was negative: 124.24% worse than M1 and 60.61% worse than M0 despite reduced clipping.",
            "- M2, M4 and M5 were rejected at the mechanism-specificity gate; M6/M7 reproduced too much or all of their gains.",
            "- Fresh-holdout collection was intentionally not run; this is compliance with the stopping rule, not missing execution.",
            "",
            "## Artifact hashes",
            "",
            _markdown_table(
                ["artifact", "SHA-256"],
                [[name, artifact_hashes[name]] for name in required],
            ),
            "",
            "## Next permitted experiment",
            "",
            "Do not start policy training. If this line of inquiry is revisited, preregister a new evaluation with held-out perturbation directions or naturally varying demonstration residuals so train and test action supports are not identical; use an empirical nonlinear local-effect dictionary or contact-mode-stratified model, and add a non-force-dominated robustness metric only as a separately preregistered secondary analysis. That would be a new experiment, not Stage 1.5 continuation.",
            "",
            "## Final disposition",
            "",
            "FINAL_DISPOSITION: REJECT_P15_FAMILY",
            "",
        ]
    )
    report_path = os.path.join(stage1_5_root, "STAGE1_5_REPORT.md")
    atomic_text(report_path, "\n".join(lines))
    atomic_text(os.path.join(stage1_5_root, "work", "FINAL_DISPOSITION.txt"), "REJECT_P15_FAMILY\n")
    result = {
        "created_utc": utc_now(),
        "status": "STAGE1_5_COMPLETE",
        "disposition": "REJECT_P15_FAMILY",
        "fresh_holdout_status": fresh["status"],
        "internal_screen_rows": int(screen["row_count"]),
        "revised_realized_rows": 18432,
        "bootstrap_replicates": int(bootstrap["replicates"]),
        "report_sha256": sha256_file(report_path),
        "artifact_hashes": artifact_hashes,
    }
    atomic_json(os.path.join(stage1_5_root, "work", "finalize_manifest.json"), result)
    return result
