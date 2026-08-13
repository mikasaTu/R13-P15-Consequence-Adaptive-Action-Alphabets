"""Analysis, predictor fitting, gating, and reporting for preregistered Stage 2.

This module never imports LIBERO or creates a simulator.  It consumes only
completed branch shards produced by :mod:`caaa_libero.stage2`.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import os
from collections import defaultdict

import numpy as np

from .env_adapter import FEATURE_NAMES
from .math_utils import covariance_whitener, kmeans, ridge_jacobian, spearmanr
from .pipeline import utc_now
from .stage2 import _array_hash, _candidate_shard, _fps_indices, _support_shard
from .stage2_config import (
    ACTION_BANK_SIZE,
    ACTION_LATENT_DIM,
    CALIBRATION_EPISODES,
    CONTACT_CONFIDENCE_GRID,
    CONTACT_SENSITIVE_TASKS,
    DEVELOPMENT_EPISODES,
    GATES,
    GLOBAL_SEED,
    HUBER_CAP,
    HUBER_DELTA,
    LINEAR_NEIGHBOR_GRID,
    LINEAR_RIDGE_GRID,
    PHASES,
    PREDICTED_CONTINUOUS_INDICES,
    PREDICTOR_ARCHITECTURES,
    PREDICTOR_BATCH_SIZE,
    PREDICTOR_ENSEMBLE_SIZE,
    PREDICTOR_LEARNING_RATE,
    PREDICTOR_MAX_EPOCHS,
    PREDICTOR_MIN_DELTA,
    PREDICTOR_PATIENCE,
    PREDICTOR_WEIGHT_DECAY,
    PRIMARY_GROUPS,
    PRIMARY_K,
    SCALE_FLOORS,
    TASKS,
    TRAIN_EPISODES,
    UNCERTAINTY_COVERAGES,
)
from .storage import atomic_json, atomic_npz, atomic_text, validate_complete


TASK_IDS = tuple(task["task_id"] for task in TASKS)
TASK_TO_ID = {name: index for index, name in enumerate(TASK_IDS)}
PHASE_TO_ID = {name: index for index, name in enumerate(PHASES)}
CONTINUOUS_INDICES = np.asarray(PREDICTED_CONTINUOUS_INDICES, dtype=np.int64)
CONTACT_MODE_COUNT = 4


def _seed(*parts):
    digest = hashlib.sha256("|".join(str(x) for x in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _load_npz_checked(path):
    valid, evidence = validate_complete(path)
    if not valid:
        raise RuntimeError("incomplete shard %s: %s" % (path, evidence))
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]).copy() for name in data.files}


def load_state_records(output_root, splits):
    """Load aligned target-support and common-bank effects for requested splits."""
    records = []
    with open(os.path.join(output_root, "development_split.json"), "r", encoding="utf-8") as handle:
        frozen = json.load(handle)["records"]
    if "confirmation" in splits:
        with open(os.path.join(output_root, "confirmation_split.json"), "r", encoding="utf-8") as handle:
            frozen += json.load(handle)["records"]
    for meta in frozen:
        if meta["split"] not in splits:
            continue
        support_path = _support_shard(
            output_root, meta["split"], meta["task_id"], meta["episode_id"], meta["phase"]
        )
        support = _load_npz_checked(support_path)
        record = {"meta": meta, "support": support, "support_path": support_path}
        if meta["split"] != "train":
            candidate_path = _candidate_shard(
                output_root, meta["split"], meta["task_id"], meta["episode_id"], meta["phase"]
            )
            record["candidate"] = _load_npz_checked(candidate_path)
            record["candidate_path"] = candidate_path
            if not np.array_equal(record["candidate"]["bank_index"][1:], np.arange(ACTION_BANK_SIZE)):
                raise RuntimeError("Stage 2 expected all frozen bank members to be valid")
        records.append(record)
    records.sort(
        key=lambda row: (
            TASK_TO_ID[row["meta"]["task_id"]],
            row["meta"]["episode_id"],
            PHASE_TO_ID[row["meta"]["phase"]],
        )
    )
    return records


def _effect(shard):
    return np.asarray(shard["settled"] - shard["settled"][[0]], dtype=np.float64)


def _feature_floor(index):
    name = FEATURE_NAMES[int(index)]
    if "pos_" in name:
        return SCALE_FLOORS["position"]
    if "rot6" in name:
        return SCALE_FLOORS["rotation6d"]
    if name == "gripper_width":
        return SCALE_FLOORS["gripper_width"]
    if name.startswith("articulated"):
        return SCALE_FLOORS["articulation"]
    if name == "task_progress":
        return SCALE_FLOORS["task_progress"]
    if name.startswith("contact_force"):
        return SCALE_FLOORS["log_contact_force"]
    if name == "max_penetration":
        return SCALE_FLOORS["penetration"]
    if name == "joint_limit_violation":
        return SCALE_FLOORS["joint_violation"]
    raise KeyError(name)


def fit_train_scaling(records):
    rows, masks = [], []
    for record in records:
        values = _effect(record["support"])[1:]
        rows.append(values)
        masks.append(np.asarray(record["support"]["mask"][1:], dtype=bool))
    values = np.concatenate(rows, axis=0)
    masks = np.concatenate(masks, axis=0)
    scale = np.ones(len(FEATURE_NAMES), dtype=np.float64)
    evidence = []
    for index in range(len(FEATURE_NAMES)):
        observed = values[masks[:, index], index]
        floor = _feature_floor(index)
        if len(observed):
            median = float(np.median(observed))
            mad = 1.4826 * float(np.median(np.abs(observed - median)))
            q25, q75 = np.percentile(observed, [25.0, 75.0])
            iqr = float(q75 - q25) / 1.349
            selected = max(mad, iqr, floor)
        else:
            median, mad, iqr, selected = 0.0, 0.0, 0.0, floor
        scale[index] = selected
        evidence.append(
            {
                "index": index,
                "feature": FEATURE_NAMES[index],
                "observations": int(len(observed)),
                "median": median,
                "mad_scale": mad,
                "iqr_scale": iqr,
                "floor": floor,
                "selected_scale": selected,
            }
        )
    return scale, evidence


def _huber(values):
    absolute = np.minimum(np.abs(np.asarray(values, dtype=np.float64)), HUBER_CAP)
    return np.where(
        absolute <= HUBER_DELTA,
        0.5 * absolute * absolute,
        HUBER_DELTA * (absolute - 0.5 * HUBER_DELTA),
    )


def balanced_group_errors(target, decoded, target_mask, decoded_mask, target_mode, decoded_mode, scale):
    """Return vectorized errors for each preregistered equal-weight group."""
    target = np.asarray(target, dtype=np.float64)
    decoded = np.asarray(decoded, dtype=np.float64)
    target_mask = np.asarray(target_mask, dtype=bool)
    decoded_mask = np.asarray(decoded_mask, dtype=bool)
    if target.ndim == 1:
        target = target[None, :]
        decoded = decoded[None, :]
        target_mask = target_mask[None, :]
        decoded_mask = decoded_mask[None, :]
    target_mode = np.broadcast_to(np.asarray(target_mode).reshape(-1), (len(target),))
    decoded_mode = np.broadcast_to(np.asarray(decoded_mode).reshape(-1), (len(target),))
    group_values = {}
    for group, indices in PRIMARY_GROUPS.items():
        indices = np.asarray(indices, dtype=np.int64)
        active = target_mask[:, indices] & decoded_mask[:, indices]
        normalized = (target[:, indices] - decoded[:, indices]) / scale[indices][None, :]
        losses = _huber(normalized)
        numerator = np.sum(losses * active, axis=1)
        denominator = np.maximum(np.sum(active, axis=1), 1)
        value = numerator / denominator
        if group == "contact_mode_and_penetration":
            mismatch = (target_mode != decoded_mode).astype(np.float64)
            active_penetration = np.any(active, axis=1).astype(np.float64)
            value = (value * active_penetration + mismatch) / (active_penetration + 1.0)
        group_values[group] = value
    return group_values


def balanced_error(target, decoded, target_mask, decoded_mask, target_mode, decoded_mode, scale):
    """Vectorized target-vs-decoded BALANCED_TASK_EFFECT error."""
    groups = balanced_group_errors(
        target, decoded, target_mask, decoded_mask, target_mode, decoded_mode, scale
    )
    return np.mean(np.stack([groups[name] for name in PRIMARY_GROUPS], axis=1), axis=1)


def effect_embedding(effect, mask, mode, scale):
    """Equal-group Euclidean embedding used only by deterministic atlas FPS."""
    effect = np.asarray(effect, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if effect.ndim == 1:
        effect = effect[None, :]
        mask = mask[None, :]
    blocks = []
    for group, indices in PRIMARY_GROUPS.items():
        indices = np.asarray(indices, dtype=np.int64)
        block = np.clip(effect[:, indices] / scale[indices][None, :], -HUBER_CAP, HUBER_CAP)
        block = block * mask[:, indices]
        block /= math.sqrt(max(len(indices), 1) * len(PRIMARY_GROUPS))
        blocks.append(block)
        if group == "contact_mode_and_penetration":
            one_hot = np.eye(CONTACT_MODE_COUNT)[np.asarray(mode, dtype=np.int64)]
            blocks.append(one_hot / math.sqrt(CONTACT_MODE_COUNT * len(PRIMARY_GROUPS)))
    return np.concatenate(blocks, axis=1)


def _fps_tie_stable(values, k):
    values = np.asarray(values, dtype=np.float64)
    chosen = [0]
    minimum = np.sum((values - values[0]) ** 2, axis=1)
    minimum[0] = -1.0
    while len(chosen) < int(k):
        index = int(np.argmax(minimum))
        chosen.append(index)
        distance = np.sum((values - values[index]) ** 2, axis=1)
        minimum = np.minimum(minimum, distance)
        minimum[np.asarray(chosen, dtype=np.int64)] = -1.0
    return np.asarray(chosen, dtype=np.int64)


def _nearest_rows(target, centers):
    distance = np.sum((target[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(distance, axis=1)


def _kmeans_medoids(values, k, seed):
    """Map deterministic k-means centers to unique executable bank members."""
    values = np.asarray(values, dtype=np.float64)
    centers, _, _ = kmeans(values, k, seed)
    distances = np.sum((centers[:, None, :] - values[None, :, :]) ** 2, axis=2)
    selected = []
    for center_id in range(len(centers)):
        for index in np.argsort(distances[center_id], kind="mergesort"):
            if int(index) not in selected:
                selected.append(int(index))
                break
    if len(selected) < int(k):
        minimum = np.min(
            np.sum((values[:, None, :] - values[np.asarray(selected)][None, :, :]) ** 2, axis=2),
            axis=1,
        )
        minimum[np.asarray(selected, dtype=np.int64)] = -1.0
        while len(selected) < int(k):
            index = int(np.argmax(minimum))
            selected.append(index)
            minimum = np.minimum(minimum, np.sum((values - values[index]) ** 2, axis=1))
            minimum[np.asarray(selected, dtype=np.int64)] = -1.0
    return np.asarray(selected, dtype=np.int64)


def build_action_baseline_codebooks(output_root):
    with np.load(os.path.join(output_root, "action_bank.npz"), allow_pickle=False) as data:
        residuals = np.asarray(data["residuals"], dtype=np.float64)
        phases = data["source_phase"].astype(str)
    _, whitening, _, eigenvalues = covariance_whitener(residuals, regularization=1e-6)
    whitened = residuals.dot(whitening.T)
    b1 = _kmeans_medoids(whitened, PRIMARY_K, _seed(GLOBAL_SEED, "B1_kmeans"))
    b2 = {}
    for phase in PHASES:
        pool = np.flatnonzero(phases == phase)
        chosen_local = _kmeans_medoids(
            residuals[pool], min(PRIMARY_K, len(pool)), _seed(GLOBAL_SEED, "B2_kmeans", phase)
        )
        b2[phase] = pool[chosen_local]
    b3 = _fps_tie_stable(residuals, PRIMARY_K)
    return {
        "residuals": residuals,
        "whitening": whitening,
        "covariance_eigenvalues": eigenvalues,
        "B1": b1,
        "B2": b2,
        "B3": b3,
    }


def _action_assignment(target_residuals, bank_residuals, code_indices, transform=None):
    target = np.asarray(target_residuals, dtype=np.float64)
    bank = np.asarray(bank_residuals[code_indices], dtype=np.float64)
    if transform is not None:
        target = target.dot(transform.T)
        bank = bank.dot(transform.T)
    return np.asarray(code_indices, dtype=np.int64)[_nearest_rows(target, bank)]


def _evaluate_decoded(record, decoded_bank_indices, method, scale, extra=None):
    target_effect = _effect(record["support"])[1:]
    target_mask = np.asarray(record["support"]["mask"][1:], dtype=bool)
    target_mode = np.asarray(record["support"]["contact_mode"][1:], dtype=np.int64)
    candidate_effect = _effect(record["candidate"])[1:]
    candidate_mask = np.asarray(record["candidate"]["mask"][1:], dtype=bool)
    candidate_mode = np.asarray(record["candidate"]["contact_mode"][1:], dtype=np.int64)
    decoded = np.asarray(decoded_bank_indices, dtype=np.int64)
    group_errors = balanced_group_errors(
        target_effect,
        candidate_effect[decoded],
        target_mask,
        candidate_mask[decoded],
        target_mode,
        candidate_mode[decoded],
        scale,
    )
    error = np.mean(np.stack([group_errors[name] for name in PRIMARY_GROUPS], axis=1), axis=1)
    target_action = np.asarray(record["support"]["residual_action"][1:], dtype=np.float64)
    decoded_action = np.asarray(record["candidate"]["residual_action"][1:][decoded], dtype=np.float64)
    rows = []
    for index in range(len(error)):
        row = {
            "task_id": record["meta"]["task_id"],
            "episode_id": int(record["meta"]["episode_id"]),
            "split": record["meta"]["split"],
            "phase": record["meta"]["phase"],
            "target_id": int(index),
            "direction_id": int(record["support"]["direction_id"][index + 1]),
            "direction_family_id": int(record["support"]["direction_family_id"][index + 1]),
            "radius": float(record["support"]["radius"][index + 1]),
            "sign": int(record["support"]["sign"][index + 1]),
            "method": method,
            "decoded_bank_index": int(decoded[index]),
            "balanced_task_effect_error": float(error[index]),
            "action_reconstruction_rmse": float(np.sqrt(np.mean((target_action[index] - decoded_action[index]) ** 2))),
            "contact_mode_preserved": int(target_mode[index] == candidate_mode[decoded[index]]),
            "task_progress_abs_error": float(abs(target_effect[index, 40] - candidate_effect[decoded[index], 40])),
            "contact_force_effect_error": float(np.mean(np.abs(target_effect[index, 41:44] - candidate_effect[decoded[index], 41:44]))),
            "target_mode": int(target_mode[index]),
            "decoded_mode": int(candidate_mode[decoded[index]]),
            "clipped": 0,
        }
        for group in PRIMARY_GROUPS:
            row["error_group_" + group] = float(group_errors[group][index])
        if extra:
            for name, values in extra.items():
                row[name] = values[index] if np.ndim(values) else values
        rows.append(row)
    return rows


def evaluate_action_baselines(records, codebooks, scale):
    rows = []
    bank = codebooks["residuals"]
    for record in records:
        target = np.asarray(record["support"]["residual_action"][1:], dtype=np.float64)
        choices = {
            "B1_centered_covariance": _action_assignment(
                target, bank, codebooks["B1"], codebooks["whitening"]
            ),
            "B2_phase_residual": _action_assignment(
                target, bank, codebooks["B2"][record["meta"]["phase"]]
            ),
            "B3_dynamic_action_medoids": _action_assignment(target, bank, codebooks["B3"]),
        }
        for method, decoded in choices.items():
            rows.extend(_evaluate_decoded(record, decoded, method, scale))
    return rows


def evaluate_continuous_upper_bound(records):
    rows = []
    for record in records:
        for index in range(len(record["support"]["residual_action"]) - 1):
            row = {
                "task_id": record["meta"]["task_id"],
                "episode_id": int(record["meta"]["episode_id"]),
                "split": record["meta"]["split"],
                "phase": record["meta"]["phase"],
                "target_id": index,
                "direction_id": int(record["support"]["direction_id"][index + 1]),
                "direction_family_id": int(record["support"]["direction_family_id"][index + 1]),
                "radius": float(record["support"]["radius"][index + 1]),
                "sign": int(record["support"]["sign"][index + 1]),
                "method": "B0_continuous_target",
                "decoded_bank_index": -1,
                "balanced_task_effect_error": 0.0,
                "action_reconstruction_rmse": 0.0,
                "contact_mode_preserved": 1,
                "task_progress_abs_error": 0.0,
                "contact_force_effect_error": 0.0,
                "target_mode": int(record["support"]["contact_mode"][index + 1]),
                "decoded_mode": int(record["support"]["contact_mode"][index + 1]),
                "clipped": 0,
            }
            for group in PRIMARY_GROUPS:
                row["error_group_" + group] = 0.0
            rows.append(row)
    return rows


def evaluate_true_oracle(records, scale):
    rows = []
    for record in records:
        candidate_effect = _effect(record["candidate"])[1:]
        candidate_mask = np.asarray(record["candidate"]["mask"][1:], dtype=bool)
        candidate_mode = np.asarray(record["candidate"]["contact_mode"][1:], dtype=np.int64)
        target_effect = _effect(record["support"])[1:]
        target_mask = np.asarray(record["support"]["mask"][1:], dtype=bool)
        target_mode = np.asarray(record["support"]["contact_mode"][1:], dtype=np.int64)
        embedded_candidates = effect_embedding(candidate_effect, candidate_mask, candidate_mode, scale)
        embedded_target = effect_embedding(target_effect, target_mask, target_mode, scale)
        atlas = _fps_tie_stable(embedded_candidates, PRIMARY_K)
        decoded = atlas[_nearest_rows(embedded_target, embedded_candidates[atlas])]
        rows.extend(
            _evaluate_decoded(
                record,
                decoded,
                "O1_true_effect_oracle",
                scale,
                extra={"atlas_size": PRIMARY_K, "atlas_unique": len(np.unique(atlas))},
            )
        )
    return rows


def _state_vector(record):
    initial = np.asarray(record["support"]["initial"][0], dtype=np.float64)
    mask = np.asarray(record["support"]["mask"][0], dtype=np.float64)
    task = np.eye(len(TASK_IDS))[TASK_TO_ID[record["meta"]["task_id"]]]
    phase = np.eye(len(PHASES))[PHASE_TO_ID[record["meta"]["phase"]]]
    current_contact = np.asarray([float(np.max(initial[41:44]) > math.log1p(1e-6))])
    return np.concatenate((initial, mask, task, phase, current_contact))


def fit_local_jacobians(train_records, scale, ridge):
    models = []
    for record in train_records:
        action = np.asarray(record["support"]["residual_action"][1:], dtype=np.float64)
        effect = _effect(record["support"])[1:][:, CONTINUOUS_INDICES]
        effect = effect / scale[CONTINUOUS_INDICES][None, :]
        models.append(
            {
                "task_id": record["meta"]["task_id"],
                "phase": record["meta"]["phase"],
                "episode_id": int(record["meta"]["episode_id"]),
                "state": _state_vector(record),
                "j": ridge_jacobian(action, effect, ridge),
            }
        )
    state_matrix = np.stack([row["state"] for row in models])
    center = np.median(state_matrix, axis=0)
    scale_state = np.maximum(np.median(np.abs(state_matrix - center), axis=0) * 1.4826, 1e-3)
    return models, center, scale_state


def _interpolated_j(record, models, center, state_scale, neighbors):
    candidates = [row for row in models if row["task_id"] == record["meta"]["task_id"]]
    same_phase = [row for row in candidates if row["phase"] == record["meta"]["phase"]]
    candidates = same_phase if len(same_phase) >= neighbors else candidates
    state = (_state_vector(record) - center) / state_scale
    distances = np.asarray(
        [np.mean((state - (row["state"] - center) / state_scale) ** 2) for row in candidates]
    )
    order = np.argsort(distances, kind="mergesort")[: int(neighbors)]
    weights = 1.0 / np.maximum(distances[order], 1e-9)
    weights /= np.sum(weights)
    return np.sum(np.stack([candidates[int(index)]["j"] for index in order]) * weights[:, None, None], axis=0)


def evaluate_linear_atlas(records, models, center, state_scale, neighbors, scale, method="O2_linear_j_oracle"):
    rows = []
    prediction_rows = []
    bank_residual = np.asarray(records[0]["candidate"]["residual_action"][1:], dtype=np.float64)
    for record in records:
        j = _interpolated_j(record, models, center, state_scale, neighbors)
        target_action = np.asarray(record["support"]["residual_action"][1:], dtype=np.float64)
        predicted_candidate = bank_residual.dot(j.T)
        predicted_target = target_action.dot(j.T)
        atlas = _fps_tie_stable(predicted_candidate, PRIMARY_K)
        decoded = atlas[_nearest_rows(predicted_target, predicted_candidate[atlas])]
        rows.extend(_evaluate_decoded(record, decoded, method, scale))
        true_target = _effect(record["support"])[1:][:, CONTINUOUS_INDICES] / scale[CONTINUOUS_INDICES][None, :]
        for index in range(len(true_target)):
            prediction_rows.append(
                {
                    "task_id": record["meta"]["task_id"],
                    "episode_id": int(record["meta"]["episode_id"]),
                    "split": record["meta"]["split"],
                    "phase": record["meta"]["phase"],
                    "target_id": index,
                    "method": "LJ_linear_j_atlas",
                    "squared_error": float(np.mean((true_target[index] - predicted_target[index]) ** 2)),
                    "true_norm": float(np.linalg.norm(true_target[index])),
                    "predicted_norm": float(np.linalg.norm(predicted_target[index])),
                    "contact_correct": 0,
                    "uncertainty": float("nan"),
                }
            )
    return rows, prediction_rows


def summarize_rows(rows, method=None):
    selected = [row for row in rows if method is None or row["method"] == method]
    if not selected:
        return {"n": 0, "mean": float("nan")}
    values = np.asarray([row["balanced_task_effect_error"] for row in selected], dtype=np.float64)
    return {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values))}


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({name for row in rows for name in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_predictor_samples(records, consequence_scale):
    """Flatten split-specific target supports into predictor samples."""
    state, residual, target, contact = [], [], [], []
    task_id, phase_id, episode_id, target_id = [], [], [], []
    for record in records:
        n = len(record["support"]["residual_action"]) - 1
        state.append(np.repeat(_state_vector(record)[None, :], n, axis=0))
        residual.append(np.asarray(record["support"]["residual_action"][1:], dtype=np.float64))
        effect = _effect(record["support"])[1:][:, CONTINUOUS_INDICES]
        target.append(effect / consequence_scale[CONTINUOUS_INDICES][None, :])
        contact.append(np.asarray(record["support"]["contact_mode"][1:], dtype=np.int64))
        task_id.extend([TASK_TO_ID[record["meta"]["task_id"]]] * n)
        phase_id.extend([PHASE_TO_ID[record["meta"]["phase"]]] * n)
        episode_id.extend([int(record["meta"]["episode_id"])] * n)
        target_id.extend(range(n))
    return {
        "state": np.concatenate(state, axis=0),
        "residual": np.concatenate(residual, axis=0),
        "target": np.concatenate(target, axis=0),
        "contact": np.concatenate(contact),
        "task_id": np.asarray(task_id, dtype=np.int64),
        "phase_id": np.asarray(phase_id, dtype=np.int64),
        "episode_id": np.asarray(episode_id, dtype=np.int64),
        "target_id": np.asarray(target_id, dtype=np.int64),
    }


def fit_predictor_input_scaler(train_samples):
    state = np.asarray(train_samples["state"], dtype=np.float64)
    # Only physical state coordinates are normalized. Masks and categorical
    # indicators retain their exact 0/1 semantics.
    center = np.zeros(state.shape[1], dtype=np.float64)
    scale = np.ones(state.shape[1], dtype=np.float64)
    for index in range(len(FEATURE_NAMES)):
        values = state[:, index]
        center[index] = np.median(values)
        scale[index] = max(1.4826 * np.median(np.abs(values - center[index])), 1e-3)
    return center, scale


def predictor_input(samples, center, scale, state_override=None, residual_override=None):
    state = np.asarray(samples["state"] if state_override is None else state_override, dtype=np.float64)
    residual = np.asarray(
        samples["residual"] if residual_override is None else residual_override, dtype=np.float64
    )
    normalized_state = (state - center[None, :]) / scale[None, :]
    # Radius normalization keeps residual inputs numerically comparable with
    # normalized state features and is frozen independently of results.
    return np.concatenate((normalized_state, residual / 0.12), axis=1).astype(np.float32)


def _stratified_permutation(task, phase, seed, include_phase=True):
    task = np.asarray(task, dtype=np.int64)
    phase = np.asarray(phase, dtype=np.int64)
    rng = np.random.RandomState(int(seed))
    output = np.arange(len(task), dtype=np.int64)
    keys = [(int(t), int(p)) for t, p in zip(task, phase)] if include_phase else [(int(t),) for t in task]
    groups = defaultdict(list)
    for index, key in enumerate(keys):
        groups[key].append(index)
    for key in sorted(groups):
        indices = np.asarray(groups[key], dtype=np.int64)
        output[indices] = rng.permutation(indices)
    return output


def _torch_device(requested=None):
    import torch

    if requested:
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _make_mlp(input_dim, hidden, output_dim):
    import torch.nn as nn

    layers = []
    previous = input_dim
    for width in hidden:
        layers.extend((nn.Linear(previous, int(width)), nn.ReLU()))
        previous = int(width)
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class _GlobalPredictor:
    @staticmethod
    def create(input_dim, hidden, continuous_dim):
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = _make_mlp(input_dim, hidden, continuous_dim + CONTACT_MODE_COUNT)

            def forward(self, x, phase):
                output = self.network(x)
                return output[:, :continuous_dim], output[:, continuous_dim:]

        return Model()


class _ModePredictor:
    @staticmethod
    def create(input_dim, hidden, continuous_dim):
        import torch
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.trunk = nn.Sequential(nn.Linear(input_dim, int(hidden[0])), nn.ReLU())
                tail = tuple(hidden[1:])
                self.heads = nn.ModuleList(
                    [_make_mlp(int(hidden[0]), tail, continuous_dim + CONTACT_MODE_COUNT) for _ in PHASES]
                )

            def forward(self, x, phase):
                latent = self.trunk(x)
                output = torch.empty(
                    (len(x), continuous_dim + CONTACT_MODE_COUNT), dtype=x.dtype, device=x.device
                )
                for head_id, head in enumerate(self.heads):
                    keep = phase == head_id
                    if torch.any(keep):
                        output[keep] = head(latent[keep])
                return output[:, :continuous_dim], output[:, continuous_dim:]

        return Model()


def _prepare_control_samples(samples, control, seed):
    output = {name: np.asarray(value).copy() for name, value in samples.items()}
    if control == "state_shuffle":
        order = _stratified_permutation(output["task_id"], output["phase_id"], seed, include_phase=True)
        output["state"] = output["state"][order]
    elif control == "effect_shuffle":
        order = _stratified_permutation(output["task_id"], output["phase_id"], seed, include_phase=True)
        output["target"] = output["target"][order]
        output["contact"] = output["contact"][order]
    elif control == "mode_shuffle":
        # Head assignment is shuffled within task while true state features and
        # effect labels remain paired. Inference always uses the true phase.
        order = _stratified_permutation(output["task_id"], output["phase_id"], seed, include_phase=False)
        output["phase_id"] = output["phase_id"][order]
    elif control not in (None, "none"):
        raise KeyError(control)
    return output


def _train_one_predictor(
    train_samples,
    calibration_samples,
    center,
    input_scale,
    hidden,
    seed,
    mode_conditioned,
    control,
    device,
):
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    transformed_train = _prepare_control_samples(train_samples, control, _seed(seed, "control"))
    x_train = predictor_input(transformed_train, center, input_scale)
    x_cal = predictor_input(calibration_samples, center, input_scale)
    y_train = np.asarray(transformed_train["target"], dtype=np.float32)
    y_cal = np.asarray(calibration_samples["target"], dtype=np.float32)
    c_train = np.asarray(transformed_train["contact"], dtype=np.int64)
    c_cal = np.asarray(calibration_samples["contact"], dtype=np.int64)
    p_train = np.asarray(transformed_train["phase_id"], dtype=np.int64)
    p_cal = np.asarray(calibration_samples["phase_id"], dtype=np.int64)
    factory = _ModePredictor if mode_conditioned else _GlobalPredictor
    model = factory.create(x_train.shape[1], hidden, y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=PREDICTOR_LEARNING_RATE, weight_decay=PREDICTOR_WEIGHT_DECAY
    )
    generator = np.random.RandomState(_seed(seed, "batches"))
    calibration_tensors = (
        torch.as_tensor(x_cal, device=device),
        torch.as_tensor(y_cal, device=device),
        torch.as_tensor(c_cal, device=device),
        torch.as_tensor(p_cal, device=device),
    )
    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    patience = 0
    trace = []
    for epoch in range(PREDICTOR_MAX_EPOCHS):
        model.train()
        order = generator.permutation(len(x_train))
        train_losses = []
        for start in range(0, len(order), PREDICTOR_BATCH_SIZE):
            indices = order[start : start + PREDICTOR_BATCH_SIZE]
            x = torch.as_tensor(x_train[indices], device=device)
            y = torch.as_tensor(y_train[indices], device=device)
            contact = torch.as_tensor(c_train[indices], device=device)
            phase = torch.as_tensor(p_train[indices], device=device)
            pred, logits = model(x, phase)
            loss = functional.smooth_l1_loss(pred, y) + 0.25 * functional.cross_entropy(logits, contact)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            x, y, contact, phase = calibration_tensors
            pred, logits = model(x, phase)
            cal_loss = float(
                (functional.smooth_l1_loss(pred, y) + 0.25 * functional.cross_entropy(logits, contact))
                .detach()
                .cpu()
            )
        trace.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "calibration_loss": cal_loss})
        if cal_loss < best_loss - PREDICTOR_MIN_DELTA:
            best_loss = cal_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PREDICTOR_PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "seed": int(seed),
        "hidden": list(hidden),
        "mode_conditioned": bool(mode_conditioned),
        "control": control or "none",
        "best_epoch": int(best_epoch),
        "best_calibration_loss": float(best_loss),
        "epochs_ran": len(trace),
        "trace": trace,
    }


def _save_torch_model(path, model, metadata):
    import torch

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def _predict_ensemble(models, samples, center, input_scale, device, phase_override=None):
    import torch

    x = predictor_input(samples, center, input_scale)
    phase = np.asarray(samples["phase_id"] if phase_override is None else phase_override, dtype=np.int64)
    predictions, probabilities = [], []
    batch = 4096
    for model in models:
        model.eval()
        model_prediction, model_probability = [], []
        with torch.no_grad():
            for start in range(0, len(x), batch):
                inputs = torch.as_tensor(x[start : start + batch], device=device)
                phases = torch.as_tensor(phase[start : start + batch], device=device)
                pred, logits = model(inputs, phases)
                model_prediction.append(pred.detach().cpu().numpy())
                model_probability.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        predictions.append(np.concatenate(model_prediction))
        probabilities.append(np.concatenate(model_probability))
    predictions = np.stack(predictions)
    probabilities = np.stack(probabilities)
    mean = np.mean(predictions, axis=0)
    probability = np.mean(probabilities, axis=0)
    uncertainty = np.mean(np.var(predictions, axis=0), axis=1)
    uncertainty += np.mean(np.var(probabilities, axis=0), axis=1)
    return {"mean": mean, "probability": probability, "mode": np.argmax(probability, axis=1), "uncertainty": uncertainty}


def train_predictor_families(train_records, calibration_records, consequence_scale, output_root, device_name=None):
    """Calibration-select P1/P2 architectures, then fit frozen controls."""
    import torch

    device = _torch_device(device_name)
    train_samples = build_predictor_samples(train_records, consequence_scale)
    calibration_samples = build_predictor_samples(calibration_records, consequence_scale)
    center, input_scale = fit_predictor_input_scaler(train_samples)
    model_root = os.path.join(output_root, "work", "predictors")
    families = {}
    selection_trace = []
    for family, conditioned in (("NCEA", False), ("MC_NCEA", True)):
        candidates = []
        for hidden in PREDICTOR_ARCHITECTURES:
            models, metadata = [], []
            for ensemble_id in range(PREDICTOR_ENSEMBLE_SIZE):
                seed = _seed(GLOBAL_SEED, family, hidden, ensemble_id)
                model, meta = _train_one_predictor(
                    train_samples,
                    calibration_samples,
                    center,
                    input_scale,
                    hidden,
                    seed,
                    conditioned,
                    None,
                    device,
                )
                models.append(model)
                metadata.append(meta)
            score = float(np.mean([row["best_calibration_loss"] for row in metadata]))
            candidates.append((score, tuple(hidden), models, metadata))
            selection_trace.append({"family": family, "hidden": list(hidden), "mean_calibration_loss": score})
        score, hidden, models, metadata = min(candidates, key=lambda row: (row[0], row[1]))
        for ensemble_id, (model, meta) in enumerate(zip(models, metadata)):
            _save_torch_model(
                os.path.join(model_root, "%s_member_%d.pt" % (family, ensemble_id)), model, meta
            )
        families[family] = {
            "models": models,
            "hidden": hidden,
            "mode_conditioned": conditioned,
            "calibration_loss": score,
            "metadata": metadata,
        }
        # Explicitly release unselected candidate models before the next family.
        del candidates
        if device.type == "cuda":
            torch.cuda.empty_cache()

    controls = (
        ("P3_mode_shuffled_atlas", True, "mode_shuffle", families["MC_NCEA"]["hidden"]),
        ("P4_state_shuffled_atlas", False, "state_shuffle", families["NCEA"]["hidden"]),
        ("P5_effect_shuffled_atlas", False, "effect_shuffle", families["NCEA"]["hidden"]),
    )
    for family, conditioned, control, hidden in controls:
        models, metadata = [], []
        for ensemble_id in range(PREDICTOR_ENSEMBLE_SIZE):
            seed = _seed(GLOBAL_SEED, family, hidden, ensemble_id)
            model, meta = _train_one_predictor(
                train_samples,
                calibration_samples,
                center,
                input_scale,
                hidden,
                seed,
                conditioned,
                control,
                device,
            )
            models.append(model)
            metadata.append(meta)
            _save_torch_model(
                os.path.join(model_root, "%s_member_%d.pt" % (family, ensemble_id)), model, meta
            )
        families[family] = {
            "models": models,
            "hidden": hidden,
            "mode_conditioned": conditioned,
            "calibration_loss": float(np.mean([row["best_calibration_loss"] for row in metadata])),
            "metadata": metadata,
        }
    freeze = {
        "created_utc": utc_now(),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "train_samples": int(len(train_samples["target"])),
        "calibration_samples": int(len(calibration_samples["target"])),
        "input_center": center.tolist(),
        "input_scale": input_scale.tolist(),
        "consequence_scale": consequence_scale.tolist(),
        "architecture_selection": selection_trace,
        "selected": {
            name: {
                "hidden": list(value["hidden"]),
                "mode_conditioned": value["mode_conditioned"],
                "calibration_loss": value["calibration_loss"],
                "members": PREDICTOR_ENSEMBLE_SIZE,
            }
            for name, value in families.items()
        },
    }
    atomic_json(os.path.join(output_root, "work", "predictor_freeze.json"), freeze)
    atomic_npz(
        os.path.join(output_root, "work", "predictor_scalers.npz"),
        input_center=center,
        input_scale=input_scale,
        consequence_scale=consequence_scale,
    )
    return families, train_samples, calibration_samples, center, input_scale, device, freeze


def _random_feature_fit(train_samples, calibration_samples, center, input_scale, hidden_width):
    x_train = predictor_input(train_samples, center, input_scale).astype(np.float64)
    x_cal = predictor_input(calibration_samples, center, input_scale).astype(np.float64)
    y = np.concatenate(
        (
            np.asarray(train_samples["target"], dtype=np.float64),
            np.eye(CONTACT_MODE_COUNT)[train_samples["contact"]],
        ),
        axis=1,
    )
    members = []
    for member in range(PREDICTOR_ENSEMBLE_SIZE):
        rng = np.random.RandomState(_seed(GLOBAL_SEED, "P6", member))
        weight = rng.normal(scale=1.0 / math.sqrt(x_train.shape[1]), size=(x_train.shape[1], hidden_width))
        bias = rng.normal(scale=0.1, size=hidden_width)
        feature = np.maximum(x_train.dot(weight) + bias, 0.0)
        gram = feature.T.dot(feature) + 1e-3 * np.eye(hidden_width)
        readout = np.linalg.solve(gram, feature.T.dot(y))
        cal_feature = np.maximum(x_cal.dot(weight) + bias, 0.0)
        cal_output = cal_feature.dot(readout)
        cal_loss = float(np.mean((cal_output[:, : len(CONTINUOUS_INDICES)] - calibration_samples["target"]) ** 2))
        members.append({"weight": weight, "bias": bias, "readout": readout, "calibration_mse": cal_loss})
    return members


def _random_feature_predict(members, samples, center, input_scale):
    x = predictor_input(samples, center, input_scale).astype(np.float64)
    output = np.stack(
        [np.maximum(x.dot(row["weight"]) + row["bias"], 0.0).dot(row["readout"]) for row in members]
    )
    continuous = output[:, :, : len(CONTINUOUS_INDICES)]
    logits = output[:, :, len(CONTINUOUS_INDICES) :]
    logits -= np.max(logits, axis=2, keepdims=True)
    probability = np.exp(logits)
    probability /= np.sum(probability, axis=2, keepdims=True)
    return {
        "mean": np.mean(continuous, axis=0),
        "probability": np.mean(probability, axis=0),
        "mode": np.argmax(np.mean(probability, axis=0), axis=1),
        "uncertainty": np.mean(np.var(continuous, axis=0), axis=1) + np.mean(np.var(probability, axis=0), axis=1),
    }


def _query_samples(record, residuals):
    residuals = np.asarray(residuals, dtype=np.float64)
    n = len(residuals)
    return {
        "state": np.repeat(_state_vector(record)[None, :], n, axis=0),
        "residual": residuals,
        "target": np.zeros((n, len(CONTINUOUS_INDICES)), dtype=np.float64),
        "contact": np.zeros(n, dtype=np.int64),
        "task_id": np.full(n, TASK_TO_ID[record["meta"]["task_id"]], dtype=np.int64),
        "phase_id": np.full(n, PHASE_TO_ID[record["meta"]["phase"]], dtype=np.int64),
        "episode_id": np.full(n, int(record["meta"]["episode_id"]), dtype=np.int64),
        "target_id": np.arange(n, dtype=np.int64),
    }


def _predicted_embedding(prediction, record):
    normalized = np.asarray(prediction["mean"], dtype=np.float64)
    mode = np.asarray(prediction["mode"], dtype=np.int64)
    mask = np.asarray(record["support"]["mask"][0], dtype=bool)
    blocks = []
    index_to_column = {int(index): column for column, index in enumerate(CONTINUOUS_INDICES)}
    for group, indices in PRIMARY_GROUPS.items():
        columns = [index_to_column[int(index)] for index in indices]
        block = np.clip(normalized[:, columns], -HUBER_CAP, HUBER_CAP)
        active = mask[np.asarray(indices, dtype=np.int64)].astype(np.float64)
        block = block * active[None, :]
        block /= math.sqrt(max(len(indices), 1) * len(PRIMARY_GROUPS))
        blocks.append(block)
        if group == "contact_mode_and_penetration":
            blocks.append(np.eye(CONTACT_MODE_COUNT)[mode] / math.sqrt(CONTACT_MODE_COUNT * len(PRIMARY_GROUPS)))
    return np.concatenate(blocks, axis=1)


def _prediction_metric_rows(record, method, prediction, consequence_scale):
    true = _effect(record["support"])[1:][:, CONTINUOUS_INDICES]
    true_normalized = true / consequence_scale[CONTINUOUS_INDICES][None, :]
    target_mode = np.asarray(record["support"]["contact_mode"][1:], dtype=np.int64)
    rows = []
    for index in range(len(true)):
        squared = float(np.mean((true_normalized[index] - prediction["mean"][index]) ** 2))
        denom = max(float(np.mean(true_normalized[index] ** 2)), 1e-12)
        rows.append(
            {
                "task_id": record["meta"]["task_id"],
                "episode_id": int(record["meta"]["episode_id"]),
                "split": record["meta"]["split"],
                "phase": record["meta"]["phase"],
                "target_id": index,
                "method": method,
                "squared_error": squared,
                "nrmse": math.sqrt(squared / denom),
                "true_norm": float(np.linalg.norm(true_normalized[index])),
                "predicted_norm": float(np.linalg.norm(prediction["mean"][index])),
                "contact_correct": int(target_mode[index] == prediction["mode"][index]),
                "contact_probability": float(prediction["probability"][index, target_mode[index]]),
                "uncertainty": float(prediction["uncertainty"][index]),
            }
        )
    return rows


def evaluate_predictor_atlases(
    records,
    families,
    random_members,
    center,
    input_scale,
    consequence_scale,
    device,
):
    quantization_rows, prediction_rows = [], []
    for record in records:
        target_residual = np.asarray(record["support"]["residual_action"][1:], dtype=np.float64)
        bank_residual = np.asarray(record["candidate"]["residual_action"][1:], dtype=np.float64)
        target_samples = _query_samples(record, target_residual)
        bank_samples = _query_samples(record, bank_residual)
        for method, family in families.items():
            target_prediction = _predict_ensemble(
                family["models"], target_samples, center, input_scale, device
            )
            bank_prediction = _predict_ensemble(
                family["models"], bank_samples, center, input_scale, device
            )
            target_embedding = _predicted_embedding(target_prediction, record)
            bank_embedding = _predicted_embedding(bank_prediction, record)
            atlas = _fps_tie_stable(bank_embedding, PRIMARY_K)
            decoded = atlas[_nearest_rows(target_embedding, bank_embedding[atlas])]
            quantization_rows.extend(
                _evaluate_decoded(
                    record,
                    decoded,
                    method,
                    consequence_scale,
                    extra={
                        "target_uncertainty": target_prediction["uncertainty"],
                        "decoded_uncertainty": bank_prediction["uncertainty"][decoded],
                        "nearest_predicted_distance": np.sqrt(
                            np.sum((target_embedding - bank_embedding[decoded]) ** 2, axis=1)
                        ),
                        "atlas_size": PRIMARY_K,
                        "atlas_unique": len(np.unique(atlas)),
                    },
                )
            )
            prediction_rows.extend(
                _prediction_metric_rows(record, method, target_prediction, consequence_scale)
            )
        target_prediction = _random_feature_predict(random_members, target_samples, center, input_scale)
        bank_prediction = _random_feature_predict(random_members, bank_samples, center, input_scale)
        target_embedding = _predicted_embedding(target_prediction, record)
        bank_embedding = _predicted_embedding(bank_prediction, record)
        atlas = _fps_tie_stable(bank_embedding, PRIMARY_K)
        decoded = atlas[_nearest_rows(target_embedding, bank_embedding[atlas])]
        method = "P6_random_latent_atlas"
        quantization_rows.extend(
            _evaluate_decoded(
                record,
                decoded,
                method,
                consequence_scale,
                extra={
                    "target_uncertainty": target_prediction["uncertainty"],
                    "decoded_uncertainty": bank_prediction["uncertainty"][decoded],
                    "nearest_predicted_distance": np.sqrt(
                        np.sum((target_embedding - bank_embedding[decoded]) ** 2, axis=1)
                    ),
                    "atlas_size": PRIMARY_K,
                    "atlas_unique": len(np.unique(atlas)),
                },
            )
        )
        prediction_rows.extend(
            _prediction_metric_rows(record, method, target_prediction, consequence_scale)
        )
    return quantization_rows, prediction_rows


def train_action_autoencoder(train_records, calibration_records, action_bank, center, input_scale, hidden, output_root, device):
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    input_dim = len(center) + action_bank.shape[1]
    state_dim = len(center)

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _make_mlp(input_dim, hidden, ACTION_LATENT_DIM)
            self.decoder = _make_mlp(ACTION_LATENT_DIM + state_dim, tuple(reversed(hidden)), action_bank.shape[1])

        def encode(self, state, action):
            return self.encoder(torch.cat((state, action), dim=1))

        def forward(self, state, action):
            latent = self.encode(state, action)
            return self.decoder(torch.cat((latent, state), dim=1)), latent

    def pairs(records):
        states = np.stack([_state_vector(row) for row in records])
        states = (states - center[None, :]) / input_scale[None, :]
        states = np.repeat(states, len(action_bank), axis=0).astype(np.float32)
        actions = np.tile(action_bank, (len(records), 1)).astype(np.float32) / 0.12
        return states, actions

    x_train, a_train = pairs(train_records)
    x_cal, a_cal = pairs(calibration_records)
    seed = _seed(GLOBAL_SEED, "B4_state_action_vq")
    torch.manual_seed(seed)
    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PREDICTOR_LEARNING_RATE, weight_decay=PREDICTOR_WEIGHT_DECAY)
    rng = np.random.RandomState(seed)
    best_loss, best_state, best_epoch, patience = float("inf"), None, -1, 0
    trace = []
    for epoch in range(PREDICTOR_MAX_EPOCHS):
        order = rng.permutation(len(x_train))
        losses = []
        model.train()
        for start in range(0, len(order), PREDICTOR_BATCH_SIZE):
            index = order[start : start + PREDICTOR_BATCH_SIZE]
            state = torch.as_tensor(x_train[index], device=device)
            action = torch.as_tensor(a_train[index], device=device)
            decoded, _ = model(state, action)
            loss = functional.mse_loss(decoded, action)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            decoded, _ = model(torch.as_tensor(x_cal, device=device), torch.as_tensor(a_cal, device=device))
            cal_loss = float(functional.mse_loss(decoded, torch.as_tensor(a_cal, device=device)).detach().cpu())
        trace.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "calibration_loss": cal_loss})
        if cal_loss < best_loss - PREDICTOR_MIN_DELTA:
            best_loss, best_epoch, patience = cal_loss, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            patience += 1
            if patience >= PREDICTOR_PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    metadata = {
        "seed": seed,
        "hidden": list(hidden),
        "latent_dim": ACTION_LATENT_DIM,
        "best_epoch": best_epoch,
        "best_calibration_loss": best_loss,
        "epochs_ran": len(trace),
        "consequence_labels_used": False,
        "trace": trace,
    }
    _save_torch_model(os.path.join(output_root, "work", "predictors", "B4_state_action_vq.pt"), model, metadata)
    return model, metadata


def evaluate_action_autoencoder(records, model, action_bank, center, input_scale, consequence_scale, device):
    import torch

    rows = []
    for record in records:
        state = (_state_vector(record) - center) / input_scale
        target = np.asarray(record["support"]["residual_action"][1:], dtype=np.float64)
        with torch.no_grad():
            state_bank = torch.as_tensor(
                np.repeat(state[None, :], len(action_bank), axis=0), dtype=torch.float32, device=device
            )
            state_target = torch.as_tensor(
                np.repeat(state[None, :], len(target), axis=0), dtype=torch.float32, device=device
            )
            bank_tensor = torch.as_tensor(action_bank / 0.12, dtype=torch.float32, device=device)
            target_tensor = torch.as_tensor(target / 0.12, dtype=torch.float32, device=device)
            bank_latent = model.encode(state_bank, bank_tensor).cpu().numpy()
            target_latent = model.encode(state_target, target_tensor).cpu().numpy()
        atlas = _fps_tie_stable(bank_latent, PRIMARY_K)
        decoded = atlas[_nearest_rows(target_latent, bank_latent[atlas])]
        rows.extend(
            _evaluate_decoded(
                record,
                decoded,
                "B4_state_action_vq",
                consequence_scale,
                extra={"atlas_size": PRIMARY_K, "atlas_unique": len(np.unique(atlas))},
            )
        )
    return rows


def _mean_by(rows, value, methods=None, task=None, phase=None):
    selected = [
        row
        for row in rows
        if (methods is None or row["method"] in methods)
        and (task is None or row["task_id"] == task)
        and (phase is None or row["phase"] == phase)
    ]
    return float(np.mean([row[value] for row in selected])) if selected else float("nan")


def _relative_gain(baseline, method):
    return float((baseline - method) / baseline) if baseline > 0 else 0.0


def select_linear_configuration(train_records, calibration_records, consequence_scale):
    trace = []
    candidates = {}
    for ridge in LINEAR_RIDGE_GRID:
        models, center, state_scale = fit_local_jacobians(train_records, consequence_scale, ridge)
        candidates[ridge] = (models, center, state_scale)
        for neighbors in LINEAR_NEIGHBOR_GRID:
            _, prediction = evaluate_linear_atlas(
                calibration_records,
                models,
                center,
                state_scale,
                neighbors,
                consequence_scale,
            )
            mse = float(np.mean([row["squared_error"] for row in prediction]))
            trace.append({"ridge": float(ridge), "neighbors": int(neighbors), "calibration_mse": mse})
    chosen = min(trace, key=lambda row: (row["calibration_mse"], row["ridge"], row["neighbors"]))
    models, center, state_scale = candidates[chosen["ridge"]]
    return models, center, state_scale, int(chosen["neighbors"]), chosen, trace


def _predictor_metric_summary(rows):
    output = []
    methods = sorted({row["method"] for row in rows})
    splits = sorted({row["split"] for row in rows})
    for split in splits:
        for method in methods:
            partitions = [("pooled", "ALL", "ALL")]
            partitions += [("task", task, "ALL") for task in TASK_IDS]
            partitions += [("phase", "ALL", phase) for phase in PHASES]
            for level, task, phase in partitions:
                selected = [
                    row
                    for row in rows
                    if row["split"] == split
                    and row["method"] == method
                    and (task == "ALL" or row["task_id"] == task)
                    and (phase == "ALL" or row["phase"] == phase)
                ]
                if not selected:
                    continue
                true_norm = np.asarray([row["true_norm"] for row in selected])
                pred_norm = np.asarray([row["predicted_norm"] for row in selected])
                errors = np.asarray([row["squared_error"] for row in selected])
                uncertainties = np.asarray([row["uncertainty"] for row in selected])
                output.append(
                    {
                        "split": split,
                        "method": method,
                        "level": level,
                        "task_id": task,
                        "phase": phase,
                        "n": len(selected),
                        "mse": float(np.mean(errors)),
                        "rmse": float(np.sqrt(np.mean(errors))),
                        "mean_nrmse": float(np.mean([row.get("nrmse", float("nan")) for row in selected])),
                        "contact_accuracy": float(np.mean([row["contact_correct"] for row in selected])),
                        "effect_norm_spearman": spearmanr(true_norm, pred_norm),
                        "uncertainty_error_spearman": spearmanr(uncertainties, errors),
                    }
                )
    return output


def _quantization_summary(rows):
    output = []
    methods = sorted({row["method"] for row in rows})
    for method in methods:
        partitions = [("pooled", "ALL", "ALL")]
        partitions += [("task", task, "ALL") for task in TASK_IDS]
        partitions += [("phase", "ALL", phase) for phase in PHASES]
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
            decoded = np.asarray([row["decoded_bank_index"] for row in selected], dtype=np.int64)
            decoded = decoded[decoded >= 0]
            counts = np.bincount(decoded, minlength=ACTION_BANK_SIZE)
            probability = counts[counts > 0] / max(np.sum(counts), 1)
            perplexity = float(np.exp(-np.sum(probability * np.log(probability)))) if len(probability) else 0.0
            output.append(
                {
                    "split": selected[0]["split"],
                    "method": method,
                    "level": level,
                    "task_id": task,
                    "phase": phase,
                    "n": len(selected),
                    "balanced_task_effect_error": float(
                        np.mean([row["balanced_task_effect_error"] for row in selected])
                    ),
                    "action_reconstruction_rmse": float(
                        np.mean([row["action_reconstruction_rmse"] for row in selected])
                    ),
                    "contact_mode_preservation": float(
                        np.mean([row["contact_mode_preserved"] for row in selected])
                    ),
                    "task_progress_abs_error": float(
                        np.mean([row["task_progress_abs_error"] for row in selected])
                    ),
                    "contact_force_effect_error": float(
                        np.mean([row["contact_force_effect_error"] for row in selected])
                    ),
                    "clipping_rate": float(np.mean([row["clipped"] for row in selected])),
                    "unique_codes": int(np.sum(counts > 0)),
                    "code_perplexity": perplexity,
                    "normalized_code_utilization": perplexity / PRIMARY_K,
                }
            )
    return output


def _rows_by_key(rows, method):
    return {
        (row["task_id"], row["episode_id"], row["phase"], row["target_id"]): row
        for row in rows
        if row["method"] == method
    }


def build_uncertainty_gates(calibration_rows, nonlinear_method, fallback_method):
    nonlinear = _rows_by_key(calibration_rows, nonlinear_method)
    fallback = _rows_by_key(calibration_rows, fallback_method)
    keys = sorted(set(nonlinear) & set(fallback))
    risk = np.asarray(
        [
            float(nonlinear[key].get("target_uncertainty", 0.0))
            + float(nonlinear[key].get("decoded_uncertainty", 0.0))
            + float(nonlinear[key].get("nearest_predicted_distance", 0.0))
            for key in keys
        ],
        dtype=np.float64,
    )
    return {
        "nonlinear_method": nonlinear_method,
        "fallback_method": fallback_method,
        "risk_definition": "target_uncertainty + decoded_uncertainty + nearest_predicted_distance",
        "thresholds": {
            str(coverage): float(np.quantile(risk, coverage)) for coverage in UNCERTAINTY_COVERAGES
        },
        "calibration_risk_spearman": spearmanr(
            risk, [nonlinear[key]["balanced_task_effect_error"] for key in keys]
        ),
    }


def apply_uncertainty_gates(rows, gate):
    nonlinear = _rows_by_key(rows, gate["nonlinear_method"])
    fallback = _rows_by_key(rows, gate["fallback_method"])
    keys = sorted(set(nonlinear) & set(fallback))
    output = []
    for coverage_text, threshold in gate["thresholds"].items():
        coverage = float(coverage_text)
        for key in keys:
            source = nonlinear[key]
            risk = (
                float(source.get("target_uncertainty", 0.0))
                + float(source.get("decoded_uncertainty", 0.0))
                + float(source.get("nearest_predicted_distance", 0.0))
            )
            use_nonlinear = risk <= threshold
            row = dict(source if use_nonlinear else fallback[key])
            row["method"] = "UG_NCEA_coverage_%02d" % int(round(coverage * 100))
            row["quantization_coverage_target"] = coverage
            row["used_consequence_quantizer"] = int(use_nonlinear)
            row["uncertainty_risk"] = risk
            row["uncertainty_threshold"] = threshold
            output.append(row)
    return output


def _gate_a(rows):
    baseline_methods = ("B1_centered_covariance", "B2_phase_residual", "B3_dynamic_action_medoids")
    baseline_errors = {method: _mean_by(rows, "balanced_task_effect_error", (method,)) for method in baseline_methods}
    strongest = min(baseline_errors, key=lambda name: (baseline_errors[name], name))
    baseline = baseline_errors[strongest]
    oracle = _mean_by(rows, "balanced_task_effect_error", ("O1_true_effect_oracle",))
    per_task = {}
    for task in TASK_IDS:
        base = _mean_by(rows, "balanced_task_effect_error", (strongest,), task=task)
        value = _mean_by(rows, "balanced_task_effect_error", ("O1_true_effect_oracle",), task=task)
        per_task[task] = {"baseline": base, "oracle": value, "relative_gain": _relative_gain(base, value), "improved": value < base}
    gain = _relative_gain(baseline, oracle)
    improved = sum(int(row["improved"]) for row in per_task.values())
    passed = gain >= GATES["A"]["oracle_relative_gain_min"] and improved >= GATES["A"]["tasks_improved_min"]
    return {
        "passed": bool(passed),
        "strongest_baseline": strongest,
        "baseline_errors": baseline_errors,
        "baseline_error": baseline,
        "oracle_error": oracle,
        "relative_gain": gain,
        "tasks_improved": improved,
        "per_task": per_task,
        "requirements": GATES["A"],
    }


def _control_rows(rows, best_method, baseline_method):
    baseline = _mean_by(rows, "balanced_task_effect_error", (baseline_method,))
    best = _mean_by(rows, "balanced_task_effect_error", (best_method,))
    gain = baseline - best
    output = []
    for method in (
        "P3_mode_shuffled_atlas",
        "P4_state_shuffled_atlas",
        "P5_effect_shuffled_atlas",
        "P6_random_latent_atlas",
    ):
        error = _mean_by(rows, "balanced_task_effect_error", (method,))
        retention = max(0.0, baseline - error) / max(gain, 1e-12)
        output.append(
            {
                "split": rows[0]["split"],
                "best_method": best_method,
                "baseline_method": baseline_method,
                "control_method": method,
                "baseline_error": baseline,
                "best_error": best,
                "control_error": error,
                "gain_retention": retention,
                "reproduced_gain": bool(retention >= 1.0),
            }
        )
    return output


def support_nonlinearity_diagnostics(records, consequence_scale):
    rows = []
    for record in records:
        shard = record["support"]
        action = np.asarray(shard["residual_action"][1:], dtype=np.float64)
        effect = _effect(shard)[1:][:, CONTINUOUS_INDICES] / consequence_scale[CONTINUOUS_INDICES][None, :]
        direction = np.asarray(shard["direction_id"][1:], dtype=np.int64)
        family = np.asarray(shard["direction_family_id"][1:], dtype=np.int64)
        radius = np.asarray(shard["radius"][1:], dtype=np.float64)
        sign = np.asarray(shard["sign"][1:], dtype=np.int64)
        small = np.zeros(len(direction), dtype=bool)
        for direction_id in np.unique(direction):
            keep = np.flatnonzero(direction == direction_id)
            unique_radius = np.unique(radius[keep])
            small[keep[np.isclose(radius[keep], unique_radius[0])]] = True
        j = ridge_jacobian(action[small], effect[small], 1e-4)
        prediction = action[~small].dot(j.T)
        truth = effect[~small]
        linear_nrmse = math.sqrt(
            float(np.mean((prediction - truth) ** 2)) / max(float(np.mean(truth**2)), 1e-12)
        )
        asymmetry, scaling = [], []
        for direction_id in np.unique(direction):
            keep_direction = direction == direction_id
            unique_radius = np.sort(np.unique(radius[keep_direction]))
            norms = {}
            for value in unique_radius:
                plus = effect[keep_direction & np.isclose(radius, value) & (sign == 1)][0]
                minus = effect[keep_direction & np.isclose(radius, value) & (sign == -1)][0]
                denominator = max(0.5 * (np.linalg.norm(plus) + np.linalg.norm(minus)), 1e-12)
                asymmetry.append(float(np.linalg.norm(plus + minus) / denominator))
                norms[value] = 0.5 * (np.linalg.norm(plus) + np.linalg.norm(minus))
            expected = unique_radius[1] / unique_radius[0]
            observed = norms[unique_radius[1]] / max(norms[unique_radius[0]], 1e-12)
            scaling.append(float(abs(math.log(max(observed, 1e-12) / expected))))
        rows.append(
            {
                "split": record["meta"]["split"],
                "task_id": record["meta"]["task_id"],
                "episode_id": int(record["meta"]["episode_id"]),
                "phase": record["meta"]["phase"],
                "linear_extrapolation_nrmse": linear_nrmse,
                "mean_antithetic_asymmetry": float(np.mean(asymmetry)),
                "mean_log_radius_scaling_deviation": float(np.mean(scaling)),
            }
        )
    return rows


def mechanism_decomposition(quantization_rows, nonlinearity_rows):
    output = []
    for method in sorted({row["method"] for row in quantization_rows}):
        selected_method = [row for row in quantization_rows if row["method"] == method]
        for dimension, values in (
            ("task", TASK_IDS),
            ("phase", PHASES),
            ("direction_family", (0, 1, 2)),
        ):
            for value in values:
                selected = [
                    row
                    for row in selected_method
                    if (row["task_id"] if dimension == "task" else row["phase"] if dimension == "phase" else row["direction_family_id"])
                    == value
                ]
                if not selected:
                    continue
                result = {
                    "row_type": "quantization_decomposition",
                    "method": method,
                    "dimension": dimension,
                    "value": value,
                    "n": len(selected),
                    "balanced_task_effect_error": float(
                        np.mean([row["balanced_task_effect_error"] for row in selected])
                    ),
                    "action_reconstruction_rmse": float(
                        np.mean([row["action_reconstruction_rmse"] for row in selected])
                    ),
                }
                for group in PRIMARY_GROUPS:
                    result["error_group_" + group] = float(
                        np.mean([row["error_group_" + group] for row in selected])
                    )
                output.append(result)
    for row in nonlinearity_rows:
        result = {"row_type": "support_nonlinearity", "method": "PHYSICAL_SUPPORT"}
        result.update(row)
        output.append(result)
    return output


def _gate_b(quantization_rows, prediction_rows):
    nonlinear_methods = ("NCEA", "MC_NCEA")
    prediction_error = {
        method: _mean_by(prediction_rows, "squared_error", (method,)) for method in nonlinear_methods
    }
    best = min(prediction_error, key=lambda name: (prediction_error[name], name))
    best_error = prediction_error[best]
    linear_error = _mean_by(prediction_rows, "squared_error", ("LJ_linear_j_atlas",))
    prediction_gain = _relative_gain(linear_error, best_error)
    contact_task_improvements = 0
    per_task = {}
    for task in CONTACT_SENSITIVE_TASKS:
        linear = _mean_by(prediction_rows, "squared_error", ("LJ_linear_j_atlas",), task=task)
        value = _mean_by(prediction_rows, "squared_error", (best,), task=task)
        per_task[task] = {"linear": linear, "nonlinear": value, "improved": value < linear}
        contact_task_improvements += int(value < linear)
    o1 = _mean_by(quantization_rows, "balanced_task_effect_error", ("O1_true_effect_oracle",))
    o2 = _mean_by(quantization_rows, "balanced_task_effect_error", ("O2_linear_j_oracle",))
    nonlinear = _mean_by(quantization_rows, "balanced_task_effect_error", (best,))
    gap_closed = (o2 - nonlinear) / (o2 - o1) if o2 > o1 else 0.0
    control_prediction = {
        method: _mean_by(prediction_rows, "squared_error", (method,))
        for method in (
            "P3_mode_shuffled_atlas",
            "P4_state_shuffled_atlas",
            "P5_effect_shuffled_atlas",
            "P6_random_latent_atlas",
        )
    }
    beats_controls = all(best_error < error for error in control_prediction.values())
    passed = (
        prediction_gain >= GATES["B"]["prediction_relative_gain_vs_linear_min"]
        and contact_task_improvements >= GATES["B"]["contact_sensitive_tasks_improved_min"]
        and gap_closed >= GATES["B"]["oracle_gap_fraction_closed_min"]
        and beats_controls
    )
    return {
        "passed": bool(passed),
        "best_nonlinear": best,
        "prediction_errors": prediction_error,
        "linear_prediction_error": linear_error,
        "prediction_relative_gain": prediction_gain,
        "contact_sensitive_tasks_improved": contact_task_improvements,
        "per_contact_task": per_task,
        "o1_error": o1,
        "o2_error": o2,
        "nonlinear_realized_error": nonlinear,
        "oracle_gap_fraction_closed": gap_closed,
        "control_prediction_errors": control_prediction,
        "beats_all_controls": bool(beats_controls),
        "requirements": GATES["B"],
    }


def _gate_c(rows, strongest_baseline):
    candidates = ("NCEA", "MC_NCEA")
    baseline_error = _mean_by(rows, "balanced_task_effect_error", (strongest_baseline,))
    method_errors = {method: _mean_by(rows, "balanced_task_effect_error", (method,)) for method in candidates}
    best = min(method_errors, key=lambda name: (method_errors[name], name))
    best_error = method_errors[best]
    task_results = {}
    for task in TASK_IDS:
        baseline = _mean_by(rows, "balanced_task_effect_error", (strongest_baseline,), task=task)
        method = _mean_by(rows, "balanced_task_effect_error", (best,), task=task)
        task_results[task] = {
            "baseline": baseline,
            "method": method,
            "relative_gain": _relative_gain(baseline, method),
            "improved": method < baseline,
        }
    improved = sum(int(row["improved"]) for row in task_results.values())
    contact_improved = sum(int(task_results[task]["improved"]) for task in CONTACT_SENSITIVE_TASKS)
    bowl_degradation = -task_results["bowl_on_plate"]["relative_gain"]
    summary = next(
        row for row in _quantization_summary(rows) if row["method"] == best and row["level"] == "pooled"
    )
    baseline_summary = next(
        row
        for row in _quantization_summary(rows)
        if row["method"] == strongest_baseline and row["level"] == "pooled"
    )
    reconstruction_degradation = (
        summary["action_reconstruction_rmse"] - baseline_summary["action_reconstruction_rmse"]
    ) / max(baseline_summary["action_reconstruction_rmse"], 1e-12)
    controls = _control_rows(rows, best, strongest_baseline)
    controls_destroy_gain = all(not row["reproduced_gain"] for row in controls)
    relative_gain = _relative_gain(baseline_error, best_error)
    passed = (
        relative_gain >= GATES["C"]["relative_gain_min"]
        and improved >= GATES["C"]["tasks_improved_min"]
        and contact_improved >= GATES["C"]["contact_sensitive_tasks_improved_min"]
        and bowl_degradation <= GATES["C"]["bowl_on_plate_max_degradation"]
        and summary["clipping_rate"] < GATES["C"]["clipping_rate_max"]
        and summary["normalized_code_utilization"] > GATES["C"]["normalized_code_utilization_min"]
        and controls_destroy_gain
        and reconstruction_degradation < GATES["C"]["action_reconstruction_degradation_max"]
    )
    return {
        "passed": bool(passed),
        "best_method": best,
        "strongest_calibration_baseline": strongest_baseline,
        "baseline_error": baseline_error,
        "method_errors": method_errors,
        "best_error": best_error,
        "relative_gain": relative_gain,
        "tasks_improved": improved,
        "contact_sensitive_tasks_improved": contact_improved,
        "bowl_on_plate_degradation": bowl_degradation,
        "clipping_rate": summary["clipping_rate"],
        "normalized_code_utilization": summary["normalized_code_utilization"],
        "action_reconstruction_degradation": reconstruction_degradation,
        "controls_destroy_gain": controls_destroy_gain,
        "per_task": task_results,
        "controls": controls,
        "requirements": GATES["C"],
    }


def _save_random_members(path, members):
    atomic_npz(
        path,
        weight=np.stack([row["weight"] for row in members]),
        bias=np.stack([row["bias"] for row in members]),
        readout=np.stack([row["readout"] for row in members]),
        calibration_mse=np.asarray([row["calibration_mse"] for row in members]),
    )


def _empty_confirmation(output_root, disposition, gate):
    import zarr

    destination = os.path.join(output_root, "confirmation_rollouts.zarr")
    root = zarr.open_group(destination, mode="w")
    root.attrs.update(
        {
            "status": "NOT_RUN_DEVELOPMENT_GATE_FAILED",
            "observations": 0,
            "disposition": disposition,
            "created_utc": utc_now(),
            "confirmation_results_accessed": False,
        }
    )
    write_csv(
        os.path.join(output_root, "confirmation_quantization.csv"),
        [],
        fieldnames=("status", "observations", "disposition"),
    )
    atomic_json(
        os.path.join(output_root, "bootstrap_results.json"),
        {
            "status": "NOT_RUN_DEVELOPMENT_GATE_FAILED",
            "observations": 0,
            "replicates_executed": 0,
            "replicates_preregistered": 10000,
            "confirmation_results_accessed": False,
            "disposition": disposition,
            "development_gate": gate,
        },
    )


def run_development(output_root, device_name=None):
    """Fit on train/calibration, freeze choices, then execute sequential dev gates."""
    train_records = load_state_records(output_root, {"train"})
    calibration_records = load_state_records(output_root, {"calibration"})
    consequence_scale, scale_evidence = fit_train_scaling(train_records)
    atomic_json(
        os.path.join(output_root, "work", "consequence_scale_evidence.json"),
        {"created_utc": utc_now(), "rows": scale_evidence, "scale": consequence_scale.tolist()},
    )
    codebooks = build_action_baseline_codebooks(output_root)
    atomic_npz(
        os.path.join(output_root, "work", "baseline_codebooks.npz"),
        B1=codebooks["B1"],
        B2_free_space=codebooks["B2"]["free_space"],
        B2_pre_contact=codebooks["B2"]["pre_contact"],
        B2_contact_onset=codebooks["B2"]["contact_onset"],
        B2_post_contact=codebooks["B2"]["post_contact"],
        B3=codebooks["B3"],
        whitening=codebooks["whitening"],
        covariance_eigenvalues=codebooks["covariance_eigenvalues"],
    )
    linear_models, linear_center, linear_scale, neighbors, linear_choice, linear_trace = select_linear_configuration(
        train_records, calibration_records, consequence_scale
    )
    families, train_samples, calibration_samples, input_center, input_scale, device, predictor_freeze = train_predictor_families(
        train_records, calibration_records, consequence_scale, output_root, device_name=device_name
    )
    random_members = _random_feature_fit(
        train_samples, calibration_samples, input_center, input_scale, families["NCEA"]["hidden"][-1]
    )
    _save_random_members(os.path.join(output_root, "work", "predictors", "P6_random_latent.npz"), random_members)
    autoencoder, autoencoder_meta = train_action_autoencoder(
        train_records,
        calibration_records,
        codebooks["residuals"],
        input_center,
        input_scale,
        families["NCEA"]["hidden"],
        output_root,
        device,
    )

    calibration_quant = evaluate_continuous_upper_bound(calibration_records)
    calibration_quant += evaluate_action_baselines(calibration_records, codebooks, consequence_scale)
    calibration_quant += evaluate_action_autoencoder(
        calibration_records, autoencoder, codebooks["residuals"], input_center, input_scale, consequence_scale, device
    )
    calibration_o2, calibration_linear_prediction = evaluate_linear_atlas(
        calibration_records,
        linear_models,
        linear_center,
        linear_scale,
        neighbors,
        consequence_scale,
    )
    calibration_quant += calibration_o2
    nonlinear_quant, nonlinear_prediction = evaluate_predictor_atlases(
        calibration_records,
        families,
        random_members,
        input_center,
        input_scale,
        consequence_scale,
        device,
    )
    calibration_quant += nonlinear_quant
    calibration_prediction = calibration_linear_prediction + nonlinear_prediction
    deployable = (
        "B1_centered_covariance",
        "B2_phase_residual",
        "B3_dynamic_action_medoids",
        "B4_state_action_vq",
    )
    calibration_errors = {
        method: _mean_by(calibration_quant, "balanced_task_effect_error", (method,)) for method in deployable
    }
    strongest_baseline = min(calibration_errors, key=lambda name: (calibration_errors[name], name))
    nonlinear_errors = {
        method: _mean_by(calibration_quant, "balanced_task_effect_error", (method,))
        for method in ("NCEA", "MC_NCEA")
    }
    uncertainty_source = min(nonlinear_errors, key=lambda name: (nonlinear_errors[name], name))
    uncertainty_gate = build_uncertainty_gates(calibration_quant, uncertainty_source, strongest_baseline)
    contact_threshold_trace = []
    for threshold in CONTACT_CONFIDENCE_GRID:
        selected = [row for row in calibration_prediction if row["method"] == uncertainty_source]
        accuracy = float(
            np.mean(
                [
                    row["contact_correct"]
                    if row.get("contact_probability", 0.0) >= threshold
                    else 0
                    for row in selected
                ]
            )
        )
        contact_threshold_trace.append({"threshold": threshold, "calibration_accuracy_with_abstention_as_error": accuracy})
    contact_threshold = max(contact_threshold_trace, key=lambda row: (row["calibration_accuracy_with_abstention_as_error"], row["threshold"]))["threshold"]
    calibration_choices = {
        "created_utc": utc_now(),
        "development_results_accessed": False,
        "strongest_deployable_baseline": strongest_baseline,
        "deployable_baseline_errors": calibration_errors,
        "linear_choice": linear_choice,
        "linear_trace": linear_trace,
        "predictor_freeze_sha256": _array_hash(np.frombuffer(json.dumps(predictor_freeze, sort_keys=True).encode(), dtype=np.uint8)),
        "nonlinear_calibration_errors": nonlinear_errors,
        "uncertainty_gate": uncertainty_gate,
        "contact_confidence_threshold": contact_threshold,
        "contact_threshold_trace": contact_threshold_trace,
        "autoencoder": autoencoder_meta,
        "k": PRIMARY_K,
    }
    atomic_json(os.path.join(output_root, "work", "calibration_choices.json"), calibration_choices)
    write_csv(os.path.join(output_root, "work", "calibration_quantization.csv"), calibration_quant)
    write_csv(
        os.path.join(output_root, "work", "calibration_predictor_metrics_raw.csv"), calibration_prediction
    )

    # Development is loaded only after every fitting/calibration choice above is serialized.
    development_records = load_state_records(output_root, {"development"})
    nonlinearity_rows = support_nonlinearity_diagnostics(development_records, consequence_scale)
    development_quant = evaluate_continuous_upper_bound(development_records)
    development_quant += evaluate_action_baselines(development_records, codebooks, consequence_scale)
    development_quant += evaluate_action_autoencoder(
        development_records, autoencoder, codebooks["residuals"], input_center, input_scale, consequence_scale, device
    )
    development_quant += evaluate_true_oracle(development_records, consequence_scale)
    gate_a = _gate_a(development_quant)
    gate = {
        "created_utc": utc_now(),
        "sequential_testing": True,
        "gate_A": gate_a,
        "gate_B": {"status": "NOT_RUN_GATE_A_FAILED"},
        "gate_C": {"status": "NOT_RUN_GATE_A_FAILED"},
        "confirmation_unlocked": False,
    }
    predictor_rows = list(calibration_prediction)
    controls = []
    if not gate_a["passed"]:
        disposition = GATES["A"]["failure_disposition"]
    else:
        development_o2, development_linear_prediction = evaluate_linear_atlas(
            development_records,
            linear_models,
            linear_center,
            linear_scale,
            neighbors,
            consequence_scale,
        )
        development_quant += development_o2
        nonlinear_quant, nonlinear_prediction = evaluate_predictor_atlases(
            development_records,
            families,
            random_members,
            input_center,
            input_scale,
            consequence_scale,
            device,
        )
        development_quant += nonlinear_quant
        predictor_rows += development_linear_prediction + nonlinear_prediction
        development_quant += apply_uncertainty_gates(development_quant, uncertainty_gate)
        gate_b = _gate_b(development_quant, development_linear_prediction + nonlinear_prediction)
        gate["gate_B"] = gate_b
        gate["gate_C"] = {"status": "NOT_RUN_GATE_B_FAILED"}
        if not gate_b["passed"]:
            disposition = GATES["B"]["failure_disposition"]
        else:
            gate_c = _gate_c(development_quant, strongest_baseline)
            gate["gate_C"] = gate_c
            controls = gate_c["controls"]
            if not gate_c["passed"]:
                disposition = GATES["C"]["failure_disposition"]
            else:
                disposition = "DEVELOPMENT_PASSED_CONFIRMATION_REQUIRED"
                gate["confirmation_unlocked"] = True
    gate["development_disposition"] = disposition
    atomic_json(os.path.join(output_root, "work", "development_gate.json"), gate)
    write_csv(os.path.join(output_root, "development_quantization.csv"), development_quant)
    predictor_summary = _predictor_metric_summary(predictor_rows)
    write_csv(os.path.join(output_root, "predictor_metrics.csv"), predictor_summary)
    if not controls and gate_a["passed"]:
        best_available = min(
            ("NCEA", "MC_NCEA"),
            key=lambda method: _mean_by(development_quant, "balanced_task_effect_error", (method,)),
        )
        controls = _control_rows(development_quant, best_available, strongest_baseline)
    if not controls:
        controls = [
            {
                "split": "development",
                "status": "NOT_RUN_GATE_A_FAILED",
                "reason": "Sequential Gate A stopped deployable/control evaluation",
            }
        ]
    controls += mechanism_decomposition(development_quant, nonlinearity_rows)
    write_csv(os.path.join(output_root, "development_controls.csv"), controls)
    atomic_json(
        os.path.join(output_root, "work", "development_summary.json"),
        {
            "created_utc": utc_now(),
            "gate": gate,
            "quantization_summary": _quantization_summary(development_quant),
            "predictor_summary": predictor_summary,
            "support_nonlinearity": nonlinearity_rows,
            "calibration_choices": calibration_choices,
        },
    )
    if not gate["confirmation_unlocked"]:
        _empty_confirmation(output_root, disposition, gate)
    return {
        "development_disposition": disposition,
        "confirmation_unlocked": gate["confirmation_unlocked"],
        "gate_A": gate_a,
        "gate_B": gate["gate_B"],
        "gate_C": gate["gate_C"],
    }
