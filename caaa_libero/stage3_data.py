"""Read, align, and featurize frozen Stage 3 simulator shards."""

from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np

from .env_adapter import FEATURE_NAMES
from .stage2_analysis import (
    CONTACT_MODE_COUNT,
    CONTINUOUS_INDICES,
    PHASE_TO_ID,
    TASK_TO_ID,
    balanced_error,
    fit_train_scaling,
)
from .stage3_collection import (
    context_shard,
    resolved_candidate_shard,
    resolved_support_shard,
)
from .stage3_config import (
    ACTION_BANK_SIZE,
    CONTINUOUS_DIM,
    PAIR_CONTACT_CHANGE_MAX,
    PAIR_HARD_NEGATIVE_END,
    PAIR_HARD_NEGATIVE_START,
    PAIR_RANDOM_NEGATIVES,
    PAIR_TOP_POSITIVES,
    PHASES,
    SCRATCH_ROOT,
    TASK_IDS,
)
from .storage import atomic_json, validate_complete


CONTEXT_SLICES = {
    "state": (0, 46),
    "state_mask": (46, 92),
    "history": (92, 184),
    "history_mask": (184, 276),
    "previous_actions": (276, 290),
    "previous_action_mask": (290, 292),
    "current_contact": (292, 293),
    "nominal_action": (293, 317),
    "task_one_hot": (317, 321),
}
CONTEXT_DIM = 321


def _seed(*parts):
    value = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little", signed=False)


def _load_npz_checked(path):
    valid, evidence = validate_complete(path)
    if not valid:
        raise RuntimeError("incomplete Stage 3 shard %s: %s" % (path, evidence))
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]).copy() for name in data.files}


def load_records(project_root, output_root, splits, scratch_root=SCRATCH_ROOT):
    requested = set(splits)
    with open(os.path.join(output_root, "episode_split.json"), "r", encoding="utf-8") as handle:
        metadata = json.load(handle)["snapshots"]
    records = []
    for meta in metadata:
        if meta["split"] not in requested:
            continue
        task_id = meta["task_id"]
        episode_id = int(meta["episode_id"])
        phase = meta["phase"]
        support_path = resolved_support_shard(
            project_root, scratch_root, meta["split"], task_id, episode_id, phase
        )
        candidate_path = resolved_candidate_shard(
            project_root, scratch_root, meta["split"], task_id, episode_id, phase
        )
        context_path = context_shard(scratch_root, task_id, episode_id, phase)
        support = _load_npz_checked(support_path)
        candidate = _load_npz_checked(candidate_path)
        context = _load_npz_checked(context_path)
        if support["residual_action"].shape[1] != CONTINUOUS_DIM:
            raise RuntimeError("support action semantics changed")
        if candidate["residual_action"].shape != (ACTION_BANK_SIZE + 1, CONTINUOUS_DIM):
            raise RuntimeError("candidate bank semantics changed")
        if not np.array_equal(candidate["bank_index"][1:], np.arange(ACTION_BANK_SIZE)):
            raise RuntimeError("candidate bank order changed")
        difference = float(
            np.max(
                np.abs(
                    np.asarray(context["observable_state"], dtype=np.float64)
                    - np.asarray(support["initial"][0], dtype=np.float64)
                )
            )
        )
        if difference > 1e-12:
            raise RuntimeError("context/support snapshot mismatch %.17g for %s" % (difference, meta["key"]))
        if not np.array_equal(context["observable_mask"], support["mask"][0]):
            raise RuntimeError("context/support mask mismatch for " + meta["key"])
        records.append(
            {
                "meta": dict(meta),
                "support": support,
                "candidate": candidate,
                "context": context,
                "support_path": support_path,
                "candidate_path": candidate_path,
                "context_path": context_path,
            }
        )
    records.sort(
        key=lambda row: (
            TASK_TO_ID[row["meta"]["task_id"]],
            int(row["meta"]["episode_id"]),
            PHASE_TO_ID[row["meta"]["phase"]],
        )
    )
    return records


def effect(shard):
    return np.asarray(shard["settled"] - shard["settled"][[0]], dtype=np.float64)


def raw_context(record, include_phase=False):
    value = record["context"]
    state = np.asarray(value["observable_state"], dtype=np.float64)
    state_mask = np.asarray(value["observable_mask"], dtype=np.float64)
    history = np.asarray(value["history_delta"], dtype=np.float64).reshape(-1)
    history_mask = np.asarray(value["history_delta_mask"], dtype=np.float64).reshape(-1)
    previous_actions = np.asarray(value["previous_action"], dtype=np.float64).reshape(-1)
    previous_action_mask = np.asarray(value["previous_action_mask"], dtype=np.float64).reshape(-1)
    current_contact = np.asarray([float(value["current_contact"].item())], dtype=np.float64)
    nominal = np.asarray(value["nominal_continuous"], dtype=np.float64).reshape(-1)
    task = np.eye(len(TASK_IDS), dtype=np.float64)[TASK_TO_ID[record["meta"]["task_id"]]]
    output = np.concatenate(
        (
            state,
            state_mask,
            history,
            history_mask,
            previous_actions,
            previous_action_mask,
            current_contact,
            nominal,
            task,
        )
    )
    if output.shape != (CONTEXT_DIM,):
        raise AssertionError(output.shape)
    if include_phase:
        phase = np.eye(len(PHASES), dtype=np.float64)[PHASE_TO_ID[record["meta"]["phase"]]]
        output = np.concatenate((output, phase))
    return output


def fit_context_scaler(records):
    values = np.stack([raw_context(record) for record in records])
    center = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - center[None, :]), axis=0)
    # Binary masks/indicators and task identity keep their exact semantics.
    for name in (
        "state_mask",
        "history_mask",
        "previous_action_mask",
        "current_contact",
        "task_one_hot",
    ):
        start, stop = CONTEXT_SLICES[name]
        center[start:stop] = 0.0
        scale[start:stop] = 1.0
    scale = np.maximum(scale, 1e-3)
    return center.astype(np.float32), scale.astype(np.float32)


def normalized_context(record, center, scale):
    return ((raw_context(record) - center) / scale).astype(np.float32)


def transformed_contexts(records, center, scale, control=None, seed=13150300):
    raw = np.stack([raw_context(record) for record in records])
    if control:
        rng = np.random.RandomState(_seed(seed, control))
        tasks = np.asarray([TASK_TO_ID[row["meta"]["task_id"]] for row in records])
        order = np.arange(len(records), dtype=np.int64)
        for task_id in sorted(set(tasks.tolist())):
            indices = np.flatnonzero(tasks == task_id)
            order[indices] = rng.permutation(indices)
        if control == "no_nominal_action":
            start, stop = CONTEXT_SLICES["nominal_action"]
            raw[:, start:stop] = 0.0
        elif control == "nominal_action_shuffled_within_task":
            start, stop = CONTEXT_SLICES["nominal_action"]
            raw[:, start:stop] = raw[order, start:stop]
        elif control == "state_shuffled_within_task":
            start, stop = CONTEXT_SLICES["state"]
            raw[:, start:stop] = raw[order, start:stop]
        elif control == "joint_state_nominal_shuffled_within_task":
            for name in ("state", "nominal_action"):
                start, stop = CONTEXT_SLICES[name]
                raw[:, start:stop] = raw[order, start:stop]
        elif control == "history_shuffled":
            for name in ("history", "previous_actions"):
                start, stop = CONTEXT_SLICES[name]
                raw[:, start:stop] = raw[order, start:stop]
        elif control in ("consequence_labels_shuffled", "action_only_pair_ranker"):
            if control == "action_only_pair_ranker":
                raw[:, :] = 0.0
        else:
            raise KeyError(control)
    return ((raw - center[None, :]) / scale[None, :]).astype(np.float32)


def fit_scales(records, output_root=None):
    consequence_scale, consequence_evidence = fit_train_scaling(records)
    context_center, context_scale = fit_context_scaler(records)
    if output_root:
        atomic_json(
            os.path.join(output_root, "scaling_evidence.json"),
            {
                "consequence": consequence_evidence,
                "context_center": context_center.tolist(),
                "context_scale": context_scale.tolist(),
                "context_slices": {name: list(value) for name, value in CONTEXT_SLICES.items()},
                "fit_split": "train episodes 16-31 only",
            },
        )
    return consequence_scale, context_center, context_scale


def build_branch_dataset(records, consequence_scale, context_center, context_scale, support_only=False):
    contexts = transformed_contexts(records, context_center, context_scale)
    state_index = []
    residual = []
    target = []
    target_mask = []
    contact = []
    source = []
    for record_index, record in enumerate(records):
        shards = ((0, record["support"]),) if support_only else (
            (0, record["support"]),
            (1, record["candidate"]),
        )
        for source_id, shard in shards:
            n = len(shard["residual_action"]) - 1
            state_index.extend([record_index] * n)
            residual.append(np.asarray(shard["residual_action"][1:], dtype=np.float64))
            normalized = effect(shard)[1:][:, CONTINUOUS_INDICES]
            normalized = normalized / consequence_scale[CONTINUOUS_INDICES][None, :]
            target.append(normalized)
            target_mask.append(np.asarray(shard["mask"][1:][:, CONTINUOUS_INDICES], dtype=bool))
            contact.append(np.asarray(shard["contact_mode"][1:], dtype=np.int64))
            source.extend([source_id] * n)
    return {
        "contexts": contexts,
        "state_index": np.asarray(state_index, dtype=np.int32),
        "residual": np.concatenate(residual).astype(np.float32),
        "target": np.concatenate(target).astype(np.float32),
        "target_mask": np.concatenate(target_mask),
        "contact": np.concatenate(contact).astype(np.int64),
        "source": np.asarray(source, dtype=np.int8),
    }


def true_distance_matrix(record, consequence_scale):
    target_effect = effect(record["support"])[1:]
    target_mask = np.asarray(record["support"]["mask"][1:], dtype=bool)
    target_mode = np.asarray(record["support"]["contact_mode"][1:], dtype=np.int64)
    candidate_effect = effect(record["candidate"])[1:]
    candidate_mask = np.asarray(record["candidate"]["mask"][1:], dtype=bool)
    candidate_mode = np.asarray(record["candidate"]["contact_mode"][1:], dtype=np.int64)
    output = np.empty((len(target_effect), len(candidate_effect)), dtype=np.float32)
    for target_id in range(len(target_effect)):
        output[target_id] = balanced_error(
            np.repeat(target_effect[target_id][None, :], len(candidate_effect), axis=0),
            candidate_effect,
            np.repeat(target_mask[target_id][None, :], len(candidate_effect), axis=0),
            candidate_mask,
            np.full(len(candidate_effect), target_mode[target_id], dtype=np.int64),
            candidate_mode,
            consequence_scale,
        ).astype(np.float32)
    return output


def _selected_candidate_ids(distances, target_mode, candidate_modes, seed):
    order = np.argsort(distances, kind="mergesort")
    selected = []
    category = {}

    def add(indices, label):
        for index in indices:
            index = int(index)
            if index not in category:
                selected.append(index)
                category[index] = label

    add(order[:PAIR_TOP_POSITIVES], "oracle_top8")
    add(
        order[PAIR_HARD_NEGATIVE_START - 1 : PAIR_HARD_NEGATIVE_END],
        "hard_rank9_32",
    )
    remaining = order[PAIR_HARD_NEGATIVE_END:]
    rng = np.random.RandomState(int(seed))
    random_ids = rng.choice(
        remaining,
        size=min(PAIR_RANDOM_NEGATIVES, len(remaining)),
        replace=False,
    )
    add(random_ids, "random")
    changed = order[np.asarray(candidate_modes[order] != int(target_mode), dtype=bool)]
    add(changed[:PAIR_CONTACT_CHANGE_MAX], "contact_change")
    return np.asarray(selected, dtype=np.int16), [category[index] for index in selected]


def build_pair_dataset(records, consequence_scale, context_center, context_scale, label_shuffle=False):
    contexts = transformed_contexts(records, context_center, context_scale)
    maximum = (
        PAIR_TOP_POSITIVES
        + (PAIR_HARD_NEGATIVE_END - PAIR_HARD_NEGATIVE_START + 1)
        + PAIR_RANDOM_NEGATIVES
        + PAIR_CONTACT_CHANGE_MAX
    )
    state_indices = []
    target_ids = []
    target_residuals = []
    candidate_ids = []
    candidate_residuals = []
    distances = []
    masks = []
    categories = []
    metadata = []
    for state_index, record in enumerate(records):
        matrix = true_distance_matrix(record, consequence_scale)
        target_mode = np.asarray(record["support"]["contact_mode"][1:], dtype=np.int64)
        candidate_mode = np.asarray(record["candidate"]["contact_mode"][1:], dtype=np.int64)
        targets = np.asarray(record["support"]["residual_action"][1:], dtype=np.float32)
        bank = np.asarray(record["candidate"]["residual_action"][1:], dtype=np.float32)
        for target_id in range(len(targets)):
            selected, labels = _selected_candidate_ids(
                matrix[target_id],
                target_mode[target_id],
                candidate_mode,
                _seed(
                    13150300,
                    record["meta"]["task_id"],
                    record["meta"]["episode_id"],
                    record["meta"]["phase"],
                    target_id,
                ),
            )
            count = len(selected)
            ids = np.full(maximum, -1, dtype=np.int16)
            ids[:count] = selected
            candidate = np.zeros((maximum, CONTINUOUS_DIM), dtype=np.float32)
            candidate[:count] = bank[selected]
            distance = np.zeros(maximum, dtype=np.float32)
            distance[:count] = matrix[target_id, selected]
            mask = np.zeros(maximum, dtype=bool)
            mask[:count] = True
            category = np.full(maximum, "padding", dtype="U18")
            category[:count] = np.asarray(labels)
            state_indices.append(state_index)
            target_ids.append(target_id)
            target_residuals.append(targets[target_id])
            candidate_ids.append(ids)
            candidate_residuals.append(candidate)
            distances.append(distance)
            masks.append(mask)
            categories.append(category)
            metadata.append(
                {
                    "task_id": record["meta"]["task_id"],
                    "episode_id": int(record["meta"]["episode_id"]),
                    "split": record["meta"]["split"],
                    "phase": record["meta"]["phase"],
                    "target_id": int(target_id),
                    "direction_id": int(record["support"]["direction_id"][target_id + 1]),
                    "direction_family_id": int(
                        record["support"]["direction_family_id"][target_id + 1]
                    ),
                    "radius": float(record["support"]["radius"][target_id + 1]),
                    "sign": int(record["support"]["sign"][target_id + 1]),
                    "target_mode": int(target_mode[target_id]),
                    "candidate_modes": candidate_mode[selected].astype(np.int8),
                }
            )
    distance_array = np.asarray(distances, dtype=np.float32)
    if label_shuffle:
        rng = np.random.RandomState(_seed(13150300, "consequence_labels_shuffled"))
        state_values = np.asarray(state_indices, dtype=np.int32)
        for state_id in sorted(set(state_values.tolist())):
            keep = np.flatnonzero(state_values == state_id)
            distance_array[keep] = distance_array[rng.permutation(keep)]
    return {
        "contexts": contexts,
        "state_index": np.asarray(state_indices, dtype=np.int32),
        "target_id": np.asarray(target_ids, dtype=np.int16),
        "target_residual": np.asarray(target_residuals, dtype=np.float32),
        "candidate_id": np.asarray(candidate_ids, dtype=np.int16),
        "candidate_residual": np.asarray(candidate_residuals, dtype=np.float32),
        "true_distance": distance_array,
        "candidate_mask": np.asarray(masks, dtype=bool),
        "category": np.asarray(categories),
        "metadata": metadata,
    }


def write_training_pairs(path, pair_dataset):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = {
        "task_id": [],
        "episode_id": [],
        "phase": [],
        "target_id": [],
        "direction_id": [],
        "direction_family_id": [],
        "radius": [],
        "sign": [],
        "target_mode": [],
        "candidate_bank_index": [],
        "candidate_mode": [],
        "category": [],
        "true_balanced_effect_distance": [],
    }
    for group_index, meta in enumerate(pair_dataset["metadata"]):
        valid = np.flatnonzero(pair_dataset["candidate_mask"][group_index])
        modes = np.asarray(meta["candidate_modes"], dtype=np.int8)
        for local_index, position in enumerate(valid):
            rows["task_id"].append(meta["task_id"])
            rows["episode_id"].append(meta["episode_id"])
            rows["phase"].append(meta["phase"])
            rows["target_id"].append(meta["target_id"])
            rows["direction_id"].append(meta["direction_id"])
            rows["direction_family_id"].append(meta["direction_family_id"])
            rows["radius"].append(meta["radius"])
            rows["sign"].append(meta["sign"])
            rows["target_mode"].append(meta["target_mode"])
            rows["candidate_bank_index"].append(
                int(pair_dataset["candidate_id"][group_index, position])
            )
            rows["candidate_mode"].append(int(modes[local_index]))
            rows["category"].append(str(pair_dataset["category"][group_index, position]))
            rows["true_balanced_effect_distance"].append(
                float(pair_dataset["true_distance"][group_index, position])
            )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table = pa.table(rows)
    pq.write_table(table, path, compression="zstd", row_group_size=65536)
    return {"rows": int(table.num_rows), "bytes": int(os.path.getsize(path))}


def routing_labels(records):
    labels = []
    for record in records:
        context = record["context"]
        contact = int(bool(context["current_contact"].item()))
        progress_index = FEATURE_NAMES.index("task_progress")
        delta = float(context["history_delta"][0, progress_index])
        labels.append(2 * contact + int(delta >= 0.0))
    return np.asarray(labels, dtype=np.int64)
