"""C4-free Stage 6-A C3 alphabet construction and selection.

The module is intentionally small so absence of the Stage 3 pair ranker is
machine-auditable from its import and call graph.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from .stage3_models import create_biencoder, embed_actions
from .stage5_oracle import _assign, deterministic_kmedoids_precomputed
from .storage import sha256_file


def load_c3_ensemble(stage3_root, device):
    import torch

    with open(
        os.path.join(stage3_root, "trained_model_registry.json"),
        "r",
        encoding="utf-8",
    ) as handle:
        registry = json.load(handle)
    with np.load(
        os.path.join(stage3_root, registry["scalers"]), allow_pickle=False
    ) as data:
        center = np.asarray(data["context_center"], dtype=np.float32)
        scale = np.asarray(data["context_scale"], dtype=np.float32)
    models, checkpoints = [], []
    for relative in registry["models"]["C3_NC_BIENCODER"]["members"]:
        path = os.path.join(stage3_root, relative)
        payload = torch.load(path, map_location=device, weights_only=True)
        model = create_biencoder(len(center)).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models.append(model)
        checkpoints.append({"path": relative, "sha256": sha256_file(path)})
    return models, center, scale, checkpoints


def stage3_context_from_stage5(cache, center, scale):
    cached = np.asarray(cache["context"], dtype=np.float32)
    raw = (
        cached * np.asarray(cache["context_scale"], dtype=np.float32)[None, :]
        + np.asarray(cache["context_center"], dtype=np.float32)[None, :]
    )
    return ((raw - center[None, :]) / scale[None, :]).astype(np.float32)


def ensemble_embeddings(models, context, residuals, device):
    values = [embed_actions(model, context, residuals, device) for model in models]
    return np.concatenate(values, axis=1) / math.sqrt(float(len(values)))


def squared_distance(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=-1)


def build_c3_kmedoids_atlas(candidate_embedding, source_ids, k):
    distance = squared_distance(candidate_embedding, candidate_embedding)
    distance = 0.5 * (distance + distance.T)
    np.fill_diagonal(distance, 0.0)
    return deterministic_kmedoids_precomputed(distance, int(k), source_ids)


def select_c3_only(target_embedding, candidate_embedding, atlas, source_ids):
    score = squared_distance(target_embedding, candidate_embedding)
    selected = _assign(score, np.asarray(atlas, dtype=np.int64), source_ids)
    return selected.astype(np.int64), score
