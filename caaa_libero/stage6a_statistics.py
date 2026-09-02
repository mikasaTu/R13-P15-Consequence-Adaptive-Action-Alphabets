"""Registered Stage 6-A clustered bootstrap and disposition precedence."""

from __future__ import annotations

import numpy as np

from .stage6a_config import FINAL_DISPOSITIONS


def clustered_paired_bootstrap(left, right, episode_ids, replicates, seed):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    episode_ids = np.asarray(episode_ids).astype(str)
    if left.shape != right.shape or left.shape != episode_ids.shape:
        raise ValueError("bootstrap arrays must have identical shape")
    clusters = np.unique(episode_ids)
    by_cluster = {key: np.flatnonzero(episode_ids == key) for key in clusters}
    rng = np.random.RandomState(int(seed))
    values = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        indices = np.concatenate([by_cluster[key] for key in sampled])
        values[replicate] = float(np.mean(left[indices] - right[indices]))
    return values


def choose_disposition(
    history_ok,
    data_ok,
    defects_ok,
    gate_h_ok,
    density_specific=None,
    gate_a_ok=None,
):
    if not history_ok:
        result = "BLOCKED_HISTORICAL_BINDING_MISMATCH"
    elif not data_ok:
        result = "BLOCKED_NO_EXECUTED_CANDIDATE_CACHE"
    elif not defects_ok:
        result = "BLOCKED_DEFECT_NOT_REPRODUCED"
    elif not gate_h_ok:
        result = "QUANTIZER_STILL_DEGENERATE"
    elif density_specific is False:
        result = "GAIN_NOT_DENSITY_SPECIFIC"
    elif gate_a_ok is True:
        result = "REPAIR_CONFIRMED_ADVANCE_TO_STAGE6B"
    else:
        result = "C4_REMOVAL_INSUFFICIENT"
    if result not in FINAL_DISPOSITIONS:
        raise AssertionError(result)
    return result
