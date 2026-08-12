"""Numerical utilities with no scikit-learn dependency."""

from __future__ import annotations

import math

import numpy as np


EPS = 1e-12


def skew_to_vector(matrix):
    return np.asarray(
        [matrix[2, 1], matrix[0, 2], matrix[1, 0]],
        dtype=np.float64,
    )


def rotation_log(matrix):
    """Stable SO(3) logarithm as a three-vector."""
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    cos_theta = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-7:
        return skew_to_vector(0.5 * (matrix - matrix.T))
    if math.pi - theta < 1e-5:
        vals, vecs = np.linalg.eigh((matrix + np.eye(3)) * 0.5)
        axis = vecs[:, int(np.argmax(vals))]
        axis = axis / max(np.linalg.norm(axis), EPS)
        ref = skew_to_vector(matrix - matrix.T)
        if np.dot(axis, ref) < 0:
            axis = -axis
        return axis * theta
    return skew_to_vector(matrix - matrix.T) * (theta / max(2.0 * math.sin(theta), EPS))


def relative_rotation_log(current, reference):
    current = np.asarray(current, dtype=np.float64).reshape(3, 3)
    reference = np.asarray(reference, dtype=np.float64).reshape(3, 3)
    return rotation_log(current.dot(reference.T))


def deterministic_directions(dim, seed):
    """Return a deterministic orthonormal direction matrix, rows are directions."""
    rng = np.random.RandomState(int(seed))
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = q * signs[None, :]
    return q.T.astype(np.float64)


def robust_center_scale(values, mask=None):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must be 2D")
    if mask is None:
        mask = np.isfinite(values)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(values)
    center = np.zeros(values.shape[1], dtype=np.float64)
    scale = np.ones(values.shape[1], dtype=np.float64)
    for j in range(values.shape[1]):
        col = values[mask[:, j], j]
        if col.size == 0:
            continue
        med = float(np.median(col))
        mad = float(np.median(np.abs(col - med)))
        q25, q75 = np.percentile(col, [25.0, 75.0])
        robust = max(1.4826 * mad, float(q75 - q25) / 1.349, 1e-6)
        center[j] = med
        scale[j] = robust
    return center, scale


def ridge_jacobian(delta_a, delta_y, ridge, weights=None):
    """Fit Y = A J^T and return J with shape [y_dim, action_dim]."""
    a = np.asarray(delta_a, dtype=np.float64)
    y = np.asarray(delta_y, dtype=np.float64)
    if weights is not None:
        w = np.sqrt(np.asarray(weights, dtype=np.float64).reshape(-1, 1))
        a = a * w
        y = y * w
    lhs = a.T.dot(a) + float(ridge) * np.eye(a.shape[1])
    rhs = a.T.dot(y)
    return np.linalg.solve(lhs, rhs).T


def r2_score(y_true, y_pred, mask=None):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if mask is None:
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(y_true) & np.isfinite(y_pred)
    scores = []
    for j in range(y_true.shape[1]):
        keep = mask[:, j]
        if keep.sum() < 2:
            continue
        yt = y_true[keep, j]
        yp = y_pred[keep, j]
        denom = float(np.sum((yt - np.mean(yt)) ** 2))
        if denom <= EPS:
            continue
        scores.append(1.0 - float(np.sum((yt - yp) ** 2)) / denom)
    return float(np.mean(scores)) if scores else float("nan")


def rankdata(values):
    """Average ranks for ties, equivalent to scipy.stats.rankdata(method='average')."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x, dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearmanr(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return float("nan")
    rx, ry = rankdata(x[keep]), rankdata(y[keep])
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    return float(np.dot(rx, ry) / denom) if denom > EPS else float("nan")


def pairwise_squared_distances(x, centers):
    x = np.asarray(x, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    x2 = np.sum(x * x, axis=1)[:, None]
    c2 = np.sum(centers * centers, axis=1)[None, :]
    return np.maximum(x2 + c2 - 2.0 * x.dot(centers.T), 0.0)


def farthest_point_codebook(x, k, seed):
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] == 0:
        raise ValueError("empty training set")
    rng = np.random.RandomState(int(seed))
    first = int(rng.randint(x.shape[0]))
    chosen = [first]
    min_dist = pairwise_squared_distances(x, x[[first]])[:, 0]
    while len(chosen) < int(k):
        idx = int(np.argmax(min_dist))
        if min_dist[idx] <= EPS:
            idx = int(rng.randint(x.shape[0]))
        chosen.append(idx)
        min_dist = np.minimum(min_dist, pairwise_squared_distances(x, x[[idx]])[:, 0])
    return x[np.asarray(chosen, dtype=np.int64)].copy()


def kmeans(x, k, seed, max_iter=100, tolerance=1e-7):
    """Deterministic k-means++ / Lloyd implementation."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("x must be a non-empty matrix")
    rng = np.random.RandomState(int(seed))
    centers = np.empty((int(k), x.shape[1]), dtype=np.float64)
    centers[0] = x[int(rng.randint(x.shape[0]))]
    closest = pairwise_squared_distances(x, centers[:1])[:, 0]
    for i in range(1, int(k)):
        total = float(np.sum(closest))
        if total <= EPS:
            centers[i] = x[int(rng.randint(x.shape[0]))]
        else:
            target = float(rng.rand()) * total
            idx = int(np.searchsorted(np.cumsum(closest), target, side="right"))
            centers[i] = x[min(idx, x.shape[0] - 1)]
        closest = np.minimum(closest, pairwise_squared_distances(x, centers[i : i + 1])[:, 0])
    previous = float("inf")
    labels = np.zeros(x.shape[0], dtype=np.int64)
    for _ in range(int(max_iter)):
        distances = pairwise_squared_distances(x, centers)
        labels = np.argmin(distances, axis=1)
        inertia = float(np.sum(distances[np.arange(x.shape[0]), labels]))
        new = centers.copy()
        for j in range(int(k)):
            keep = labels == j
            if np.any(keep):
                new[j] = np.mean(x[keep], axis=0)
            else:
                farthest = int(np.argmax(np.min(distances, axis=1)))
                new[j] = x[farthest]
        centers = new
        if np.isfinite(previous) and previous - inertia <= float(tolerance) * max(previous, 1.0):
            break
        previous = inertia
    final_distances = pairwise_squared_distances(x, centers)
    labels = np.argmin(final_distances, axis=1)
    inertia = float(np.sum(final_distances[np.arange(x.shape[0]), labels]))
    return centers, labels, inertia


def pca_fit(x, max_rank=None):
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x, axis=0)
    _, singular, vh = np.linalg.svd(x - mean, full_matrices=False)
    rank = len(singular) if max_rank is None else min(int(max_rank), len(singular))
    return mean, vh[:rank], singular


def covariance_whitener(x, regularization=1e-6):
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x, axis=0)
    covariance = np.cov(x - mean, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, float(regularization))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    whitening = (vectors / np.sqrt(values)[None, :]).dot(vectors.T)
    dewhitening = (vectors * np.sqrt(values)[None, :]).dot(vectors.T)
    return mean, whitening, dewhitening, values


def truncated_pinv(matrix, relative_cutoff=1e-3, rank_cap=None):
    matrix = np.asarray(matrix, dtype=np.float64)
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    if s.size == 0 or s[0] <= EPS:
        return np.zeros((matrix.shape[1], matrix.shape[0])), np.zeros(0), 0, float("inf")
    keep = s >= float(relative_cutoff) * s[0]
    if rank_cap is not None:
        idx = np.flatnonzero(keep)[: int(rank_cap)]
        keep[:] = False
        keep[idx] = True
    rank = int(np.sum(keep))
    if rank == 0:
        keep[0] = True
        rank = 1
    inverse = (vh[keep].T / s[keep][None, :]).dot(u[:, keep].T)
    condition = float(s[keep][0] / max(s[keep][-1], EPS))
    return inverse, s, rank, condition


def metric_effective_rank(singular_values):
    s = np.asarray(singular_values, dtype=np.float64)
    energy = s * s
    if energy.sum() <= EPS:
        return 0.0
    p = energy / energy.sum()
    return float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))


def percentile_interval(values, alpha=0.05):
    values = np.asarray(values, dtype=np.float64)
    return [
        float(np.percentile(values, 100.0 * alpha * 0.5)),
        float(np.percentile(values, 100.0 * (1.0 - alpha * 0.5))),
    ]
