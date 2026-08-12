import numpy as np

from caaa_libero.math_utils import deterministic_directions, kmeans, ridge_jacobian, rotation_log


def test_rotation_log_and_directions():
    theta = 0.37
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(rotation_log(rotation), [0.0, 0.0, theta], atol=1e-10)
    directions = deterministic_directions(24, 13150015)
    assert np.allclose(directions.dot(directions.T), np.eye(24), atol=1e-12)


def test_ridge_and_kmeans_iterate():
    rng = np.random.RandomState(4)
    x = rng.normal(size=(96, 24))
    true_j = rng.normal(size=(46, 24))
    fit = ridge_jacobian(x, x.dot(true_j.T), 1e-10)
    assert np.max(np.abs(fit - true_j)) < 1e-8
    points = np.r_[rng.normal(-2.0, 0.1, (100, 2)), rng.normal(2.0, 0.1, (100, 2))]
    centers, labels, inertia = kmeans(points, 2, 7)
    assert inertia < 10.0
    assert len(np.unique(labels)) == 2

