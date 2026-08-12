import numpy as np

from caaa_libero import config
from caaa_libero.analysis import (
    build_permutation_map,
    inverse_transformed_action,
    state_transform,
    transformed_action,
)
from caaa_libero.math_utils import truncated_pinv


def _fixture():
    records = []
    models = {}
    rng = np.random.RandomState(9)
    for episode in range(2):
        record = {
            "task_id": "fixture",
            "episode_id": episode,
            "split": "train",
            "phase": "free_space",
            "key": "fixture__e%02d__free_space" % episode,
        }
        j = rng.normal(size=(46, 24))
        pinv, singular, rank, condition = truncated_pinv(j, 1e-4)
        models[record["key"]] = {
            "j": j,
            "singular_values": singular,
            "projector": pinv.dot(j),
        }
        records.append(record)
    parameters = {
        "metric_regularization": 1e-6,
        "permutation_map": build_permutation_map(records),
        "covariance_mean": np.zeros(24),
        "covariance_whitener": np.eye(24) * 2.0,
        "covariance_dewhitener": np.eye(24) * 0.5,
        "pca_mean": np.zeros(24),
        "pca_components": np.eye(24)[:8],
    }
    return records, models, parameters


def test_invertible_transforms_round_trip():
    records, models, parameters = _fixture()
    x = np.linspace(-0.5, 0.5, config.CHUNK_CONTINUOUS_DIM)
    for method in ("covariance_mahalanobis", "old_diagonal_sensitivity", "random_spd", "caaa_v2"):
        z = transformed_action(method, x, records[0], models, parameters)
        reconstructed = inverse_transformed_action(method, z, records[0], models, parameters)
        assert np.allclose(reconstructed, x, atol=1e-7)


def test_controls_have_expected_shapes():
    records, models, parameters = _fixture()
    assert state_transform("old_diagonal_sensitivity", records[0], models, parameters).shape == (24, 24)
    assert state_transform("random_spd", records[0], models, parameters).shape == (24, 24)
    assert state_transform("permuted_j", records[0], models, parameters).shape == (70, 24)
    assert state_transform("caaa_v2", records[0], models, parameters).shape == (70, 24)
