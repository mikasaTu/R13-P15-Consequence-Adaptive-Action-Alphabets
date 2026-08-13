import numpy as np

from caaa_libero.stage2 import _array_hash, generate_direction_bank
from caaa_libero.stage2_config import (
    DIRECTION_COUNT,
    DIRECTION_FAMILY_COUNTS,
    PHASES,
    RADIUS_INTERVAL,
    consequence_metric_definition,
    split_for_episode,
)
from caaa_libero.stage2_analysis import _gate_a, _kmeans_medoids, balanced_error


def test_stage2_episode_splits_are_disjoint_and_historical_ids_are_rejected():
    assert split_for_episode(16) == "train"
    assert split_for_episode(24) == "calibration"
    assert split_for_episode(28) == "development"
    assert split_for_episode(32) == "confirmation"
    try:
        split_for_episode(15)
    except KeyError:
        pass
    else:
        raise AssertionError("historical episode entered Stage 2")


def test_direction_bank_is_deterministic_normalized_and_family_balanced():
    first = generate_direction_bank("bowl_on_plate", 16, PHASES[0], "train")
    second = generate_direction_bank("bowl_on_plate", 16, PHASES[0], "train")
    assert first["seed"] == second["seed"]
    np.testing.assert_array_equal(first["directions"], second["directions"])
    np.testing.assert_array_equal(first["radii"], second["radii"])
    assert first["directions"].shape == (DIRECTION_COUNT, 24)
    np.testing.assert_allclose(np.linalg.norm(first["directions"], axis=1), 1.0)
    assert np.max(np.abs(first["directions"])) <= 0.82
    assert np.min(first["radii"]) >= RADIUS_INTERVAL[0]
    assert np.max(first["radii"]) <= RADIUS_INTERVAL[1]
    for family, count in DIRECTION_FAMILY_COUNTS.items():
        assert first["families"].count(family) == count


def test_split_supports_do_not_share_exact_direction_or_residual_hashes():
    rows = [
        generate_direction_bank("plate_push", episode, "contact_onset", split)
        for episode, split in ((16, "train"), (24, "calibration"), (28, "development"), (32, "confirmation"))
    ]
    direction_sets = [{_array_hash(value) for value in row["directions"]} for row in rows]
    residual_sets = []
    for row in rows:
        values = set()
        for direction, radii in zip(row["directions"], row["radii"]):
            for radius in radii:
                for sign in (-1, 1):
                    values.add(_array_hash(sign * radius * direction))
        residual_sets.append(values)
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            assert direction_sets[left].isdisjoint(direction_sets[right])
            assert residual_sets[left].isdisjoint(residual_sets[right])


def test_balanced_metric_has_five_equal_groups_and_excludes_raw_force():
    metric = consequence_metric_definition()
    assert metric["primary"] == "BALANCED_TASK_EFFECT"
    assert len(metric["feature_groups"]) == 5
    assert set(metric["group_weights"].values()) == {0.2}
    predicted = set(metric["predicted_continuous_indices"])
    assert predicted.isdisjoint(metric["contact_force_indices_excluded_from_primary"])


def test_balanced_error_is_zero_on_identity_and_penalizes_mode_mismatch():
    target = np.zeros(46)
    mask = np.ones(46, dtype=bool)
    scale = np.ones(46)
    assert balanced_error(target, target, mask, mask, 0, 0, scale).item() == 0.0
    assert balanced_error(target, target, mask, mask, 0, 1, scale).item() > 0.0


def test_gate_a_uses_strongest_baseline_and_three_task_rule():
    rows = []
    for task_id in ("bowl_on_plate", "plate_push", "stove_turn_on", "wine_rack"):
        for method, error in (
            ("B1_centered_covariance", 1.0),
            ("B2_phase_residual", 0.9),
            ("B3_dynamic_action_medoids", 1.1),
            ("O1_true_effect_oracle", 0.7 if task_id != "wine_rack" else 1.0),
        ):
            rows.append(
                {
                    "task_id": task_id,
                    "episode_id": 28,
                    "phase": "free_space",
                    "target_id": 0,
                    "method": method,
                    "balanced_task_effect_error": error,
                }
            )
    gate = _gate_a(rows)
    assert gate["strongest_baseline"] == "B2_phase_residual"
    assert gate["tasks_improved"] == 3
    assert gate["passed"]


def test_kmeans_centers_map_to_unique_executable_medoids():
    values = np.random.RandomState(3).normal(size=(128, 5))
    selected = _kmeans_medoids(values, 16, 9)
    assert selected.shape == (16,)
    assert len(np.unique(selected)) == 16
    assert np.all((selected >= 0) & (selected < len(values)))
