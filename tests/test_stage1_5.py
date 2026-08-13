import numpy as np

from caaa_libero import config
from caaa_libero.stage1_5 import (
    METHOD_M0,
    METHOD_M1,
    METHODS_REVISED,
    ORACLE_O1,
    ORACLE_O2,
    build_permutation_map,
    internal_screen,
    project_box_ball,
    solve_constrained_ridge,
)


def test_box_ball_projection_satisfies_both_constraints():
    value = np.asarray([4.0, -3.0, 0.5])
    lower = np.asarray([-0.2, -0.7, -0.4])
    upper = np.asarray([0.8, 0.1, 0.6])
    projected = project_box_ball(value, lower, upper, 0.75)
    assert np.all(projected >= lower - 1e-12)
    assert np.all(projected <= upper + 1e-12)
    assert np.linalg.norm(projected) <= 0.75 + 1e-10


def test_constrained_ridge_identity_hits_radius_boundary():
    target = np.asarray([1.0, 1.0])
    decoded, residual, objective, iterations = solve_constrained_ridge(
        np.eye(2), target, 0.0, np.zeros(2), 0.5
    )
    expected = target / np.linalg.norm(target) * 0.5
    assert np.allclose(decoded, expected, atol=1e-7)
    assert residual > 0.0
    assert objective > 0.0
    assert iterations > 0


def test_permutation_stays_within_task_split_phase():
    records = []
    for episode in range(3):
        records.append(
            {
                "task_id": "task",
                "split": "test",
                "phase": "contact_onset",
                "episode_id": episode,
                "key": "key%d" % episode,
            }
        )
    mapping = build_permutation_map(records)
    assert mapping == {"key0": "key1", "key1": "key2", "key2": "key0"}


def _screen_rows(control_error):
    errors = {
        METHOD_M0: 150.0,
        METHOD_M1: 100.0,
        METHODS_REVISED[0]: 80.0,
        METHODS_REVISED[1]: 110.0,
        METHODS_REVISED[2]: 80.0,
        METHODS_REVISED[3]: 90.0,
        METHODS_REVISED[4]: control_error,
        METHODS_REVISED[5]: control_error,
        ORACLE_O1: 70.0,
        ORACLE_O2: 80.0,
    }
    rows = []
    for task in config.TASKS:
        for method, error in errors.items():
            rows.append(
                {
                    "evidence_set": "old_test_internal_screen",
                    "task_id": task["task_id"],
                    "episode_id": 12,
                    "phase": "contact_onset",
                    "method": method,
                    "k": 64,
                    "code_index": 0,
                    "settled_effect_error": error,
                    "immediate_effect_error": error,
                    "contact_preserved": True,
                    "progress_preserved_005": True,
                    "progress_absolute_error": 0.0,
                    "action_reconstruction_error": 0.1,
                    "clipped_coordinates": 24 if method == METHOD_M0 else 0,
                    "infeasible_assignment": False,
                    "state_infeasible_token_rate": 0.0,
                    "solver_latency_ms": 0.0,
                }
            )
    return rows


def test_internal_screen_requires_geometry_controls_not_to_reproduce_gain():
    passing = internal_screen(_screen_rows(control_error=99.0))
    assert METHODS_REVISED[0] in passing["passing_methods"]
    reproduced = internal_screen(_screen_rows(control_error=80.0))
    assert reproduced["passing_methods"] == []
    assert reproduced["decision"] == "REJECT_P15_FAMILY"
