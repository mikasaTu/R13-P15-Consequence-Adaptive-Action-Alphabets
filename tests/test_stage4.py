import numpy as np

from caaa_libero.stage4_config import (
    FINAL_DISPOSITIONS,
    SUPPORT_DIRECTION_COUNT,
    SUPPORT_DIRECTION_FAMILIES,
    SUPPORT_TARGET_COUNT,
    TRAIN_STATE_COUNT,
    method_definitions,
)
from caaa_libero.stage4_freeze import (
    _phase_windows,
    _valid_action_chunk,
    generate_training_support_bank,
)


def test_stage4_support_bank_is_balanced_deterministic_and_antithetic():
    left = generate_training_support_bank()
    right = generate_training_support_bank()
    assert left["directions"].shape == (SUPPORT_DIRECTION_COUNT, 24)
    assert left["residuals"].shape == (SUPPORT_TARGET_COUNT, 24)
    assert np.array_equal(left["directions"], right["directions"])
    assert np.array_equal(left["residuals"], right["residuals"])
    counts = np.bincount(
        left["residual_family_id"], minlength=len(SUPPORT_DIRECTION_FAMILIES)
    )
    assert len(set(counts.tolist())) == 1
    residuals = {tuple(np.round(row, 14)) for row in left["residuals"]}
    assert all(tuple(np.round(-row, 14)) in residuals for row in left["residuals"])
    assert np.allclose(np.linalg.norm(left["directions"], axis=1), 1.0)


def test_stage4_phase_windows_cover_episode_without_overlap():
    windows = _phase_windows(
        {
            "free_space": 10,
            "pre_contact": 30,
            "contact_onset": 38,
            "post_contact": 70,
        },
        90,
    )
    covered = []
    for low, high in windows.values():
        covered.extend(range(low, high + 1))
    assert sorted(covered) == list(range(91))
    assert len(covered) == len(set(covered))


def test_stage4_action_validity_enforces_no_clipping():
    actions = np.zeros((8, 7), dtype=np.float64)
    bank = np.zeros((256, 24), dtype=np.float64)
    supports = np.zeros((96, 24), dtype=np.float64)
    valid, chunk = _valid_action_chunk(actions, 1, bank, supports)
    assert valid
    assert chunk.shape == (24,)
    actions[1, 0] = 0.9
    valid, _ = _valid_action_chunk(actions, 1, bank, supports)
    assert not valid


def test_stage4_method_contract_forbids_pai_policy_and_synthesis():
    methods = method_definitions()
    assert methods["pai_jobs_allowed"] is False
    assert methods["policy_training_allowed"] is False
    assert methods["maximum_local_training_gpus"] == 1
    assert methods["trust_region"]["outputs_executable_bank_member"] is True
    assert methods["trust_region"]["clipping_or_action_synthesis_forbidden"] is True
    assert methods["execute_all_experiments_after_gate_failure"] is True
    assert methods["final_dispositions"] == list(FINAL_DISPOSITIONS)
    assert TRAIN_STATE_COUNT == 768
