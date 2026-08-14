import numpy as np

from caaa_libero.stage3 import generate_support_codebooks, support_separation_evidence
from caaa_libero.stage3_config import (
    CONFIRMATION_INTEGRITY_AMENDMENT,
    DIRECTION_FAMILY_COUNTS,
    EXECUTION_AMENDMENT,
    MAX_CROSS_SPLIT_ABS_COSINE,
    model_definitions,
    split_for_episode,
)


def test_stage3_episode_split_is_frozen():
    assert split_for_episode(15) == "historical"
    assert split_for_episode(16) == "train"
    assert split_for_episode(31) == "train"
    assert split_for_episode(32) == "calibration"
    assert split_for_episode(36) == "development"
    assert split_for_episode(40) == "confirmation"
    assert split_for_episode(49) == "confirmation"


def test_stage3_support_codebooks_are_deterministic_and_separated():
    left = generate_support_codebooks()
    right = generate_support_codebooks()
    for split in ("calibration", "development", "confirmation"):
        assert np.array_equal(left[split]["directions"], right[split]["directions"])
        assert np.array_equal(left[split]["radii"], right[split]["radii"])
        counts = np.bincount(left[split]["family_id"], minlength=3)
        assert counts.tolist() == list(DIRECTION_FAMILY_COUNTS.values())
    # An intentionally unrelated bank is sufficient to exercise all split
    # overlap/cosine checks without relying on a published artifact in unit tests.
    action_bank = np.eye(24, dtype=np.float64)
    action_bank = np.concatenate((action_bank, -action_bank), axis=0)
    action_bank = np.tile(action_bank, (6, 1))[:256] * 0.01
    evidence = support_separation_evidence(left, action_bank)
    assert not any(evidence["exact_direction_overlap"].values())
    assert not any(evidence["exact_residual_overlap"].values())
    assert max(evidence["maximum_cross_split_absolute_cosine_similarity"].values()) <= (
        MAX_CROSS_SPLIT_ABS_COSINE + 1e-12
    )


def test_stage3_models_exclude_privileged_phase_from_primary_inputs():
    definitions = model_definitions()
    assert "demonstration-derived future phase label" in definitions["forbidden_inputs"]
    assert definitions["candidate_permutation"][
        "required_exact_selected_bank_index_invariance"
    ]
    assert definitions["pair_construction"]["oracle_top_positives"] == 8


def test_stage3_user_amendment_preserves_scientific_labeling():
    assert EXECUTION_AMENDMENT["execute_all_development_methods_after_gate_failure"]
    assert EXECUTION_AMENDMENT["execute_episodes_40_49_after_gate_failure"]
    assert not EXECUTION_AMENDMENT["failed_gate_holdout_is_confirmatory"]
    assert not EXECUTION_AMENDMENT["failed_gate_holdout_can_unlock_small_bc"]
    assert not CONFIRMATION_INTEGRITY_AMENDMENT[
        "strict_untouched_confirmation_available"
    ]
    assert not CONFIRMATION_INTEGRITY_AMENDMENT["go_to_small_bc_available"]
