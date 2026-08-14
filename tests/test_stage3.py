import csv

import numpy as np
import torch

from caaa_libero.stage3 import generate_support_codebooks, support_separation_evidence
from caaa_libero.stage3_metrics import ranking_metrics, stable_fps, write_csv
from caaa_libero.stage3_models import create_pair_ranker
from caaa_libero.stage3_data import HISTORY_CONTROL_SLICES, STATE_CONTROL_SLICES
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


def test_pair_ranker_is_exactly_symmetric_and_zero_on_self():
    torch.manual_seed(7)
    model = create_pair_ranker(11)
    context = torch.randn(5, 11)
    left = torch.randn(5, 24)
    right = torch.randn(5, 24)
    assert torch.equal(model(context, left, right), model(context, right, left))
    assert torch.equal(model(context, left, left), torch.zeros(5))


def test_fps_is_invariant_to_candidate_order_when_ids_are_frozen():
    rng = np.random.RandomState(8)
    values = rng.normal(size=(40, 6))
    ids = np.arange(40)
    expected = stable_fps(values, 12, frozen_ids=ids)
    permutation = rng.permutation(len(values))
    observed = stable_fps(values[permutation], 12, frozen_ids=ids[permutation])
    assert np.array_equal(expected, observed)


def test_ranking_metrics_are_exact_for_an_ideal_order():
    distance = np.linspace(0.0, 1.0, 256)
    metrics = ranking_metrics(distance, distance)
    assert metrics["pairwise_accuracy"] == 1.0
    assert abs(metrics["candidate_distance_spearman"] - 1.0) < 1e-12
    assert abs(metrics["kendall_tau"] - 1.0) < 1e-12
    assert abs(metrics["ndcg_at_16"] - 1.0) < 1e-12
    assert metrics["oracle_neighbor_recall_at_1"] == 1
    assert metrics["oracle_neighbor_recall_at_8"] == 1.0
    assert metrics["oracle_regret"] == 0.0


def test_stage3_csv_uses_frozen_compact_float_precision(tmp_path):
    path = tmp_path / "metrics.csv"
    write_csv(
        str(path),
        [{"metric": 0.12345678901234567, "bank_index": 17}],
    )
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["metric"] == "0.123456789012"
    assert row["bank_index"] == "17"


def test_shuffle_controls_include_semantic_availability_indicators():
    assert STATE_CONTROL_SLICES == ("state", "state_mask", "current_contact")
    assert HISTORY_CONTROL_SLICES == (
        "history",
        "history_mask",
        "previous_actions",
        "previous_action_mask",
    )
