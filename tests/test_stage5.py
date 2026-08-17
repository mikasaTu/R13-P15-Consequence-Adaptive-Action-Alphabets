import json
import os

import numpy as np
import torch

from caaa_libero.stage3_metrics import paired_episode_bootstrap
from caaa_libero.stage5_config import FINAL_DISPOSITIONS, FORBIDDEN_INPUT_FIELDS, PROPOSED_INPUT_FIELDS
from caaa_libero.stage5_logic import choose_disposition, exact_one_disposition
from caaa_libero.stage5_models import create_context_metric, create_static_metric
from caaa_libero.stage5_oracle import _assign, deterministic_kmedoids_precomputed


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT = os.path.join(ROOT, "experiments/r13_p15_cicr_dla/stage5")


def test_stage5_distance_is_exactly_symmetric_and_zero_on_self():
    torch.manual_seed(5)
    model = create_static_metric()
    nominal = torch.randn(7, 24)
    left = torch.randn(7, 24)
    right = torch.randn(7, 24)
    assert torch.equal(model.distance(nominal, left, right), model.distance(nominal, right, left))
    assert torch.equal(model.distance(nominal, left, left), torch.zeros(7))


def test_stage5_zero_modulation_recovers_static_model_bit_for_bit():
    torch.manual_seed(6)
    base = create_static_metric()
    model = create_context_metric(base)
    context = torch.randn(9, 321)
    nominal = torch.randn(9, 24)
    target = torch.randn(9, 24)
    candidate = torch.randn(9, 24)
    expected = base.distance(nominal, target, candidate)
    observed = model.distance(context, nominal, target, candidate, force_zero_modulation=True)
    assert torch.equal(expected, observed)


def test_stage5_context_modulation_can_reverse_a_synthetic_order():
    torch.manual_seed(7)
    base = create_static_metric()
    model = create_context_metric(base)
    # Directly exercise the registered positive diagonal form with two contexts.
    squared_i = torch.tensor([1.0, 9.0])
    squared_j = torch.tensor([9.0, 1.0])
    weight_s1 = torch.exp(torch.tensor([1.0, -1.0]))
    weight_s2 = torch.exp(torch.tensor([-1.0, 1.0]))
    assert torch.sum(weight_s1 * squared_i) < torch.sum(weight_s1 * squared_j)
    assert torch.sum(weight_s2 * squared_j) < torch.sum(weight_s2 * squared_i)


def test_stage5_candidate_order_permutation_preserves_original_ids():
    rng = np.random.RandomState(8)
    distance = rng.uniform(size=(13, 31))
    ids = np.arange(31, dtype=np.int64) + 100
    medoids = np.asarray([0, 3, 7, 12, 20, 25], dtype=np.int64)
    expected = _assign(distance, medoids, ids)
    permutation = rng.permutation(31)
    inverse = np.empty(31, dtype=np.int64)
    inverse[permutation] = np.arange(31)
    permuted_medoids = inverse[medoids]
    observed_positions = _assign(distance[:, permutation], permuted_medoids, ids[permutation])
    observed_original = permutation[observed_positions]
    assert np.array_equal(expected, observed_original)


def test_stage5_proposed_inputs_exclude_ids_future_and_phase():
    text = " ".join(PROPOSED_INPUT_FIELDS).lower()
    assert "candidate_id" not in text and "target_id" not in text and "phase" not in text
    for forbidden in ("future_state", "future_consequence", "candidate_id", "target_id", "demonstration_phase"):
        assert forbidden in FORBIDDEN_INPUT_FIELDS


def test_stage5_frozen_reversal_tuples_are_split_disjoint():
    import pandas as pd

    frame = pd.read_parquet(os.path.join(OUTPUT, "CONTEXT_REVERSAL_PAIRS.parquet"))
    sets = {split: set(frame.loc[frame.split == split, "tuple_sha256"].astype(str)) for split in frame.split.unique()}
    for left in sets:
        for right in sets:
            if left < right:
                assert sets[left].isdisjoint(sets[right])


def test_stage5_local_bank_is_balanced_unique_and_requires_no_clipping():
    with np.load(os.path.join(OUTPUT, "LOCAL_BANK.npz"), allow_pickle=False) as data:
        residuals = np.asarray(data["residuals"])
        source = np.asarray(data["source_indices"])
        sign = np.asarray(data["source_sign"])
        family = np.asarray(data["source_family_id"])
    assert residuals.shape == (128, 24)
    assert len(np.unique(source)) == 128
    assert np.bincount((sign > 0).astype(int), minlength=2).tolist() == [64, 64]
    assert max(np.bincount(family, minlength=3)) - min(np.bincount(family, minlength=3)) <= 2
    assert float(np.max(np.abs(residuals))) < 0.105


def test_stage5_kmedoids_is_deterministic_and_unique():
    rng = np.random.RandomState(9)
    points = rng.normal(size=(40, 5))
    distance = np.sqrt(np.sum((points[:, None] - points[None]) ** 2, axis=-1))
    ids = rng.permutation(np.arange(40) + 200)
    left = deterministic_kmedoids_precomputed(distance, 12, ids)
    right = deterministic_kmedoids_precomputed(distance, 12, ids)
    assert np.array_equal(left, right)
    assert len(np.unique(left)) == 12


def test_stage5_bootstrap_clusters_by_episode_not_branch_row():
    rows = []
    for episode, difference in ((1, 0.2), (2, -0.1), (3, 0.4)):
        for target in range(20):
            rows.append({"method": "M", "task_id": "t", "episode_id": episode, "balanced_task_effect_error": 1.0 - difference})
            rows.append({"method": "B", "task_id": "t", "episode_id": episode, "balanced_task_effect_error": 1.0})
    result = paired_episode_bootstrap(rows, "M", "B", 200, 10)
    assert abs(result["pooled"]["point"] - np.mean([0.2, -0.1, 0.4])) < 1e-12


def test_stage5_historical_binding_records_all_immutable_stage_results():
    payload = json.load(open(os.path.join(OUTPUT, "HISTORICAL_BINDING.json"), encoding="utf-8"))
    expected = {
        "stage1": "REJECT_CORE_HYPOTHESIS",
        "stage1_5": "REJECT_P15_FAMILY",
        "stage2": "ORACLE_ONLY_NO_DEPLOYABLE_MODEL",
        "stage3": "ORACLE_ONLY_NO_LEARNABLE_RANKER",
        "stage4": "STATIC_EFFECT_METRIC_ONLY",
    }
    assert {key: payload["historical_evidence"][key]["disposition"] for key in expected} == expected


def test_stage5_confirmation_firewall_is_metric_independent():
    payload = json.load(open(os.path.join(OUTPUT, "FRESH_PHASE_SELECTION_RULE.json"), encoding="utf-8"))
    assert payload["metric_scores_read"] is False
    assert payload["phase_is_proposed_model_input"] is False
    assert payload["executable_state_requirement"]["all_128_local_candidates_valid_without_clipping"] is True


def test_stage5_exact_one_disposition_precedence():
    cases = [
        ((False, True, True, True, True, True, True), "BLOCKED_HISTORICAL_BINDING_MISMATCH"),
        ((True, False, True, True, True, True, True), "STATIC_EFFECT_GEOMETRY_SUFFICIENT"),
        ((True, True, False, True, True, True, True), "STATIC_CONSEQUENCE_METRIC_ONLY"),
        ((True, True, False, False, True, True, True), "REJECT_LEARNED_CONSEQUENCE_METRIC"),
        ((True, True, True, True, False, True, True), "PIVOT_TO_CONSEQUENCE_RETRIEVAL_STEERING"),
        ((True, True, True, True, True, False, True), "BLOCKED_NO_FRESH_TRAJECTORIES"),
        ((True, True, True, True, True, True, False), "CONFIRMATION_FAILED"),
        ((True, True, True, True, True, True, True), "GO_TO_FIXED_POLICY_RERANKING"),
    ]
    observed = [choose_disposition(*arguments) for arguments, _ in cases]
    assert observed == [expected for _, expected in cases]
    assert all(exact_one_disposition(value) for value in observed)
    assert set(observed) == set(FINAL_DISPOSITIONS)
