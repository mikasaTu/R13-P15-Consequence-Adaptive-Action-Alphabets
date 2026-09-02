import ast
import inspect
import json
import os

import numpy as np
import pytest

from caaa_libero import stage6a_selection
from caaa_libero.stage5_oracle import deterministic_kmedoids_precomputed
from caaa_libero.stage6a_evaluation import (
    executed_consequence_lookup,
    require_gate_h,
    selection_input_contract,
)
from caaa_libero.stage6a_health import normalized_distinct_utilization
from caaa_libero.stage6a_statistics import clustered_paired_bootstrap, choose_disposition


ROOT = os.path.dirname(os.path.dirname(__file__))
OUTPUT = os.path.join(ROOT, "experiments", "r13_p15_cdaa", "stage6a")


def _distance(values):
    return np.sum((values[:, None] - values[None, :]) ** 2, axis=-1)


def test_c4_absent_from_repaired_selection_ast():
    tree = ast.parse(inspect.getsource(stage6a_selection))
    names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    forbidden = {"create_pair_ranker", "score_pairs", "_ensemble_pair_score", "C5", "C6"}
    assert not (names & forbidden)


def test_kmedoids_deterministic_and_id_stable():
    values = np.asarray([[0.0], [1.0], [3.0], [8.0], [9.0]])
    ids = np.asarray([50, 10, 30, 20, 40])
    first = deterministic_kmedoids_precomputed(_distance(values), 3, ids)
    second = deterministic_kmedoids_precomputed(_distance(values), 3, ids)
    assert np.array_equal(first, second)
    permutation = np.asarray([3, 0, 4, 1, 2])
    permuted = deterministic_kmedoids_precomputed(
        _distance(values[permutation]), 3, ids[permutation]
    )
    assert set(ids[first]) == set(ids[permutation][permuted])


def test_candidate_order_selection_permutation_invariance():
    candidate = np.asarray([[0.0], [2.0], [5.0], [9.0]])
    target = np.asarray([[1.0], [7.0]])
    ids = np.asarray([40, 10, 30, 20])
    atlas = stage6a_selection.build_c3_kmedoids_atlas(candidate, ids, 3)
    selected, _ = stage6a_selection.select_c3_only(target, candidate, atlas, ids)
    permutation = np.asarray([2, 0, 3, 1])
    atlas_p = stage6a_selection.build_c3_kmedoids_atlas(
        candidate[permutation], ids[permutation], 3
    )
    selected_p, _ = stage6a_selection.select_c3_only(
        target, candidate[permutation], atlas_p, ids[permutation]
    )
    assert np.array_equal(ids[selected], ids[permutation][selected_p])


def test_model_input_contract_contains_no_ids():
    context, target, candidate = selection_input_contract(
        np.zeros(3), np.zeros((2, 4)), np.zeros((5, 4))
    )
    assert context.shape == (3,)
    assert target.shape == (2, 4)
    assert candidate.shape == (5, 4)
    assert len(inspect.signature(selection_input_contract).parameters) == 3


def test_health_zero_clipping_and_distinct_utilization():
    health = json.load(open(os.path.join(OUTPUT, "QUANTIZER_HEALTH.json"), encoding="utf-8"))
    assert health["metrics"]["coordinate_clipping_operations"] == 0
    assert health["metrics"]["median_realized_clipped_coordinate_fraction"] == 0.0
    assert normalized_distinct_utilization([1, 1, 2, 2], 4) == 0.5


def test_lookup_uses_executed_table_and_gate_guard():
    table = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    selected = np.asarray([[0, 1, 2], [3, 2, 1]])
    result = executed_consequence_lookup(table, selected)
    assert result.tolist() == [[0, 5, 10], [15, 18, 21]]
    with pytest.raises(RuntimeError):
        require_gate_h({"passed": False})


def test_historical_paths_immutable():
    binding = json.load(open(os.path.join(OUTPUT, "HISTORICAL_BINDING.json"), encoding="utf-8"))
    assert binding["passed"] is True
    assert binding["checks"]["historical_paths_immutable"] is True


def test_bootstrap_resamples_complete_episode_clusters():
    left = np.asarray([1.0, 1.0, 5.0, 5.0])
    right = np.zeros(4)
    episode = np.asarray(["a", "a", "b", "b"])
    result = clustered_paired_bootstrap(left, right, episode, 64, 17)
    assert set(np.unique(result)).issubset({1.0, 3.0, 5.0})


def test_exactly_one_disposition_precedence():
    assert choose_disposition(True, True, True, False) == "QUANTIZER_STILL_DEGENERATE"
    assert choose_disposition(False, False, False, False) == "BLOCKED_HISTORICAL_BINDING_MISMATCH"
    assert choose_disposition(True, True, True, True, False, False) == "GAIN_NOT_DENSITY_SPECIFIC"
    assert choose_disposition(True, True, True, True, True, True) == "REPAIR_CONFIRMED_ADVANCE_TO_STAGE6B"
