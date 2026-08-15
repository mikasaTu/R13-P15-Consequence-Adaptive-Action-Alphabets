import numpy as np

from caaa_libero.stage4_config import (
    FINAL_DISPOSITIONS,
    SUPPORT_DIRECTION_COUNT,
    SUPPORT_DIRECTION_FAMILIES,
    SUPPORT_TARGET_COUNT,
    TASKS,
    TRAIN_EPISODES,
    TRAIN_STATE_COUNT,
    method_definitions,
)
from caaa_libero.stage4_collection import _worker_unit
from caaa_libero.stage4_freeze import (
    _phase_windows,
    _valid_action_chunk,
    generate_training_support_bank,
)
from caaa_libero.stage4_data import reversal_pairs
from caaa_libero.stage4_models import create_cr_model, parameter_count
from caaa_libero.stage4_reselect import deterministic_kmedoids


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


def test_stage4_episode_workers_are_disjoint_and_balanced():
    units = []
    for task in TASKS:
        for episode_id in TRAIN_EPISODES:
            unit = _worker_unit(
                {"task_id": task["task_id"], "episode_id": episode_id}
            )
            units.append(unit)
    assert sorted(units) == list(range(64))
    assignments = [unit % 16 for unit in units]
    assert np.bincount(assignments, minlength=16).tolist() == [4] * 16


def test_stage4_predicted_space_kmedoids_is_deterministic_and_unique():
    rng = np.random.default_rng(13150400)
    values = rng.normal(size=(32, 5))
    left = deterministic_kmedoids(values, k=8)
    right = deterministic_kmedoids(values, k=8)
    assert np.array_equal(left, right)
    assert left.shape == (8,)
    assert len(np.unique(left)) == 8
    assert np.all((left >= 0) & (left < len(values)))


def test_stage4_reversal_pairs_are_balanced_and_strict():
    phases = ("free_space", "pre_contact", "contact_onset", "post_contact")
    tasks = tuple(task["task_id"] for task in TASKS)
    task_id, phase, keys, episode, snapshot, distances = [], [], [], [], [], []
    for task in tasks:
        for phase_name in phases:
            for side in range(2):
                task_id.append(task)
                phase.append(phase_name)
                keys.append(f"{task}__{phase_name}__{side}")
                episode.append(16 + side)
                snapshot.append(side)
                candidate = np.arange(256, dtype=np.float32)
                if side:
                    candidate = candidate[::-1].copy()
                distances.append(np.repeat(candidate[None, :], 96, axis=0))
    cache = {
        "true_distance": np.asarray(distances),
        "task_id": np.asarray(task_id),
        "phase": np.asarray(phase),
        "key": np.asarray(keys),
        "episode_id": np.asarray(episode),
        "snapshot_index": np.asarray(snapshot),
        "direction_family_id": np.repeat(np.arange(3), 32),
    }
    margins = {(task, phase_name): 0.1 for task in tasks for phase_name in phases}
    rows = reversal_pairs(cache, margins, count_per_task_phase=96)
    assert len(rows) == len(tasks) * len(phases) * 96
    counts = {}
    for row in rows:
        key = (row["task_id"], row["phase"])
        counts[key] = counts.get(key, 0) + 1
        assert row["true_gap_s1_j_minus_i"] > row["margin"]
        assert row["true_gap_s2_j_minus_i"] < -row["margin"]
    assert set(counts.values()) == {96}


def test_stage4_cr_families_expose_equal_group_distance_contract():
    import torch

    context = torch.zeros((3, 321))
    target = torch.zeros((3, 24))
    candidate = torch.ones((3, 24))
    shared = create_cr_model("CR_C3_SHARED")
    grouped = create_cr_model("CR_C3_GROUP")
    assert shared.embed(context, target).shape == (3, 1, 32)
    assert grouped.embed(context, target).shape == (3, 5, 16)
    assert shared.pair_distance(context, target, candidate).shape == (3,)
    assert grouped.pair_distance(context, target, candidate).shape == (3,)
    assert parameter_count(shared) > 0
    assert parameter_count(grouped) > 0
