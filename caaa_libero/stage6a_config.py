"""Frozen configuration for the Stage 6-A defect-repair replay."""

from __future__ import annotations

import os


OUTPUT_RELATIVE = os.path.join("experiments", "r13_p15_cdaa", "stage6a")
STAGE3_RELATIVE = os.path.join("experiments", "r13_p15_ncer_aa", "stage3")
STAGE5_RELATIVE = os.path.join("experiments", "r13_p15_cicr_dla", "stage5")
STAGE5_SCRATCH_ROOT = (
    "/mnt/cpfs/zbl-cpfs-new/dataset/leon/experiments/"
    "r13_p15_cicr_dla/stage5_work"
)
STAGE1_SCRATCH_ROOT = (
    "/mnt/cpfs/zbl-cpfs-new/dataset/leon/experiments/"
    "r13_p15_caaa_v2/stage1/work"
)

TASK_IDS = ("bowl_on_plate", "plate_push", "stove_turn_on", "wine_rack")
PHASES = ("free_space", "pre_contact", "contact_onset", "post_contact")
CONTACT_SENSITIVE_TASKS = ("plate_push", "stove_turn_on", "wine_rack")

PRIMARY_K = 64
MIN_VALID_CANDIDATES = 96
ATLAS_SEED = 13150601
CONTROL_SEED = 13150602
BOOTSTRAP_SEED = 13150603
BOOTSTRAP_REPLICATES = 10_000

GATE_H = {
    "median_normalized_assignment_utilization_strictly_greater_than": 0.50,
    "median_realized_clipped_coordinate_fraction_less_than": 0.05,
    "pooled_dead_code_fraction_less_than": 0.10,
    "action_reconstruction_rmse_ratio_at_most": 1.25,
    "minimum_valid_candidates_per_state": MIN_VALID_CANDIDATES,
}

GATE_A = {
    "minimum_relative_improvement": 0.08,
    "bootstrap_ci_lower_bound_strictly_greater_than": 0.0,
    "minimum_improved_tasks": 3,
    "minimum_improved_contact_sensitive_tasks": 2,
    "minimum_recovered_fraction": 0.20,
    "maximum_random_gain_retention": 0.50,
    "maximum_uniform_gain_retention": 0.50,
    "maximum_label_shuffle_gain_retention": 0.25,
    "maximum_action_rmse_degradation": 0.20,
    "maximum_contact_preservation_drop": 0.01,
}

FINAL_DISPOSITIONS = (
    "BLOCKED_HISTORICAL_BINDING_MISMATCH",
    "BLOCKED_NO_EXECUTED_CANDIDATE_CACHE",
    "BLOCKED_DEFECT_NOT_REPRODUCED",
    "QUANTIZER_STILL_DEGENERATE",
    "GAIN_NOT_DENSITY_SPECIFIC",
    "C4_REMOVAL_INSUFFICIENT",
    "REPAIR_CONFIRMED_ADVANCE_TO_STAGE6B",
)

HISTORICAL_DISPOSITIONS = {
    "stage1": "REJECT_CORE_HYPOTHESIS",
    "stage1_5": "REJECT_P15_FAMILY",
    "stage2": "ORACLE_ONLY_NO_DEPLOYABLE_MODEL",
    "stage3": "ORACLE_ONLY_NO_LEARNABLE_RANKER",
    "stage4": "STATIC_EFFECT_METRIC_ONLY",
    "stage5": "STATIC_CONSEQUENCE_METRIC_ONLY",
}
