"""Frozen configuration for the preregistered Stage 2 NCEA audit.

This module contains only a-priori choices.  Development or confirmation
results must never be used to change these values.
"""

from __future__ import annotations

from . import config


PROJECT_ID = "r13_p15_ncea_stage2"
OUTPUT_RELATIVE = "experiments/r13_p15_ncea/stage2"

TASKS = config.TASKS
PHASES = config.PHASES
HORIZON = 4
ACTION_DIM = 7
CONTINUOUS_DIM = 24
PRIMARY_K = 64
ACTION_BANK_SIZE = 256
MIN_VALID_BANK = 128
SETTLE_STEPS = 3

HISTORICAL_EPISODES = tuple(range(0, 16))
TRAIN_EPISODES = tuple(range(16, 24))
CALIBRATION_EPISODES = tuple(range(24, 28))
DEVELOPMENT_EPISODES = tuple(range(28, 32))
CONFIRMATION_EPISODES = tuple(range(32, 40))
ALL_FRESH_EPISODES = tuple(range(16, 40))
SPLIT_EPISODES = {
    "train": TRAIN_EPISODES,
    "calibration": CALIBRATION_EPISODES,
    "development": DEVELOPMENT_EPISODES,
    "confirmation": CONFIRMATION_EPISODES,
}

GLOBAL_SEED = 13150200
DIRECTION_COUNT = 24
DIRECTION_FAMILIES = (
    "smooth_dct",
    "suffix_contact",
    "low_rank_temporal_action",
)
DIRECTION_FAMILY_COUNTS = {
    "smooth_dct": 12,
    "suffix_contact": 6,
    "low_rank_temporal_action": 6,
}
RADIUS_INTERVAL = (0.04, 0.12)
RADII_PER_DIRECTION = 2
SIGNS = (-1, 1)
SUPPORT_ROWS_PER_STATE = DIRECTION_COUNT * RADII_PER_DIRECTION * len(SIGNS)

# Robust scale floors are in the physical units of env_adapter.FEATURE_NAMES.
SCALE_FLOORS = {
    "position": 1e-3,
    "rotation6d": 1e-2,
    "gripper_width": 1e-3,
    "articulation": 1e-3,
    "task_progress": 1e-2,
    "log_contact_force": 5e-2,
    "penetration": 1e-5,
    "joint_violation": 1e-4,
}
HUBER_DELTA = 1.5
HUBER_CAP = 4.0
PRIMARY_GROUPS = {
    "object_pose": tuple(range(0, 9)) + tuple(range(27, 36)),
    "tcp_object_relative_pose": tuple(range(18, 27)),
    "contact_mode_and_penetration": (44,),
    "gripper_and_articulation": (36, 37, 38, 39),
    "task_progress_and_constraint": (40, 45),
}
CONTACT_FORCE_INDICES = (41, 42, 43)
PREDICTED_CONTINUOUS_INDICES = tuple(
    sorted({index for values in PRIMARY_GROUPS.values() for index in values})
)

METHODS = (
    "B0_continuous_target",
    "B1_centered_covariance",
    "B2_phase_residual",
    "B3_dynamic_action_medoids",
    "B4_state_action_vq",
    "LJ_linear_j_atlas",
    "O1_true_effect_oracle",
    "O2_linear_j_oracle",
    "NCEA",
    "MC_NCEA",
    "UG_NCEA",
    "P3_mode_shuffled_atlas",
    "P4_state_shuffled_atlas",
    "P5_effect_shuffled_atlas",
    "P6_random_latent_atlas",
)
DEPLOYABLE_BASELINES = (
    "B1_centered_covariance",
    "B2_phase_residual",
    "B3_dynamic_action_medoids",
    "B4_state_action_vq",
)
PRIMARY_DEPLOYABLE = ("NCEA", "MC_NCEA")
MECHANISM_CONTROLS = (
    "P3_mode_shuffled_atlas",
    "P4_state_shuffled_atlas",
    "P5_effect_shuffled_atlas",
    "P6_random_latent_atlas",
)

PREDICTOR_ARCHITECTURES = ((128, 128), (256, 256))
PREDICTOR_ENSEMBLE_SIZE = 5
PREDICTOR_MAX_EPOCHS = 160
PREDICTOR_PATIENCE = 20
PREDICTOR_MIN_DELTA = 1e-5
PREDICTOR_BATCH_SIZE = 512
PREDICTOR_LEARNING_RATE = 1e-3
PREDICTOR_WEIGHT_DECAY = 1e-5
CONTACT_LOSS_WEIGHT = 0.25
ACTION_LATENT_DIM = 32
RANDOM_FEATURE_RIDGE = 1e-3

LINEAR_RIDGE_GRID = (1e-6, 1e-4, 1e-2, 1e-1)
LINEAR_NEIGHBOR_GRID = (1, 3, 5, 9)
CONTACT_CONFIDENCE_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)
UNCERTAINTY_COVERAGES = (0.50, 0.70, 0.90)
ATLAS_SELECTION_OPERATOR = "deterministic_farthest_point_medoid"

BOOTSTRAP_REPLICATES = 10000

GATES = {
    "A": {
        "oracle_relative_gain_min": 0.10,
        "tasks_improved_min": 3,
        "failure_disposition": "REJECT_BROAD_CONSEQUENCE_HYPOTHESIS",
    },
    "B": {
        "prediction_relative_gain_vs_linear_min": 0.20,
        "contact_sensitive_tasks_improved_min": 2,
        "oracle_gap_fraction_closed_min": 0.50,
        "must_beat_all_shuffled_and_random_controls": True,
        "failure_disposition": "ORACLE_ONLY_NO_DEPLOYABLE_MODEL",
    },
    "C": {
        "relative_gain_min": 0.08,
        "tasks_improved_min": 3,
        "contact_sensitive_tasks_improved_min": 2,
        "bowl_on_plate_max_degradation": 0.05,
        "clipping_rate_max": 0.01,
        "normalized_code_utilization_min": 0.25,
        "action_reconstruction_degradation_max": 0.10,
        "failure_disposition": "REJECT_NONLINEAR_CONSEQUENCE_ALPHABET",
    },
    "NARROW": {
        "contact_phase_gain_min": 0.10,
        "free_space_max_degradation": 0.05,
        "control_gain_retention_max": 0.25,
    },
    "GO": {
        "confirmation_relative_gain_min": 0.10,
        "ci_lower_bound_min_exclusive": 0.0,
        "tasks_improved_min": 3,
        "contact_sensitive_tasks_improved_min": 2,
        "bowl_on_plate_max_degradation": 0.05,
        "control_gain_retention_max": 0.25,
        "action_reconstruction_degradation_max": 0.10,
        "coverage_min": 0.70,
        "clipping_rate_max": 0.01,
        "normalized_code_utilization_min": 0.25,
    },
}

CONTACT_SENSITIVE_TASKS = ("plate_push", "stove_turn_on", "wine_rack")


def split_for_episode(episode_id):
    episode_id = int(episode_id)
    for split, ids in SPLIT_EPISODES.items():
        if episode_id in ids:
            return split
    raise KeyError(episode_id)


def method_definitions():
    return {
        "primary_k": PRIMARY_K,
        "common_candidate_bank_size": ACTION_BANK_SIZE,
        "selection_operator": ATLAS_SELECTION_OPERATOR,
        "methods": {
            "B0_continuous_target": "Unquantized target branch; upper bound only.",
            "B1_centered_covariance": "K=64 deterministic k-means in train-only covariance-whitened residual space; centers map to unique executable bank medoids.",
            "B2_phase_residual": "Four phase-specific deterministic K=64 residual k-means fits; centers map to unique executable bank medoids.",
            "B3_dynamic_action_medoids": "Per-state K=64 deterministic action-space FPS medoids from the valid common bank.",
            "B4_state_action_vq": "Per-state K=64 FPS in a state-conditioned action-autoencoder latent trained only for action reconstruction.",
            "LJ_linear_j_atlas": "State-neighbor interpolated train-only local Jacobian predictions; bank actions only and no pseudoinversion.",
            "O1_true_effect_oracle": "Per-state K=64 atlas and target assignment using true simulator effects for the common bank and target.",
            "O2_linear_j_oracle": "Diagnostic alias for the LJ predicted-effect atlas used to define the O1-O2 gap.",
            "NCEA": "Five-member global nonlinear consequence ensemble; dynamic predicted-effect atlas over valid bank actions.",
            "MC_NCEA": "Five-member shared-trunk predictor with four phase heads; dynamic mode-conditioned predicted-effect atlas.",
            "UG_NCEA": "Calibration-frozen uncertainty score; NCEA/MC-NCEA used on the lowest-risk 50/70/90 percent and the frozen strongest baseline elsewhere.",
            "P3_mode_shuffled_atlas": "MC-NCEA architecture trained with task-stratified shuffled phase/head assignments.",
            "P4_state_shuffled_atlas": "NCEA architecture trained after shuffling state features within task and phase.",
            "P5_effect_shuffled_atlas": "NCEA architecture trained after shuffling continuous effects and contact labels within task and phase.",
            "P6_random_latent_atlas": "Frozen random trunk with ridge readout, matched hidden width and ensemble count.",
        },
        "hard_rules": {
            "common_bank_only": True,
            "pseudoinversion": False,
            "invalid_action_clipping": False,
            "gripper_command": "copied unchanged from nominal demonstration chunk",
            "k_sensitivity_locked_until_disposition": [32, 128],
        },
    }


def consequence_metric_definition():
    return {
        "primary": "BALANCED_TASK_EFFECT",
        "feature_groups": {name: list(indices) for name, indices in PRIMARY_GROUPS.items()},
        "group_weights": {name: 0.2 for name in PRIMARY_GROUPS},
        "predicted_continuous_indices": list(PREDICTED_CONTINUOUS_INDICES),
        "contact_force_indices_excluded_from_primary": list(CONTACT_FORCE_INDICES),
        "train_only_scaling": "max(1.4826*MAD, IQR/1.349, preregistered physical-unit floor)",
        "scale_floors": dict(SCALE_FLOORS),
        "error": {
            "type": "capped_huber",
            "delta": HUBER_DELTA,
            "absolute_normalized_cap": HUBER_CAP,
            "dimensions_averaged_within_group": True,
            "active_groups_averaged_equally": True,
            "contact_mode_mismatch": "0/1 term averaged with normalized penetration inside group 3",
        },
        "secondary": [
            "CONTACT_FORCE_EFFECT",
            "FROZEN_STAGE1_CONSEQUENCE_METRIC_CONTINUITY_ONLY",
            "contact_mode_preservation",
            "task_progress_preservation",
            "action_reconstruction_error",
        ],
    }
