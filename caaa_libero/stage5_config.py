"""Frozen pre-result contract for the Stage 5 CICR-DLA audit.

Only choices that are permitted before Stage 5 development evidence is
inspected live in this module.  Calibration may choose only among the
explicit temperature candidates.  Development, historical exploratory and
fresh-confirmation outcomes may not change any value below.
"""

from __future__ import annotations

from . import config


PROJECT_ID = "r13_p15_cicr_dla_stage5"
BRANCH = "r13-p15-stage5-context-identifiable-consequence-metric"
OUTPUT_RELATIVE = "experiments/r13_p15_cicr_dla/stage5"
SCRATCH_ROOT = (
    "/mnt/cpfs/zbl-cpfs-new/dataset/leon/experiments/"
    "r13_p15_cicr_dla/stage5_work"
)

PREREGISTRATION_BASE_COMMIT = "eba489ec8f866f712b582083c088e93b0aaccf11"
PREREGISTRATION_BASE_TREE = "0137158cfd5a3f4e1162acf4f47bdc073839baf9"
STAGE4_RESULT_COMMIT = "ac861eb60f83c72ac4785d8d901356434eded2ec"
STAGE4_TREE = "75dca63f2e7c938eb9fccb0a82f7171d0eb091c0"
HISTORICAL_STAGE_TREES = {
    "stage1": "e00831263060b67d05f11483bf17fe42a4b57dbe",
    "stage1_5": "cd55cd0af69e8c8c1955182f7b459f712759e267",
    "stage2": "ba2cdd070e8d27bcd69a79cefdc8259770b1af27",
    "stage3": "2ca0d9bd79d7486bfa6ac1876ddae349b86008eb",
    "stage4": STAGE4_TREE,
}
HISTORICAL_EVIDENCE = {
    "stage1": {
        "disposition": "REJECT_CORE_HYPOTHESIS",
        "formal_commit": "34995e8e7c3069b22785ad04536f0d429e75c0fc",
        "published_commit": "434427af0f8adc844851c27cfc050b2c9c6752dc",
    },
    "stage1_5": {
        "disposition": "REJECT_P15_FAMILY",
        "preregistration_commit": "9a3ac1a4c774103fe618bd283909c2793ed581ec",
        "method_commit": "aa82d46c5e0828956aef15918c2aa7656844472f",
        "result_commit": "76433b6e58196ceeedc4ad005a1110ea8e343ae2",
    },
    "stage2": {
        "disposition": "ORACLE_ONLY_NO_DEPLOYABLE_MODEL",
        "published_commit": "74c98979910a3831d0abeb8d13111a7c9294b067",
    },
    "stage3": {
        "disposition": "ORACLE_ONLY_NO_LEARNABLE_RANKER",
        "code_commit": "04e95630aee87f356940b2522f2025faa8c7c209",
        "published_commit": "beb63576e91307260b64687e58ea99e6da93c478",
    },
    "stage4": {
        "disposition": "STATIC_EFFECT_METRIC_ONLY",
        "preregistration_commit": "6221f335d06f78505d1840de177ae3ed0c153daa",
        "development_commit": "edf55ed602ce6ed7d41870b955e4684a5b5730bc",
        "split_commit": "88ff3f55debb2237ad9e0ec1dd165f971fa1d37b",
        "published_commit": STAGE4_RESULT_COMMIT,
    },
}

TASKS = config.TASKS
TASK_IDS = tuple(task["task_id"] for task in TASKS)
CONTACT_SENSITIVE_TASKS = ("plate_push", "stove_turn_on", "wine_rack")
PHASES = config.PHASES
HORIZON = 4
ACTION_DIM = 7
CONTINUOUS_DIM = 24
SETTLE_STEPS = 3
CONTROL_MODE = "OSC_POSE"
CONTROL_FREQUENCY_HZ = 20

TRAIN_EPISODES = tuple(range(16, 32))
CALIBRATION_EPISODES = tuple(range(32, 36))
DEVELOPMENT_EPISODES = tuple(range(36, 40))
HISTORICAL_EXPLORATORY_EPISODES = tuple(range(40, 50))
GENERATOR_TRAIN_EPISODES = tuple(range(0, 32))

SOURCE_BANK_SIZE = 256
LOCAL_BANK_SIZE = 128
PRIMARY_K = 64
MINIMUM_VALID_CANDIDATES = 96
LOCAL_BANK_COVARIANCE_REGULARIZATION = 1e-6
LOCAL_BANK_NEAR_DUPLICATE_TOLERANCE = 1e-12
LOCAL_BANK_BALANCE_AXES = ("source_phase", "source_family_id", "source_sign")

GLOBAL_SEED = 13150500
LOCAL_BANK_SEED = 13150501
REVERSAL_SEED = 13150502
BOOTSTRAP_SEED = 13150504
GENERATOR_TRAIN_SEED = 13150505
MODEL_SEEDS = (13150517, 13150529, 13150543)
BOOTSTRAP_REPLICATES = 10000

REVERSAL_MARGIN_QUANTILE = 0.25
REVERSAL_TRAIN_QUOTA_PER_TASK_PHASE = 256
REVERSAL_CALIBRATION_QUOTA_PER_TASK_PHASE = 128
REVERSAL_DEVELOPMENT_QUOTA_PER_TASK_PHASE = 128
REVERSAL_ATTEMPT_MULTIPLIER = 400

EMBEDDING_DIM = 24
ACTION_ENCODER_HIDDEN = (128, 96)
CONTEXT_MODULATOR_HIDDEN = (128, 64)
MODULATION_LOG_BOUND = 1.25
TRAINING_STEPS = 2500
TRAIN_QUERY_BATCH = 8
REVERSAL_BATCH = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP_NORM = 5.0
TEMPERATURE_CANDIDATES = (0.10, 0.15, 0.20)
LOSS_WEIGHTS = {
    "distance": 1.0,
    "pairwise": 0.5,
    "listwise": 0.5,
    "reversal": 1.0,
    "gate": 0.01,
}
PAIRWISE_STRATA = {
    "oracle_top_positives": 8,
    "rank_9_32_hard_negatives": 24,
    "contact_changing": 8,
    "action_close_effect_far": 8,
    "action_far_effect_close": 8,
}
HUBER_DELTA = 1.0
P2_ENABLED = False

MODEL_FAMILIES = (
    "B1_ACTION_ONLY",
    "B2_STATIC_CONSEQUENCE",
    "P1_CONTEXT_GATED_PSD",
)
MATCHED_CONTROLS = (
    "ACTION_ONLY",
    "CONTEXT_SHUFFLED",
    "NOMINAL_SHUFFLED",
    "JOINT_STATE_NOMINAL_SHUFFLED",
    "CONSEQUENCE_LABEL_SHUFFLED",
    "NO_REVERSAL_LOSS",
    "PHASE_ONLY",
    "CURRENT_CONTACT_ONLY",
)
PROPOSED_INPUT_FIELDS = (
    "current_observable_state",
    "current_observable_state_mask",
    "previous_two_observable_state_deltas",
    "previous_two_observable_state_delta_masks",
    "previous_two_executed_actions",
    "previous_action_availability_masks",
    "current_observable_contact",
    "nominal_h4_action_chunk",
    "target_residual_action_chunk",
    "candidate_residual_action_chunk",
    "task_identity",
)
FORBIDDEN_INPUT_FIELDS = (
    "future_state",
    "future_consequence",
    "target_simulator_outcome",
    "candidate_simulator_outcome",
    "demonstration_phase",
    "episode_success",
    "future_reward",
    "candidate_id",
    "target_id",
    "row_index",
    "confirmation_result",
    "post_execution_contact",
    "oracle_atlas_membership",
)

GENERATOR_ARCHITECTURE = {
    "kind": "shared_state_h4_chunk_bc",
    "input": (
        "current_46d_physical_observable",
        "46d_observability_mask",
        "previous_7d_executed_action",
        "4d_task_one_hot",
    ),
    "hidden": [256, 256, 128],
    "output": "4x7 normalized OSC_POSE action through tanh",
    "execution": "closed-loop receding horizon; execute first predicted action",
    "optimizer": "AdamW",
    "learning_rate": 3e-4,
    "weight_decay": 1e-5,
    "batch_size": 512,
    "steps": 10000,
    "gripper_loss_weight": 0.5,
    "maximum_rollout_steps": 500,
    "training_episodes": list(GENERATOR_TRAIN_EPISODES),
    "uses_images": False,
    "uses_stage5_consequence_labels": False,
}
GENERATOR_ROLLOUT_SEED_BASE = {
    "bowl_on_plate": 131506000,
    "plate_push": 131507000,
    "stove_turn_on": 131508000,
    "wine_rack": 131509000,
}
GENERATOR_ROLLOUT_SEEDS_PER_TASK = 200
GENERATOR_REQUIRED_SUCCESSES_PER_TASK = 12
GENERATOR_SACRIFICIAL_SEEDS = {
    task: tuple(range(base + 900, base + 904))
    for task, base in GENERATOR_ROLLOUT_SEED_BASE.items()
}

FRESH_TARGET_SEED = 13150590
FRESH_TARGET_DIRECTION_COUNT = 24
FRESH_TARGET_FAMILIES = ("smooth_dct", "suffix_contact", "low_rank_temporal_action")
FRESH_TARGET_RADII = (0.055, 0.095)
FRESH_TARGET_SIGNS = (-1, 1)
FRESH_TARGET_COUNT = 96
FRESH_TARGET_MAX_COMPONENT = 0.80

GATES = {
    "oracle_adaptivity": {
        "pooled_gain_min": 0.08,
        "contact_phase_gain_min": 0.12,
        "contact_sensitive_tasks_improved_min": 2,
        "strict_pair_count_min": 1000,
        "contact_phases_with_reversal_rate_min": 2,
        "reversal_rate_min": 0.15,
        "failure_disposition": "STATIC_EFFECT_GEOMETRY_SUFFICIENT",
    },
    "context_identifiable": {
        "realized_pooled_gain_min": 0.05,
        "paired_ci_lower_bound_exclusive": 0.0,
        "tasks_improved_min": 3,
        "contact_sensitive_tasks_improved_min": 2,
        "oracle_regret_reduction_min": 0.10,
        "ndcg16_gain_min": 0.05,
        "joint_reversal_accuracy_min": 0.35,
        "joint_reversal_gain_points_min": 0.15,
        "joint_shuffle_retention_max": 0.25,
        "label_shuffle_retention_max": 0.25,
        "action_only_retention_max": 0.50,
        "all_seed_directions_same": True,
    },
    "static_consequence_value": {
        "realized_pooled_gain_min": 0.05,
        "paired_ci_lower_bound_exclusive": 0.0,
        "tasks_improved_min": 3,
        "contact_sensitive_tasks_improved_min": 2,
        "oracle_regret_reduction_min": 0.10,
        "ndcg16_gain_min": 0.05,
        "comparators": ["B0_CURRENT_CONTACT_KMEANS", "B1_ACTION_ONLY"],
    },
    "dynamic_k64": {
        "realized_gain_min": 0.08,
        "full_gain_retention_min": 0.75,
        "tasks_improved_min": 3,
        "contact_sensitive_tasks_improved_min": 2,
        "action_rmse_degradation_max": 0.20,
        "contact_preservation_drop_max_points": 0.01,
        "normalized_utilization_min": 0.25,
        "clipping_rate_max": 0.0,
        "minimum_valid_candidates": MINIMUM_VALID_CANDIDATES,
    },
    "confirmation": {
        "pooled_gain_min": 0.10,
        "paired_ci_lower_bound_exclusive": 0.0,
        "tasks_improved_min": 3,
        "contact_sensitive_tasks_improved_min": 2,
        "context_shuffle_retention_max": 0.25,
        "label_shuffle_retention_max": 0.25,
        "action_rmse_degradation_max": 0.20,
        "contact_preservation_drop_max_points": 0.01,
        "normalized_utilization_min": 0.25,
        "clipping_rate_max": 0.0,
        "all_seed_directions_same": True,
        "oracle_adaptive_headroom_min": 0.08,
    },
}

FINAL_DISPOSITIONS = (
    "BLOCKED_HISTORICAL_BINDING_MISMATCH",
    "STATIC_EFFECT_GEOMETRY_SUFFICIENT",
    "STATIC_CONSEQUENCE_METRIC_ONLY",
    "REJECT_LEARNED_CONSEQUENCE_METRIC",
    "PIVOT_TO_CONSEQUENCE_RETRIEVAL_STEERING",
    "BLOCKED_NO_FRESH_TRAJECTORIES",
    "CONFIRMATION_FAILED",
    "GO_TO_FIXED_POLICY_RERANKING",
)


def rollout_seeds():
    return {
        task: list(range(base, base + GENERATOR_ROLLOUT_SEEDS_PER_TASK))
        for task, base in GENERATOR_ROLLOUT_SEED_BASE.items()
    }


def model_definitions():
    """Return the machine-readable contract frozen before development."""
    return {
        "project_id": PROJECT_ID,
        "controller": {
            "robot": "Panda",
            "control_mode": CONTROL_MODE,
            "frequency_hz": CONTROL_FREQUENCY_HZ,
            "horizon": HORIZON,
            "settle_steps": SETTLE_STEPS,
            "continuous_action_semantics": "4x6 normalized OSC_POSE coordinates",
            "gripper_semantics": "copied stepwise from the nominal chunk for branch actions",
        },
        "primary_metric": "frozen Stage 2-4 BALANCED_TASK_EFFECT",
        "local_bank": {
            "source_size": SOURCE_BANK_SIZE,
            "primary_size": LOCAL_BANK_SIZE,
            "primary_alphabet_size": PRIMARY_K,
            "selection": (
                "24 source_phase x family x sign strata; quota 5 plus one for the "
                "first 8 lexicographic strata; within-stratum ascending train-"
                "covariance-whitened zero-origin residual norm then original ID"
            ),
            "covariance_fit": "union of frozen Stage 4 train targets and M=256 bank residuals",
            "covariance_regularization": LOCAL_BANK_COVARIANCE_REGULARIZATION,
            "dedup_tolerance": LOCAL_BANK_NEAR_DUPLICATE_TOLERANCE,
            "clipping_or_action_synthesis_forbidden": True,
        },
        "architectures": {
            "B0": "current-contact residual k-means over the executable local bank",
            "B1": "matched action encoder trained without consequence labels",
            "B2": "state-independent diagonal PSD consequence metric",
            "P1": (
                "frozen B2 representation with bounded observable-context multiplicative "
                "diagonal PSD modulation; m=0 exactly recovers B2"
            ),
            "P2": "not enabled in this preregistration",
        },
        "embedding_dim": EMBEDDING_DIM,
        "action_encoder_hidden": list(ACTION_ENCODER_HIDDEN),
        "context_modulator_hidden": list(CONTEXT_MODULATOR_HIDDEN),
        "modulation_log_bound": MODULATION_LOG_BOUND,
        "loss_weights": dict(LOSS_WEIGHTS),
        "pairwise_strata": dict(PAIRWISE_STRATA),
        "training_steps": TRAINING_STEPS,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
        },
        "temperature_candidates": list(TEMPERATURE_CANDIDATES),
        "temperature_selection_split": "calibration episodes 32-35 only",
        "model_seeds": list(MODEL_SEEDS),
        "matched_controls": list(MATCHED_CONTROLS),
        "permitted_inputs": list(PROPOSED_INPUT_FIELDS),
        "forbidden_inputs": list(FORBIDDEN_INPUT_FIELDS),
        "candidate_order_permutation_exact_invariance_required": True,
        "dynamic_atlas": "deterministic metric-space K-medoids with original-ID tie breaks",
        "generator": dict(GENERATOR_ARCHITECTURE),
        "execute_all_registered_experiments_after_gate_failure": True,
        "gate_failure_continuation_authority": "explicit user instruction",
        "maximum_local_training_gpus": 1,
        "pai_jobs_allowed_only_if_local_technically_impossible": True,
        "policy_or_vla_training_forbidden_except_nominal_generator": True,
        "final_dispositions": list(FINAL_DISPOSITIONS),
    }
