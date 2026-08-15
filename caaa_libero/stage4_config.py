"""Frozen pre-result contract for the Stage 4 CR-TR-C3 audit.

This module contains only choices that are allowed to exist before any Stage 4
method result is inspected.  Calibration may choose only from the explicitly
listed candidates.  Development, historical exploratory, and fresh
confirmation outcomes may never alter these values.
"""

from __future__ import annotations

from . import config
from .stage3_config import RANKING_OBJECTIVE_CANDIDATES


PROJECT_ID = "r13_p15_cr_trca_stage4"
OUTPUT_RELATIVE = "experiments/r13_p15_cr_trca/stage4"
HISTORICAL_REPOSITORY_ROOT = (
    "/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r13-p15-caaa-v2-libero"
)
HISTORICAL_STAGE3_RELATIVE = "experiments/r13_p15_ncer_aa/stage3"
STAGE2_ACTION_BANK_RELATIVE = "experiments/r13_p15_ncea/stage2/action_bank.npz"
SCRATCH_ROOT = (
    "/mnt/cpfs/zbl-cpfs-new/dataset/leon/experiments/"
    "r13_p15_cr_trca/stage4_work"
)

# Immutable input history.  These values are assertions, not values learned by
# the freeze command.  A mismatch stops Stage 4 before any result is produced.
STAGE4_INPUT_COMMIT = "beb63576e91307260b64687e58ea99e6da93c478"
STAGE4_INPUT_TREE = "cd3f67314435103405a4df8fa597076a5a47d386"
HISTORICAL_STAGE_TREES = {
    "stage1": "e00831263060b67d05f11483bf17fe42a4b57dbe",
    "stage1_5": "cd55cd0af69e8c8c1955182f7b459f712759e267",
    "stage2": "ba2cdd070e8d27bcd69a79cefdc8259770b1af27",
    "stage3": "2ca0d9bd79d7486bfa6ac1876ddae349b86008eb",
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
        "published_commit": STAGE4_INPUT_COMMIT,
    },
}
EXPECTED_HISTORICAL_HASHES = {
    "action_bank.npz": "d41f0dc748866cae3ef151d9f16e39789485d6e633a0a88f62fe4c570661600b",
    "model_scalers.npz": "df8e0a9f4ea24c67e9305d8459a40ea50745c4b1a75060743d3798acbbcafda5",
    "C3_NC_BIENCODER_member_0.pt": "b10f3e5def68a89ad33918fbbd9c0435f8de36215a1bf74d6e1f95e42e5d6b5d",
    "C3_NC_BIENCODER_member_1.pt": "539c3ef4c2e4854265b61791f6f1c2e949b6ed86120c10d7df464dbf6fc8676b",
    "C3_NC_BIENCODER_member_2.pt": "fb314c29cda098357de70996dd9ae1071812dc2aa51634fa8ad4c8e664c82ced",
    "C4_NC_PAIR_RANKER_member_0.pt": "e68d43f3b5374bcff9f13ee3639dc934d5483bef4ca57a6fbc5145414c2cc1cd",
    "C4_NC_PAIR_RANKER_member_1.pt": "83c725da1b00af57b7780cd3422707db059e09f5f2f84589a43aba412881eff7",
    "C4_NC_PAIR_RANKER_member_2.pt": "5b98effd13607de8a5e9e37c42ac42a031a6835292d6485c8bde549bc1ca7c56",
    "development_quantization.csv": "5c1d77b24aaeb6c20703ec7fda8072eea15c70fa30afef25d4f2d5c920c40e81",
    "confirmation_quantization.csv": "8e08469f688a19a632a17cb622e2b42756cde25c61d3cbe2c55348e85c3e6776",
    "retrieval_metrics.csv": "822705b83f9132a4d5015cf2cd0d1209c689c0e2d3b43b189a4b2e3e5c5f34db",
    "development_gate.json": "21a5f633762b3c959643048d6aa8b507fac5665de832900c6e4f601324753dd4",
    "bootstrap_results.json": "718a96718ff9f66c7ca15869dacc037ec632cc5bdbdbd0f16eaaa03f22aa6cc4",
    "STAGE3_REPORT.md": "623e14a5473d3e67b175e9fdefae6951a9634c18b8bb227cab14c8dc355f7bde",
}

TASKS = config.TASKS
TASK_IDS = tuple(task["task_id"] for task in TASKS)
CONTACT_SENSITIVE_TASKS = ("plate_push", "stove_turn_on", "wine_rack")
PHASES = config.PHASES
HORIZON = 4
ACTION_DIM = 7
CONTINUOUS_DIM = 24
SETTLE_STEPS = 3
ACTION_BANK_SIZE = 256
PRIMARY_K = 64

TRAIN_EPISODES = tuple(range(16, 32))
CALIBRATION_EPISODES = tuple(range(32, 36))
DEVELOPMENT_EPISODES = tuple(range(36, 40))
HISTORICAL_EXPLORATORY_EPISODES = tuple(range(40, 50))

# Exactly 4 tasks x 16 episodes x 4 phases x 3 states = 768 independent
# training states.  The state selector must exclude every Stage 3 snapshot.
TRAIN_STATES_PER_EPISODE_PHASE = 3
TRAIN_STATE_COUNT = (
    len(TASKS)
    * len(TRAIN_EPISODES)
    * len(PHASES)
    * TRAIN_STATES_PER_EPISODE_PHASE
)
BASE_ACTION_ABS_LIMIT = 0.875

GLOBAL_SEED = 13150400
TRAIN_STATE_SELECTION_SEED = 13150401
TRAIN_SUPPORT_SEED = 13150402
REVERSAL_PAIR_SEED = 13150403
BOOTSTRAP_SEED = 13150404
CONFIRMATION_SELECTION_SEED = 13150405

SUPPORT_DIRECTION_FAMILIES = (
    "smooth_dct",
    "suffix_contact",
    "low_rank_temporal_action",
)
SUPPORT_DIRECTIONS_PER_FAMILY = 8
# 24 directions x two radii x two antithetic signs.  Keeping this explicit
# prevents confusing the number of directions with the number of target
# branches in manifests and completeness checks.
SUPPORT_DIRECTION_COUNT = (
    len(SUPPORT_DIRECTION_FAMILIES) * SUPPORT_DIRECTIONS_PER_FAMILY
)
SUPPORT_RADII = (0.06, 0.10)
SUPPORT_SIGNS = (-1, 1)
SUPPORT_TARGET_COUNT = (
    SUPPORT_DIRECTION_COUNT * len(SUPPORT_RADII) * len(SUPPORT_SIGNS)
)
SUPPORT_MAX_COMPONENT = 0.80

# Stage 3 C3 seeds are reused for the independent objective re-selection so
# that the only change is the calibration selector, not seed luck.
C3_RESELECT_SEEDS = (56229435, 2279153700, 2652429101)
CR_C3_SEEDS = (13150417, 13150429, 13150443)
CR_CONTROL_SEEDS = CR_C3_SEEDS
CR_MODEL_FAMILIES = ("CR_C3_SHARED", "CR_C3_GROUP")
CR_CONTROLS = (
    "ACTION_ONLY",
    "CONTEXT_SHUFFLED",
    "NOMINAL_SHUFFLED",
    "CONSEQUENCE_LABEL_SHUFFLED",
    "REVERSAL_LABEL_SHUFFLED",
    "NO_REVERSAL_LOSS",
)

EMBEDDING_DIM = 32
GROUP_EMBEDDING_DIM = 16
CONTEXT_HIDDEN = (192, 128)
ACTION_HIDDEN = (128, 96)
CR_MAX_EPOCHS = 30
CR_BATCH_SIZE = 16
CR_LEARNING_RATE = 3e-4
CR_WEIGHT_DECAY = 1e-5
CR_LISTWISE_WEIGHT = 1.0
CR_PAIRWISE_WEIGHT = 0.5
CR_REVERSAL_WEIGHT = 0.5
CR_TAU_TRUE = 0.15
CR_TAU_MODEL = 0.15
CR_PAIR_MARGIN_QUANTILE = 0.25
CR_REVERSAL_MARGIN_QUANTILE = 0.25
CR_REVERSALS_PER_TASK_PHASE = 256

TRUST_REGION_L = (8, 16, 32, 64)
BOUNDED_CORRECTION_GAMMA = (0.0, 0.1, 0.2)
BOOTSTRAP_REPLICATES = 10000

CONTEXT_INTERVENTIONS = (
    "correct_context",
    "nominal_zeroed",
    "nominal_shuffled_within_task",
    "state_mask_contact_shuffled_within_task",
    "history_actions_masks_shuffled_within_task",
    "state_and_nominal_jointly_shuffled",
    "all_context_zeroed_action_pair_retained",
)

FAILURE_DECOMPOSITION_METHODS = (
    "B2",
    "O_FULL",
    "O_K64",
    "C3_FULL",
    "C3_K64",
    "C5",
)

DEVELOPMENT_METHODS = (
    "B2",
    "O_FULL",
    "O_K64",
    "FROZEN_C3_FULL",
    "FROZEN_C3_K64",
    "C5",
    "C3_RESELECT_FULL",
    "C3_RESELECT_FPS64",
    "C3_RESELECT_KMEDOIDS64",
    "CR_C3_FULL",
    "CR_C3_K64",
    "CR_TR_C3_K64",
    "ACTION_ONLY_TR_K64",
    "SHUFFLED_EFFECT_TR_K64",
)

GATES = {
    "A": {
        "oracle_relative_gain_min": 0.20,
        "tasks_improved_min": 3,
        "contact_sensitive_tasks_improved_min": 2,
        "failure_disposition": "REJECT_CONSEQUENCE_HEADROOM",
    },
    "B": {
        "episodes_36_39_gain_min": 0.05,
        "historical_episodes_40_49_gain_min": 0.05,
        "pooled_development_gain_min": 0.08,
        "tasks_improved_min": 3,
        "contact_sensitive_tasks_improved_min": 2,
        "frozen_c3_gain_min": 0.05,
        "reversal_accuracy_gain_points_min": 0.10,
        "joint_state_nominal_shuffle_gain_retention_max": 0.50,
        "action_only_must_not_reproduce": True,
        "label_shuffled_must_not_reproduce": True,
        "failure_disposition": "REJECT_LEARNED_CONSEQUENCE_METRIC",
        "context_independent_disposition": "STATIC_EFFECT_METRIC_ONLY",
    },
    "C": {
        "realized_relative_gain_min": 0.08,
        "full_gain_retention_min": 0.75,
        "tasks_improved_min": 3,
        "contact_sensitive_tasks_improved_min": 2,
        "action_rmse_degradation_max": 0.20,
        "contact_preservation_drop_max_points": 0.01,
        "normalized_utilization_min": 0.25,
        "clipping_rate_max": 0.0,
        "full_only_disposition": "PIVOT_TO_CONSEQUENCE_RETRIEVAL_STEERING",
        "contact_only_disposition": "NARROW_TO_CONTACT_CONSEQUENCE_METRIC",
    },
    "GO": {
        "pooled_gain_min": 0.10,
        "paired_ci_lower_bound_exclusive": 0.0,
        "tasks_improved_min": 3,
        "contact_sensitive_tasks_improved_min": 2,
        "action_rmse_degradation_max": 0.20,
        "contact_preservation_drop_max_points": 0.01,
        "normalized_utilization_min": 0.25,
        "clipping_rate_max": 0.0,
        "context_shuffled_gain_retention_max": 0.25,
        "all_seed_directions_same": True,
        "pass_disposition": "GO_TO_SMALL_BC",
        "failure_disposition": "CONFIRMATION_FAILED",
    },
}

FINAL_DISPOSITIONS = (
    "REJECT_CONSEQUENCE_HEADROOM",
    "REJECT_LEARNED_CONSEQUENCE_METRIC",
    "STATIC_EFFECT_METRIC_ONLY",
    "PIVOT_TO_CONSEQUENCE_RETRIEVAL_STEERING",
    "NARROW_TO_CONTACT_CONSEQUENCE_METRIC",
    "BLOCKED_NO_FRESH_CONFIRMATION",
    "CONFIRMATION_FAILED",
    "GO_TO_SMALL_BC",
)


def method_definitions():
    """Return the complete pre-result method and selection contract."""
    return {
        "project_id": PROJECT_ID,
        "primary_metric": "BALANCED_TASK_EFFECT",
        "primary_k": PRIMARY_K,
        "candidate_bank_size": ACTION_BANK_SIZE,
        "controller": {
            "robot": "Panda",
            "mode": "OSC_POSE",
            "frequency_hz": 20,
            "horizon": HORIZON,
            "settle_steps": SETTLE_STEPS,
            "continuous_action_semantics": "4x6 normalized OSC_POSE coordinates",
            "gripper_semantics": "copied stepwise from the nominal demonstration",
        },
        "historical_methods": {
            "B2": "Frozen Stage 3 current-contact K=64 residual k-means.",
            "O_FULL": "Per-state true-effect nearest executable member over M=256.",
            "O_K64": "Per-state true-effect FPS K=64 atlas and true-effect decoding.",
            "C3_FULL": "Frozen Stage 3 3-member C3 embedding and full-bank nearest neighbor.",
            "C3_K64": "Frozen Stage 3 C3 predicted-embedding FPS K=64 atlas and C3 decoding.",
            "C5": "Frozen C3 K=64 atlas followed by frozen Stage 3 C4 reranking.",
        },
        "new_methods": {
            "C3_RESELECT_FULL": "Stage 3 architecture, C3-alone calibration objective selection, full bank.",
            "C3_RESELECT_FPS64": "C3-alone reselected embedding with deterministic predicted-space FPS.",
            "C3_RESELECT_KMEDOIDS64": "C3-alone reselected embedding with deterministic predicted-space k-medoids.",
            "CR_C3_SHARED": "One context-conditioned effect embedding trained with full-bank listwise, pairwise, and context-reversal losses.",
            "CR_C3_GROUP": "Five factorized consequence-group embeddings with frozen equal-weight mean distance.",
            "CR_TR_C3_K64": "Whitened action-local top-L filter over executable atlas members, then CR-C3 distance.",
        },
        "failure_decomposition": {
            "oracle_bank_compression_loss": "O_K64 - O_FULL",
            "learned_metric_loss": "C3_FULL - O_FULL",
            "learned_compression_loss": "C3_K64 - C3_FULL",
            "c4_override_loss": "C5 - C3_K64",
        },
        "permitted_inputs": [
            "current observable state and availability mask",
            "two prior observable state deltas and masks",
            "two previous executed actions and masks",
            "current observable contact",
            "nominal H=4 action chunk",
            "task identity",
            "target or candidate residual",
        ],
        "forbidden_inputs": [
            "future state or consequence at inference",
            "candidate or target simulator outcome at inference",
            "demonstration phase label",
            "episode outcome",
            "target ID",
            "candidate ID",
        ],
        "c3_objective_candidates": [dict(value) for value in RANKING_OBJECTIVE_CANDIDATES],
        "c3_reselection": {
            "screening_seed": C3_RESELECT_SEEDS[0],
            "ensemble_seeds": list(C3_RESELECT_SEEDS),
            "selection": [
                "lowest calibration C3_FULL oracle regret",
                "highest calibration C3_FULL NDCG@16",
                "lowest frozen tuple index",
            ],
            "development_or_historical_exploratory_selection_forbidden": True,
        },
        "cr_c3": {
            "families": list(CR_MODEL_FAMILIES),
            "seeds": list(CR_C3_SEEDS),
            "controls": list(CR_CONTROLS),
            "controls_use_same_seeds_architecture_parameters_and_budget": True,
            "epochs": CR_MAX_EPOCHS,
            "batch_size_queries": CR_BATCH_SIZE,
            "learning_rate": CR_LEARNING_RATE,
            "weight_decay": CR_WEIGHT_DECAY,
            "loss_weights": {
                "full_bank_listwise": CR_LISTWISE_WEIGHT,
                "pairwise": CR_PAIRWISE_WEIGHT,
                "context_reversal": CR_REVERSAL_WEIGHT,
            },
            "temperatures": {"true": CR_TAU_TRUE, "model": CR_TAU_MODEL},
            "reversal_loss": "softplus(m+d_s1(t,i)-d_s1(t,j)) + softplus(m+d_s2(t,j)-d_s2(t,i))",
            "family_selection": [
                "lowest calibration full-bank oracle regret",
                "highest calibration context-reversal accuracy",
                "highest calibration NDCG@16",
                "lowest frozen family index",
            ],
        },
        "trust_region": {
            "distance": "train-covariance-whitened residual action distance",
            "candidates_l": list(TRUST_REGION_L),
            "l_64_is_no_trust_region_control": True,
            "selection": [
                "lowest calibration realized BALANCED_TASK_EFFECT",
                "lowest calibration action reconstruction RMSE",
                "smallest L",
            ],
            "outputs_executable_bank_member": True,
            "clipping_or_action_synthesis_forbidden": True,
        },
        "bounded_correction_diagnostic": {
            "formula": "d_C3 * exp(gamma * tanh(g_symmetric))",
            "gamma": list(BOUNDED_CORRECTION_GAMMA),
            "gamma_0_exact_sham": True,
            "calibration_only": True,
            "required_for_success": False,
        },
        "context_interventions": list(CONTEXT_INTERVENTIONS),
        "gates": GATES,
        "final_dispositions": list(FINAL_DISPOSITIONS),
        "execute_all_experiments_after_gate_failure": True,
        "pai_jobs_allowed": False,
        "maximum_local_training_gpus": 1,
        "policy_training_allowed": False,
    }
