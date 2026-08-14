"""Frozen configuration for the preregistered Stage 3 NCER-AA audit.

Only a-priori choices belong in this module.  Calibration may select among
the explicitly listed candidates, but development and holdout observations
must never change these values.
"""

from __future__ import annotations

from . import config


PROJECT_ID = "r13_p15_ncer_aa_stage3"
OUTPUT_RELATIVE = "experiments/r13_p15_ncer_aa/stage3"
SCRATCH_ROOT = (
    "/mnt/cpfs/zbl-cpfs-new/dataset/leon/experiments/"
    "r13_p15_ncer_aa/stage3_work"
)

TASKS = config.TASKS
TASK_IDS = tuple(task["task_id"] for task in TASKS)
PHASES = config.PHASES
HORIZON = 4
ACTION_DIM = 7
CONTINUOUS_DIM = 24
SETTLE_STEPS = 3
ACTION_BANK_SIZE = 256
PRIMARY_K = 64

HISTORICAL_EPISODES = tuple(range(0, 16))
TRAIN_EPISODES = tuple(range(16, 32))
CALIBRATION_EPISODES = tuple(range(32, 36))
DEVELOPMENT_EPISODES = tuple(range(36, 40))
CONFIRMATION_EPISODES = tuple(range(40, 50))
SPLIT_EPISODES = {
    "historical": HISTORICAL_EPISODES,
    "train": TRAIN_EPISODES,
    "calibration": CALIBRATION_EPISODES,
    "development": DEVELOPMENT_EPISODES,
    "confirmation": CONFIRMATION_EPISODES,
}

GLOBAL_SEED = 13150300
SUPPORT_SEEDS = {
    "calibration": 13150332,
    "development": 13150336,
    "confirmation": 13150340,
}
DIRECTION_COUNT = 24
DIRECTION_FAMILIES = (
    "smooth_dct",
    "suffix_contact",
    "low_rank_temporal_action",
)
DIRECTION_FAMILY_COUNTS = {
    "smooth_dct": 8,
    "suffix_contact": 8,
    "low_rank_temporal_action": 8,
}
RADIUS_INTERVAL = (0.04, 0.12)
RADII_PER_DIRECTION = 2
SIGNS = (-1, 1)
MAX_CROSS_SPLIT_ABS_COSINE = 0.90
MAX_DIRECTION_COMPONENT = 0.80
BASE_ACTION_ABS_LIMIT = 0.895

CONTACT_SENSITIVE_TASKS = ("plate_push", "stove_turn_on", "wine_rack")
BOOTSTRAP_REPLICATES = 10000

# Pair construction is deliberately hard-negative heavy.  Contact-changing
# examples are added after the fixed rank/random strata and deduplicated.
PAIR_TOP_POSITIVES = 8
PAIR_HARD_NEGATIVE_START = 9
PAIR_HARD_NEGATIVE_END = 32
PAIR_RANDOM_NEGATIVES = 8
PAIR_CONTACT_CHANGE_MAX = 8

PREDICTOR_BATCH_SIZE = 4096
PAIR_BATCH_SIZE = 8192
MAX_EPOCHS = 60
PATIENCE = 8
MIN_DELTA = 1e-5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
CONTACT_LOSS_WEIGHT = 0.25
EMBEDDING_DIM = 32
VECTOR_HIDDEN_CANDIDATES = ((128, 128), (256, 256))
TEMPORAL_HIDDEN_CANDIDATES = (96, 160)
ENSEMBLE_SIZE = 3
SOFT_EXPERTS = 4
PAIR_HIDDEN = (192, 192)
ROUTER_HIDDEN = 64
CONTROL_ENSEMBLE_SIZE = 1
B4_LATENT_DIM = 32
B4_HIDDEN = (128, 128)
B5_NEIGHBOR_CANDIDATES = (3, 5, 9)
B5_BANDWIDTH_CANDIDATES = (0.5, 1.0, 2.0)

# C0 intentionally reproduces the Stage 2 NCEA architecture and training
# contract, while using the new frozen train/calibration split.
C0_HIDDEN = (128, 128)
C0_ENSEMBLE_SIZE = 5
C0_MAX_EPOCHS = 160
C0_PATIENCE = 20
C0_BATCH_SIZE = 512
C0_LEARNING_RATE = 1e-3
C0_WEIGHT_DECAY = 1e-5

# The C6 auxiliary routing target is derived entirely from permitted current
# and past observables. It is never a demonstration phase or future label.
SOFT_ROUTING_LABEL = (
    "2*current_observable_contact + "
    "1[previous task-progress delta is nonnegative]"
)

# Calibration selects one entire tuple.  Development and holdout cannot
# choose lambda values or temperatures.
RANKING_OBJECTIVE_CANDIDATES = (
    {
        "name": "balanced",
        "lambda_distance": 1.0,
        "lambda_pairwise": 0.50,
        "lambda_listwise": 0.50,
        "tau_true": 0.15,
        "tau_model": 0.15,
    },
    {
        "name": "distance_heavy",
        "lambda_distance": 1.0,
        "lambda_pairwise": 0.25,
        "lambda_listwise": 0.25,
        "tau_true": 0.15,
        "tau_model": 0.20,
    },
    {
        "name": "ranking_heavy",
        "lambda_distance": 0.50,
        "lambda_pairwise": 1.0,
        "lambda_listwise": 0.50,
        "tau_true": 0.10,
        "tau_model": 0.10,
    },
    {
        "name": "listwise_heavy",
        "lambda_distance": 0.50,
        "lambda_pairwise": 0.50,
        "lambda_listwise": 1.0,
        "tau_true": 0.20,
        "tau_model": 0.20,
    },
)

METHODS = (
    "B1_centered_covariance",
    "B2_current_contact_kmeans",
    "B2_PRIV_hard_phase_kmeans",
    "B3_dynamic_action_medoids",
    "B4_state_action_vq",
    "B5_local_knn_consequence",
    "C0_stage2_ncea_reproduction",
    "C1_NC_VECTOR",
    "C2_NC_TEMPORAL_VECTOR",
    "C3_NC_BIENCODER",
    "C4_NC_PAIR_RANKER",
    "C5_NCER_AA",
    "C6_SOFT_MIXTURE_NCER_AA",
    "O_FULL_true_effect_full_bank",
    "O_K64_true_effect_atlas",
)

MECHANISM_CONTROLS = (
    "no_nominal_action",
    "nominal_action_shuffled_within_task",
    "state_shuffled_within_task",
    "joint_state_nominal_shuffled_within_task",
    "history_shuffled",
    "consequence_labels_shuffled",
    "soft_routing_labels_shuffled",
    "action_only_pair_ranker",
    "candidate_order_permutation",
)

GATES = {
    "A": {
        "oracle_relative_gain_min": 0.20,
        "tasks_improved_min": 3,
        "contact_sensitive_tasks_improved_min": 2,
        "failure_disposition": "REJECT_CONSEQUENCE_EQUIVALENCE_ON_STRICT_SUPPORT",
    },
    "B": {
        "oracle_regret_relative_gain_min": 0.25,
        "ndcg16_absolute_gain_min": 0.10,
        "recall8_min": 0.50,
        "tasks_improved_min": 3,
        "contact_sensitive_tasks_improved_min": 2,
        "joint_state_nominal_shuffle_gain_retention_max": 0.25,
        "state_shuffle_gain_retention_max": 0.50,
        "nominal_shuffle_gain_retention_max": 0.50,
        "label_shuffle_must_not_reproduce_gain": True,
        "candidate_permutation_exact_invariance": True,
        "failure_disposition": "ORACLE_ONLY_NO_LEARNABLE_RANKER",
    },
    "C": {
        "realized_relative_gain_min": 0.10,
        "oracle_gap_fraction_closed_min": 0.25,
        "tasks_improved_min": 3,
        "contact_sensitive_tasks_improved_min": 2,
        "bowl_on_plate_max_degradation": 0.05,
        "action_rmse_degradation_max": 0.20,
        "contact_preservation_drop_max_points": 0.01,
        "normalized_utilization_min": 0.25,
        "clipping_rate_max": 0.0,
        "privileged_phase_forbidden": True,
        "failure_disposition": "LEARNABLE_RETRIEVAL_BUT_ALPHABET_COMPRESSION_FAILED",
        "pass_disposition": "DEVELOPMENT_PASSED_CONFIRMATION_REQUIRED",
    },
    "GO": {
        "pooled_gain_min": 0.10,
        "paired_ci_lower_bound_exclusive": 0.0,
        "tasks_improved_min": 3,
        "contact_sensitive_tasks_improved_min": 2,
        "shuffle_gain_retention_max": 0.25,
        "action_rmse_degradation_max": 0.20,
        "contact_preservation_drop_max_points": 0.01,
        "normalized_utilization_min": 0.25,
        "clipping_rate_max": 0.0,
        "pass_disposition": "GO_TO_SMALL_BC",
        "failure_disposition": "CONFIRMATION_FAILED",
    },
}

# User-directed amendment, frozen before Stage 3 results.  It changes only
# execution completeness, never the evidentiary meaning of a failed gate.
EXECUTION_AMENDMENT = {
    "execute_all_development_methods_after_gate_failure": True,
    "execute_episodes_40_49_after_gate_failure": True,
    "failed_gate_holdout_label": "FORCED_EXPLORATORY_HOLDOUT",
    "failed_gate_holdout_is_confirmatory": False,
    "failed_gate_holdout_can_unlock_small_bc": False,
    "reason": (
        "The user explicitly required every planned experiment to be run and "
        "forbade gate-triggered early stopping."
    ),
}

# Frozen before any method metric was computed or inspected.  During the
# initial pre-result replay implementation, two executions of one fixed
# confirmation-support perturbation were made at each confirmation snapshot.
# They were used only for equality/order checks, but this still violates the
# literal "do not execute confirmation target branches" rule.  The incident is
# preserved in PRE_RESULT_PROTOCOL_INCIDENT.json and conservatively disables
# an untouched-confirmation / GO claim for this Stage 3 run.
CONFIRMATION_INTEGRITY_AMENDMENT = {
    "incident_id": "stage3-pre-result-confirmation-replay-001",
    "strict_untouched_confirmation_available": False,
    "all_development_gates_pass_label": "PRE_RESULT_REPLAY_EXPOSED_HOLDOUT",
    "any_development_gate_fail_label": "FORCED_EXPLORATORY_HOLDOUT",
    "go_to_small_bc_available": False,
    "reason": (
        "A fixed confirmation support perturbation was executed twice per "
        "snapshot for deterministic replay before development gates. No "
        "effect distance, method score, or selected action was computed or "
        "inspected, but branch execution alone breaks the frozen untouched rule."
    ),
}


def split_for_episode(episode_id):
    episode_id = int(episode_id)
    for name, episodes in SPLIT_EPISODES.items():
        if episode_id in episodes:
            return name
    raise KeyError(episode_id)


def model_definitions():
    """Return the fully frozen method and model-selection contract."""
    return {
        "project_id": PROJECT_ID,
        "primary_metric": "BALANCED_TASK_EFFECT",
        "primary_k": PRIMARY_K,
        "candidate_bank_size": ACTION_BANK_SIZE,
        "models": {
            "B1": "Train-only covariance-whitened residual K=64 medoids.",
            "B2": "Current observable contact-conditioned residual K=64 k-means medoids.",
            "B2_PRIV": "Demonstration hard-phase residual k-means; diagnostic only.",
            "B3": "Per-state valid-bank deterministic action-space FPS medoids.",
            "B4": "State-conditioned action-only VQ trained for action reconstruction.",
            "B5": "Local kernel consequence interpolation over state, nominal chunk, and residual.",
            "C0": "Exact Stage 2 NCEA input and smooth-L1 plus contact loss reproduction.",
            "C1": "Observable history, nominal chunk and residual to balanced consequence vector/contact logits.",
            "C2": "C1 with separate small temporal encoders for nominal and residual Hx6 sequences.",
            "C3": "Nominal-conditioned bi-encoder with symmetric Euclidean effect-equivalence distance.",
            "C4": "Nominal-conditioned symmetric pair cross-encoder with exact self-distance zero.",
            "C5": "C3 effect-space K=64 FPS/medoids followed by C4 reranking.",
            "C6": "Observable-history soft mixture of pair-ranker experts; no true phase routing.",
        },
        "permitted_inputs": [
            "current observable state vector and mask",
            "previous two observable state deltas and masks",
            "previous two executed actions and availability masks",
            "current observable contact indicator",
            "nominal H=4 action chunk",
            "target/candidate residual chunks",
            "task identity",
        ],
        "forbidden_inputs": [
            "future state or consequence",
            "target/candidate simulator outcome at inference",
            "episode outcome",
            "target ID or bank ID",
            "demonstration-derived future phase label",
            "confirmation result",
            "post-execution oracle contact mode",
        ],
        "training": {
            "vector_hidden_candidates": [list(value) for value in VECTOR_HIDDEN_CANDIDATES],
            "temporal_hidden_candidates": list(TEMPORAL_HIDDEN_CANDIDATES),
            "embedding_dim": EMBEDDING_DIM,
            "ensemble_size": ENSEMBLE_SIZE,
            "soft_experts": SOFT_EXPERTS,
            "pair_hidden": list(PAIR_HIDDEN),
            "router_hidden": ROUTER_HIDDEN,
            "control_ensemble_size": CONTROL_ENSEMBLE_SIZE,
            "soft_routing_label": SOFT_ROUTING_LABEL,
            "soft_routing_label_uses_future_or_phase": False,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "ranking_objective_candidates": list(RANKING_OBJECTIVE_CANDIDATES),
            "selection_data": "calibration episodes 32-35 only",
            "selection_rule": (
                "lowest calibration mean true oracle regret; then highest "
                "NDCG@16; then lowest candidate tuple order"
            ),
            "architecture_selection": (
                "one frozen seed evaluates each candidate; two additional "
                "members are trained only for the selected candidate"
            ),
            "C0_stage2_reproduction": {
                "hidden": list(C0_HIDDEN),
                "ensemble_size": C0_ENSEMBLE_SIZE,
                "max_epochs": C0_MAX_EPOCHS,
                "patience": C0_PATIENCE,
                "batch_size": C0_BATCH_SIZE,
                "learning_rate": C0_LEARNING_RATE,
                "weight_decay": C0_WEIGHT_DECAY,
                "loss": "smooth_l1 normalized effect + 0.25 contact cross entropy",
            },
            "B4": {
                "hidden": list(B4_HIDDEN),
                "latent_dim": B4_LATENT_DIM,
                "consequence_labels_used": False,
            },
            "B5": {
                "neighbors": list(B5_NEIGHBOR_CANDIDATES),
                "bandwidths": list(B5_BANDWIDTH_CANDIDATES),
                "selection": "calibration BALANCED_TASK_EFFECT",
            },
        },
        "pair_construction": {
            "oracle_top_positives": PAIR_TOP_POSITIVES,
            "hard_negative_ranks_inclusive": [PAIR_HARD_NEGATIVE_START, PAIR_HARD_NEGATIVE_END],
            "random_negatives": PAIR_RANDOM_NEGATIVES,
            "contact_change_max": PAIR_CONTACT_CHANGE_MAX,
            "easy_random_negatives_primary": False,
            "symmetric": True,
            "self_distance_target": 0.0,
        },
        "mechanism_controls": list(MECHANISM_CONTROLS),
        "candidate_permutation": {
            "required_exact_selected_bank_index_invariance": True,
            "tie_break": "lowest frozen bank index",
        },
        "comparison_selection": {
            "strongest_deployable_baseline": (
                "lowest pooled calibration BALANCED_TASK_EFFECT among B1-B5 and C0"
            ),
            "strongest_learned_or_action_baseline_for_gate_b": (
                "lowest pooled calibration oracle regret among B3-B5, C0-C3"
            ),
            "development_or_holdout_cannot_select_a_comparator": True,
        },
        "gates": GATES,
        "execution_amendment": EXECUTION_AMENDMENT,
        "confirmation_integrity_amendment": CONFIRMATION_INTEGRITY_AMENDMENT,
    }
