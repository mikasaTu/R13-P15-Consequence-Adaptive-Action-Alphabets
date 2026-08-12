"""Frozen experiment configuration.

The values in this module are intentionally plain Python constants so the
formal launcher can hash and inspect them without a configuration framework.
"""

from __future__ import annotations

import os


PROJECT_ID = "r13_p15_caaa_v2_libero_stage1"
UPSTREAM_LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
LIBERO_SOURCE_DEFAULT = (
    "/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/LIBERO-original"
)
LIBERO_ENV_DEFAULT = (
    "/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero-original"
)
DATASET_ROOT_DEFAULT = (
    "/mnt/cpfs/zbl-cpfs-new/dataset/leon/embodied_benchmark/datasets/LIBERO"
)
OUTPUT_ROOT_DEFAULT = (
    "/mnt/cpfs/zbl-cpfs-new/dataset/leon/experiments/"
    "r13_p15_caaa_v2/stage1"
)

SUITE = "libero_goal"
CONTROL_MODE = "OSC_POSE"
CONTROL_FREQUENCY_HZ = 20
ACTION_DIM = 7
CONTINUOUS_ACTION_INDICES = (0, 1, 2, 3, 4, 5)
GRIPPER_ACTION_INDEX = 6
CHUNK_HORIZON = 4
CHUNK_CONTINUOUS_DIM = 24
N_EPISODES = 16
N_PHASES = 4
PHASES = ("free_space", "pre_contact", "contact_onset", "post_contact")
TRAIN_EPISODES = tuple(range(0, 8))
CALIBRATION_EPISODES = tuple(range(8, 12))
TEST_EPISODES = tuple(range(12, 16))
PERTURBATION_DIRECTIONS = 24
PERTURBATION_RADII = (0.05, 0.10)
PERTURBATION_SIGNS = (-1, 1)
SETTLE_STEPS = 3
PRIMARY_K = 64
SENSITIVITY_K = (32, 128)
K_VALUES = (32, 64, 128)
GLOBAL_SEED = 13150015
BOOTSTRAP_REPLICATES = 10000

RIDGE_GRID = (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1)
SINGULAR_CUTOFF_GRID = (1e-4, 1e-3, 1e-2, 5e-2)
METRIC_REGULARIZATION_GRID = (1e-8, 1e-6, 1e-4, 1e-2)
PCA_RANK_GRID = (4, 8, 12, 16, 20, 24)

TASKS = (
    {
        "task_id": "bowl_on_plate",
        "task_name": "put_the_bowl_on_the_plate",
        "role": "low_constraint_pick_place_control",
        "primary_object": "akita_black_bowl_1",
        "target_object": "plate_1",
        "target_contact_parent": "plate_1",
        "goal_predicate": "on",
    },
    {
        "task_id": "plate_push",
        "task_name": "push_the_plate_to_the_front_of_the_stove",
        "role": "sustained_sliding_contact",
        "primary_object": "plate_1",
        "target_object": "main_table_stove_front_region",
        "target_contact_parent": None,
        "goal_predicate": "on",
    },
    {
        "task_id": "stove_turn_on",
        "task_name": "turn_on_the_stove",
        "role": "small_articulated_contact",
        "primary_object": "flat_stove_1",
        "target_object": "flat_stove_1",
        "target_contact_parent": None,
        "goal_predicate": "turnon",
    },
    {
        "task_id": "wine_rack",
        "task_name": "put_the_wine_bottle_on_the_rack",
        "role": "precision_oriented_receptacle",
        "primary_object": "wine_bottle_1",
        "target_object": "wine_rack_1_top_region",
        "target_contact_parent": "wine_rack_1",
        "goal_predicate": "on",
    },
)

METHODS = (
    "euclidean_farthest",
    "covariance_mahalanobis",
    "global_kmeans",
    "phase_conditioned_kmeans",
    "pca_kmeans",
    "old_diagonal_sensitivity",
    "random_spd",
    "permuted_j",
    "caaa_v2",
)


def task_by_id(task_id):
    for task in TASKS:
        if task["task_id"] == task_id:
            return dict(task)
    raise KeyError(task_id)


def resolved_paths(args=None):
    """Resolve paths from CLI arguments or the frozen defaults."""
    get = lambda name, default: getattr(args, name, None) or os.environ.get(
        "CAAA_" + name.upper(), default
    )
    return {
        "libero_source": get("libero_source", LIBERO_SOURCE_DEFAULT),
        "libero_env": get("libero_env", LIBERO_ENV_DEFAULT),
        "dataset_root": get("dataset_root", DATASET_ROOT_DEFAULT),
        "output_root": get("output_root", OUTPUT_ROOT_DEFAULT),
    }
