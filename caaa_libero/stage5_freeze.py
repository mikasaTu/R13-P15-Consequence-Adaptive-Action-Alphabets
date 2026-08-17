"""Freeze Stage 5 bindings, local bank, model contract and fresh seeds.

The command is deliberately pre-result: it reads only immutable Stage 1--4
artifacts and train/calibration protocol inputs.  It never computes or opens a
Stage 5 development score.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter

import numpy as np

from . import config
from .pipeline import utc_now
from .stage2 import _array_hash
from .stage3 import _candidate_direction
from .stage5_config import (
    BRANCH,
    CALIBRATION_EPISODES,
    CONTACT_SENSITIVE_TASKS,
    DEVELOPMENT_EPISODES,
    FINAL_DISPOSITIONS,
    FRESH_TARGET_COUNT,
    FRESH_TARGET_DIRECTION_COUNT,
    FRESH_TARGET_FAMILIES,
    FRESH_TARGET_MAX_COMPONENT,
    FRESH_TARGET_RADII,
    FRESH_TARGET_SEED,
    FRESH_TARGET_SIGNS,
    GENERATOR_ARCHITECTURE,
    GENERATOR_REQUIRED_SUCCESSES_PER_TASK,
    GENERATOR_ROLLOUT_SEEDS_PER_TASK,
    GENERATOR_SACRIFICIAL_SEEDS,
    GENERATOR_TRAIN_SEED,
    GENERATOR_TRAIN_EPISODES,
    GATES,
    HISTORICAL_EVIDENCE,
    HISTORICAL_EXPLORATORY_EPISODES,
    HISTORICAL_STAGE_TREES,
    HORIZON,
    LOCAL_BANK_BALANCE_AXES,
    LOCAL_BANK_COVARIANCE_REGULARIZATION,
    LOCAL_BANK_NEAR_DUPLICATE_TOLERANCE,
    LOCAL_BANK_SIZE,
    LOCAL_BANK_SEED,
    MINIMUM_VALID_CANDIDATES,
    OUTPUT_RELATIVE,
    PHASES,
    PREREGISTRATION_BASE_COMMIT,
    PREREGISTRATION_BASE_TREE,
    PRIMARY_K,
    PROJECT_ID,
    SCRATCH_ROOT,
    SOURCE_BANK_SIZE,
    STAGE4_RESULT_COMMIT,
    STAGE4_TREE,
    TASKS,
    TASK_IDS,
    TRAIN_EPISODES,
    model_definitions,
    rollout_seeds,
)
from .storage import atomic_json, atomic_npz, sha256_file, sha256_tree


STAGE_PATHS = {
    "stage1": "experiments/r13_p15_caaa_v2/stage1",
    "stage1_5": "experiments/r13_p15_caaa_v2/stage1_5",
    "stage2": "experiments/r13_p15_ncea/stage2",
    "stage3": "experiments/r13_p15_ncer_aa/stage3",
    "stage4": "experiments/r13_p15_cr_trca/stage4",
}
STAGE4_BOUND_FILES = (
    "PREREGISTRATION.md",
    "HISTORICAL_BINDING.json",
    "METHOD_DEFINITIONS.json",
    "TRAINING_STATE_MANIFEST.json",
    "expanded_training_collection.json",
    "expanded_training_dataset.json",
    "stage4_scalers.npz",
    "training_support_bank.npz",
    "CONTEXT_REVERSAL_PAIRS.parquet",
    "MODEL_SELECTION.json",
    "DEVELOPMENT_RETRIEVAL.csv",
    "DEVELOPMENT_REALIZED.csv",
    "DEVELOPMENT_GATE.json",
    "HISTORICAL_EXPLORATORY_RETRIEVAL.csv",
    "HISTORICAL_EXPLORATORY_REALIZED.csv",
    "FRESH_STATE_INVENTORY.json",
    "FRESH_CONFIRMATION_SPLIT.json",
    "fresh_collection_manifest.json",
    "fresh_confirmation_dataset.json",
    "CONFIRMATION_RETRIEVAL.csv",
    "CONFIRMATION_REALIZED.csv",
    "BOOTSTRAP_RESULTS.json",
    "final_disposition.json",
    "MECHANISM_REVERSE_ENGINEERING.json",
    "STAGE4_REPORT.md",
)


def _git(repository, *arguments):
    return subprocess.check_output(
        ["git", "-C", repository, *arguments],
        stderr=subprocess.STDOUT,
        text=True,
    ).strip()


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _canonical_array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
    digest.update(value.tobytes())
    return digest.hexdigest()


def _file_record(path, project_root):
    return {
        "relative_path": os.path.relpath(path, project_root),
        "bytes": int(os.path.getsize(path)),
        "sha256": sha256_file(path),
    }


def _historical_binding(project_root):
    head = _git(project_root, "rev-parse", "HEAD")
    tree = _git(project_root, "rev-parse", "HEAD^{tree}")
    if head != PREREGISTRATION_BASE_COMMIT or tree != PREREGISTRATION_BASE_TREE:
        raise RuntimeError(
            "BLOCKED_HISTORICAL_BINDING_MISMATCH:base:%s/%s" % (head, tree)
        )
    stage_trees = {}
    for stage, relative in STAGE_PATHS.items():
        observed = _git(project_root, "rev-parse", "HEAD:" + relative)
        expected = HISTORICAL_STAGE_TREES[stage]
        if observed != expected:
            raise RuntimeError(
                "BLOCKED_HISTORICAL_BINDING_MISMATCH:%s:%s:%s"
                % (stage, observed, expected)
            )
        published = (
            _git(project_root, "rev-parse", STAGE4_RESULT_COMMIT + ":" + relative)
            if stage == "stage4"
            else observed
        )
        if published != observed:
            raise RuntimeError(
                "BLOCKED_HISTORICAL_BINDING_MISMATCH:published-tree:" + stage
            )
        stage_trees[stage] = {
            "path": relative,
            "observed_git_tree_object": observed,
            "expected_git_tree_object": expected,
            "published_git_tree_object": published,
            "matched": True,
        }

    stage4_root = os.path.join(project_root, STAGE_PATHS["stage4"])
    files = {}
    for relative in STAGE4_BOUND_FILES:
        path = os.path.join(stage4_root, relative)
        if not os.path.isfile(path):
            raise RuntimeError(
                "BLOCKED_HISTORICAL_BINDING_MISMATCH:missing-stage4:" + relative
            )
        files[relative] = _file_record(path, project_root)
    checkpoint_files = {}
    checkpoint_root = os.path.join(stage4_root, "models")
    for directory, _, names in os.walk(checkpoint_root):
        for name in sorted(names):
            if not name.endswith(".pt"):
                continue
            path = os.path.join(directory, name)
            relative = os.path.relpath(path, stage4_root)
            checkpoint_files[relative] = _file_record(path, project_root)
    if not checkpoint_files:
        raise RuntimeError("BLOCKED_HISTORICAL_BINDING_MISMATCH:no-stage4-checkpoints")

    environment_lock = os.path.join(
        project_root, STAGE_PATHS["stage1"], "environment_lock.json"
    )
    stage4_binding = _load_json(os.path.join(stage4_root, "HISTORICAL_BINDING.json"))
    libero_source = stage4_binding["libero"]["source_path"]
    observed_libero_tree = sha256_tree(libero_source)
    expected_libero_tree = stage4_binding["libero"]["source_tree_sha256"]
    if observed_libero_tree != expected_libero_tree:
        raise RuntimeError("BLOCKED_HISTORICAL_BINDING_MISMATCH:libero-tree")

    action_bank = os.path.join(
        project_root, STAGE_PATHS["stage2"], "action_bank.npz"
    )
    scalers = os.path.join(stage4_root, "stage4_scalers.npz")
    with np.load(scalers, allow_pickle=False) as data:
        consequence_scale = np.asarray(data["consequence_scale"], dtype=np.float64)
        context_center = np.asarray(data["context_center"], dtype=np.float32)
        context_scale = np.asarray(data["context_scale"], dtype=np.float32)

    external_caches = {}
    for artifact in ("expanded_training_dataset.json", "fresh_confirmation_dataset.json"):
        payload = _load_json(os.path.join(stage4_root, artifact))
        for key in ("cache_path", "path"):
            candidate = payload.get(key)
            if not candidate or not os.path.isfile(candidate):
                continue
            expected = payload.get("cache_sha256") or payload.get("sha256")
            observed = sha256_file(candidate)
            if expected and observed != expected:
                raise RuntimeError(
                    "BLOCKED_HISTORICAL_BINDING_MISMATCH:external-cache:" + artifact
                )
            external_caches[artifact] = {
                "path": candidate,
                "bytes": int(os.path.getsize(candidate)),
                "sha256": observed,
                "expected_sha256": expected,
                "matched": expected in (None, observed),
            }
            break

    return {
        "binding_kind": "stage5_pre_result_read_only_historical_binding",
        "created_utc": utc_now(),
        "repository_input": {
            "commit": head,
            "tree": tree,
            "requested_branch": BRANCH,
            "stage4_result_commit": STAGE4_RESULT_COMMIT,
            "stage4_tree": STAGE4_TREE,
        },
        "historical_evidence": HISTORICAL_EVIDENCE,
        "historical_stage_trees": stage_trees,
        "environment_lock": _file_record(environment_lock, project_root),
        "libero": {
            "commit": config.UPSTREAM_LIBERO_COMMIT,
            "source_path": libero_source,
            "source_tree_sha256": observed_libero_tree,
            "expected_source_tree_sha256": expected_libero_tree,
            "matched": True,
        },
        "stage4_action_bank": _file_record(action_bank, project_root),
        "stage4_scalers": {
            **_file_record(scalers, project_root),
            "consequence_scale_array_sha256": _canonical_array_hash(consequence_scale),
            "context_center_array_sha256": _canonical_array_hash(context_center),
            "context_scale_array_sha256": _canonical_array_hash(context_scale),
            "consequence_refit_for_stage5": False,
            "context_refit_for_stage5": False,
        },
        "stage4_result_files": files,
        "stage4_checkpoint_files": checkpoint_files,
        "stage4_external_caches": external_caches,
        "historical_paths_modified": False,
        "all_hashes_match": True,
        "stage5_development_metric_inspected_before_freeze": False,
        "pai_jobs_submitted": 0,
    }


def _whitener(values):
    values = np.asarray(values, dtype=np.float64)
    centered = values - np.mean(values, axis=0, keepdims=True)
    covariance = centered.T.dot(centered) / max(len(centered) - 1, 1)
    covariance = covariance + LOCAL_BANK_COVARIANCE_REGULARIZATION * np.eye(
        covariance.shape[0]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    whitening = (eigenvectors / np.sqrt(eigenvalues)[None, :]).dot(eigenvectors.T)
    return whitening, eigenvalues


def _local_bank(project_root, output_root):
    source_path = os.path.join(project_root, STAGE_PATHS["stage2"], "action_bank.npz")
    support_path = os.path.join(
        project_root, STAGE_PATHS["stage4"], "training_support_bank.npz"
    )
    with np.load(source_path, allow_pickle=False) as data:
        source = {name: np.asarray(data[name]).copy() for name in data.files}
    with np.load(support_path, allow_pickle=False) as data:
        train_targets = np.asarray(data["residuals"], dtype=np.float64)
    residuals = np.asarray(source["residuals"], dtype=np.float64)
    if residuals.shape != (SOURCE_BANK_SIZE, 24):
        raise RuntimeError("source bank shape changed")
    whitening, eigenvalues = _whitener(np.concatenate((train_targets, residuals), axis=0))
    whitened_norm = np.linalg.norm(residuals.dot(whitening.T), axis=1)
    phases = source["source_phase"].astype(str)
    families = np.asarray(source["source_family_id"], dtype=np.int8)
    signs = np.asarray(source["source_sign"], dtype=np.int8)
    strata = [
        (phase, family, sign)
        for phase in PHASES
        for family in range(3)
        for sign in (-1, 1)
    ]
    base_quota, extra = divmod(LOCAL_BANK_SIZE, len(strata))
    selected = []
    skipped_target_equal = []
    skipped_near_duplicate = []
    target_hashes = {_array_hash(row) for row in train_targets}
    for stratum_index, (phase, family, sign) in enumerate(strata):
        quota = base_quota + int(stratum_index < extra)
        pool = np.flatnonzero(
            (phases == phase) & (families == family) & (signs == sign)
        )
        pool = sorted(pool.tolist(), key=lambda index: (whitened_norm[index], index))
        accepted = []
        for index in pool:
            if _array_hash(residuals[index]) in target_hashes:
                skipped_target_equal.append(int(index))
                continue
            if any(
                np.linalg.norm(residuals[index] - residuals[old])
                <= LOCAL_BANK_NEAR_DUPLICATE_TOLERANCE
                for old in selected + accepted
            ):
                skipped_near_duplicate.append(int(index))
                continue
            accepted.append(int(index))
            if len(accepted) == quota:
                break
        if len(accepted) != quota:
            raise RuntimeError(
                "local-bank undersupplied stratum %s/%d/%d" % (phase, family, sign)
            )
        selected.extend(accepted)
    selected = np.asarray(selected, dtype=np.int64)
    if len(selected) != LOCAL_BANK_SIZE or len(np.unique(selected)) != LOCAL_BANK_SIZE:
        raise RuntimeError("local bank is not unique M=128")
    output_path = os.path.join(output_root, "LOCAL_BANK.npz")
    arrays = {
        "residuals": residuals[selected],
        "source_indices": selected,
        "source_residual_sha256": source["residual_sha256"][selected],
        "source_task_id": source["source_task_id"][selected],
        "source_episode_id": source["source_episode_id"][selected],
        "source_phase": source["source_phase"][selected],
        "source_family_id": source["source_family_id"][selected],
        "source_direction_id": source["source_direction_id"][selected],
        "source_radius_id": source["source_radius_id"][selected],
        "source_radius": source["source_radius"][selected],
        "source_sign": source["source_sign"][selected],
        "whitened_zero_origin_norm": whitened_norm[selected],
        "train_covariance_whitener": whitening,
        "train_covariance_eigenvalues": eigenvalues,
        "selection_seed": np.asarray(LOCAL_BANK_SEED, dtype=np.int64),
    }
    atomic_npz(output_path, **arrays)
    max_component = float(np.max(np.abs(arrays["residuals"])))
    counts = Counter(
        (
            str(arrays["source_phase"][i]),
            int(arrays["source_family_id"][i]),
            int(arrays["source_sign"][i]),
        )
        for i in range(LOCAL_BANK_SIZE)
    )
    binding = {
        "kind": "stage5_deterministic_local_executable_bank",
        "source": _file_record(source_path, project_root),
        "training_support_source": _file_record(support_path, project_root),
        "selection_axes": list(LOCAL_BANK_BALANCE_AXES),
        "stratum_order": [list(value) for value in strata],
        "stratum_counts": {
            "%s/family_%d/sign_%+d" % key: int(value)
            for key, value in sorted(counts.items())
        },
        "selection_rule": (
            "quota=5 plus one for the first 8 phase/family/sign strata; "
            "ascending train-covariance-whitened zero-origin norm, then original ID"
        ),
        "covariance_fit_rows": int(len(train_targets) + len(residuals)),
        "covariance_fit_sources": ["Stage 4 train target residuals", "Stage 4 M=256 residual bank"],
        "covariance_regularization": LOCAL_BANK_COVARIANCE_REGULARIZATION,
        "source_size": SOURCE_BANK_SIZE,
        "local_size": LOCAL_BANK_SIZE,
        "source_indices": selected.tolist(),
        "source_indices_preserved": True,
        "near_duplicate_tolerance": LOCAL_BANK_NEAR_DUPLICATE_TOLERANCE,
        "skipped_near_duplicate_source_indices": skipped_near_duplicate,
        "skipped_target_equal_source_indices": skipped_target_equal,
        "target_residual_equality_count": 0,
        "maximum_absolute_residual_component": max_component,
        "analytic_executability_under_phase_selector": bool(
            0.895 + max_component <= 1.0 + 1e-12
        ),
        "minimum_required_valid_candidates": MINIMUM_VALID_CANDIDATES,
        "clipping_or_synthesis": False,
        "npz": _file_record(output_path, project_root),
        "residual_array_sha256": _canonical_array_hash(arrays["residuals"]),
        "source_index_array_sha256": _canonical_array_hash(selected),
        "selection_frozen_before_development": True,
    }
    atomic_json(os.path.join(output_root, "LOCAL_BANK_BINDING.json"), binding)
    return binding


def _fresh_target_bank(project_root, output_root):
    rng = np.random.RandomState(FRESH_TARGET_SEED)
    directions = []
    family_ids = []
    family_local_ids = []
    per_family = FRESH_TARGET_DIRECTION_COUNT // len(FRESH_TARGET_FAMILIES)
    for family_id, family in enumerate(FRESH_TARGET_FAMILIES):
        local_id = 0
        attempts = 0
        while local_id < per_family:
            attempts += 1
            if attempts > 100000:
                raise RuntimeError("cannot construct fresh target bank")
            try:
                direction = _candidate_direction(rng, family, local_id)
            except ValueError:
                continue
            if float(np.max(np.abs(direction))) > FRESH_TARGET_MAX_COMPONENT + 1e-12:
                continue
            if directions and float(
                np.max(np.abs(np.asarray(directions).dot(direction)))
            ) > 0.999999:
                continue
            directions.append(direction)
            family_ids.append(family_id)
            family_local_ids.append(local_id)
            local_id += 1
    residuals = []
    direction_ids = []
    residual_families = []
    radius_ids = []
    signs = []
    for direction_id, direction in enumerate(directions):
        for radius_id, radius in enumerate(FRESH_TARGET_RADII):
            for sign in FRESH_TARGET_SIGNS:
                residuals.append(float(sign) * float(radius) * direction)
                direction_ids.append(direction_id)
                residual_families.append(family_ids[direction_id])
                radius_ids.append(radius_id)
                signs.append(sign)
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.shape != (FRESH_TARGET_COUNT, 24):
        raise AssertionError(residuals.shape)
    historical_hashes = set()
    for path in (
        os.path.join(project_root, STAGE_PATHS["stage2"], "action_bank.npz"),
        os.path.join(project_root, STAGE_PATHS["stage4"], "training_support_bank.npz"),
        os.path.join(project_root, STAGE_PATHS["stage3"], "support_codebooks.npz"),
    ):
        with np.load(path, allow_pickle=False) as data:
            if "residuals" in data.files:
                rows = np.asarray(data["residuals"], dtype=np.float64).reshape(-1, 24)
                historical_hashes.update(_array_hash(row) for row in rows)
            if "directions" in data.files and "radii" in data.files and "signs" in data.files:
                direction = np.asarray(data["directions"], dtype=np.float64).reshape(-1, 24)
                radii = np.asarray(data["radii"], dtype=np.float64).reshape(-1, 2)
                old_signs = np.asarray(data["signs"], dtype=np.int8).reshape(-1)
                for row, row_radii in zip(direction, radii):
                    for radius in row_radii:
                        for sign in old_signs:
                            historical_hashes.add(_array_hash(float(sign) * float(radius) * row))
    overlap = [i for i, row in enumerate(residuals) if _array_hash(row) in historical_hashes]
    if overlap:
        raise RuntimeError("fresh target residual overlaps historical support")
    path = os.path.join(output_root, "FRESH_TARGET_BANK.npz")
    atomic_npz(
        path,
        directions=np.asarray(directions, dtype=np.float64),
        direction_family_id=np.asarray(family_ids, dtype=np.int8),
        direction_family_local_id=np.asarray(family_local_ids, dtype=np.int8),
        direction_family_names=np.asarray(FRESH_TARGET_FAMILIES),
        residuals=residuals,
        residual_direction_id=np.asarray(direction_ids, dtype=np.int8),
        residual_family_id=np.asarray(residual_families, dtype=np.int8),
        residual_radius_id=np.asarray(radius_ids, dtype=np.int8),
        residual_radius=np.asarray(
            [FRESH_TARGET_RADII[index] for index in radius_ids], dtype=np.float64
        ),
        residual_sign=np.asarray(signs, dtype=np.int8),
        seed=np.asarray(FRESH_TARGET_SEED, dtype=np.int64),
    )
    return {
        "path": os.path.relpath(path, project_root),
        "sha256": sha256_file(path),
        "residual_array_sha256": _canonical_array_hash(residuals),
        "count": len(residuals),
        "historical_exact_overlap_count": 0,
        "candidate_exact_overlap_count": 0,
        "independent_seed": FRESH_TARGET_SEED,
    }


def freeze(project_root):
    project_root = os.path.abspath(project_root)
    output_root = os.path.join(project_root, OUTPUT_RELATIVE)
    os.makedirs(output_root, exist_ok=True)
    binding = _historical_binding(project_root)
    atomic_json(os.path.join(output_root, "HISTORICAL_BINDING.json"), binding)
    local_bank = _local_bank(project_root, output_root)
    fresh_target = _fresh_target_bank(project_root, output_root)
    definitions = model_definitions()
    atomic_json(os.path.join(output_root, "MODEL_DEFINITIONS.json"), definitions)
    seeds = {
        "kind": "precommitted_nominal_generator_rollout_seeds",
        "selection": "ascending deterministic order; retain first 12 environment-success trajectories",
        "seeds_per_task": GENERATOR_ROLLOUT_SEEDS_PER_TASK,
        "required_successes_per_task": GENERATOR_REQUIRED_SUCCESSES_PER_TASK,
        "rollout_seeds": rollout_seeds(),
        "sacrificial_replay_seeds": {
            task: list(values) for task, values in GENERATOR_SACRIFICIAL_SEEDS.items()
        },
        "metric_independent_acceptance": "environment task success only",
    }
    atomic_json(os.path.join(output_root, "FRESH_TRAJECTORY_SEEDS.json"), seeds)
    generator = {
        "kind": "stage5_nominal_generator_pre_result_binding",
        "role": "nominal trajectory production only; not an R13-P15 method",
        "preferred_existing_checkpoint_audit": {
            "status": "NO_COMPATIBLE_FROZEN_CHECKPOINT_FOUND",
            "rejected_checkpoint": (
                "/mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16p19_libero_phase1/"
                "tiny_state_bc_v1/actor/step_000003000"
            ),
            "reason": "covers stove_moka and bowl_drawer, not the four frozen Stage 5 tasks",
        },
        "fallback_authorized": True,
        "architecture": dict(GENERATOR_ARCHITECTURE),
        "training_seed": GENERATOR_TRAIN_SEED,
        "training_episodes": list(GENERATOR_TRAIN_EPISODES),
        "rollout_seed_artifact": "FRESH_TRAJECTORY_SEEDS.json",
        "checkpoint_status": "PENDING_TRAINING_BEFORE_DEVELOPMENT",
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "may_not_change_after_development": True,
        "uses_stage5_consequence_labels": False,
        "uses_images_or_vla": False,
    }
    atomic_json(os.path.join(output_root, "NOMINAL_GENERATOR_BINDING.json"), generator)
    protocol = {
        "project_id": PROJECT_ID,
        "created_utc": utc_now(),
        "historical_binding": "HISTORICAL_BINDING.json",
        "tasks": list(TASK_IDS),
        "contact_sensitive_tasks": list(CONTACT_SENSITIVE_TASKS),
        "controller": {
            "robot": "Panda",
            "control_mode": "OSC_POSE",
            "frequency_hz": 20,
            "horizon": HORIZON,
            "settle_steps": 3,
        },
        "splits": {
            "training": list(TRAIN_EPISODES),
            "calibration": list(CALIBRATION_EPISODES),
            "development": list(DEVELOPMENT_EPISODES),
            "historical_exploratory": list(HISTORICAL_EXPLORATORY_EPISODES),
            "fresh_confirmation": "12 new successful policy trajectories per task",
        },
        "primary_metric": {
            "name": "BALANCED_TASK_EFFECT",
            "source": "byte-bound Stage 2-4 definition and train-only robust scales",
            "groups": [
                "object_pose",
                "tcp_object_relative_pose",
                "contact_mode_and_penetration",
                "gripper_and_articulation",
                "task_progress_and_constraints",
            ],
            "equal_group_weights": True,
            "capped_huber": True,
            "raw_force_excluded": True,
            "scales_refit": False,
        },
        "local_bank": {
            "artifact": "LOCAL_BANK.npz",
            "binding": "LOCAL_BANK_BINDING.json",
            "size": LOCAL_BANK_SIZE,
            "primary_k": PRIMARY_K,
            "minimum_valid_candidates": MINIMUM_VALID_CANDIDATES,
        },
        "fresh_target_bank": fresh_target,
        "branch_semantics": {
            "restore_identical_simulator_state": True,
            "execute_nominal_target_and_all_128_candidates": True,
            "copy_nominal_gripper_stepwise": True,
            "clipping": False,
            "action_synthesis": False,
            "evidence_label": "FRESH_POLICY_TRAJECTORY_CONFIRMATION",
        },
        "reversal": {
            "same_target_and_candidate_pair_across_states": True,
            "same_task": True,
            "prefer_same_current_contact": True,
            "margin_quantile": 0.25,
            "margin_fit": "training episodes 16-31 only",
            "margin_relaxation": False,
            "fabricated_labels": False,
            "episode_split_disjoint": True,
            "exact_tuple_overlap_allowed": False,
        },
        "bootstrap": {
            "cluster_unit": "source episode",
            "paired": True,
            "replicates": 10000,
        },
        "gates": GATES,
        "gate_failures_stop_later_registered_experiments": False,
        "reason_for_gate_continuation": "explicit user instruction",
        "final_dispositions": list(FINAL_DISPOSITIONS),
        "local_execution": {
            "cpu_simulation_preferred": True,
            "maximum_visible_training_gpus": 1,
            "pai_only_if_local_technically_impossible": True,
        },
    }
    atomic_json(os.path.join(output_root, "DATA_PROTOCOL.json"), protocol)
    return {
        "output_root": output_root,
        "historical_binding": "matched",
        "local_bank_sha256": local_bank["npz"]["sha256"],
        "fresh_target_bank_sha256": fresh_target["sha256"],
        "development_metrics_computed": False,
    }


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    args = parser.parse_args(argv)
    print(json.dumps(freeze(args.project_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
