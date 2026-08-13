"""Ordinary JSON release checks for the completed Stage 2 audit."""

from __future__ import annotations

import csv
import glob
import json
import os
import subprocess

import zarr

from .pipeline import utc_now
from .stage2 import LIBERO_TREE_SHA256, STAGE1_TREE_SHA256
from .storage import atomic_json, sha256_file, sha256_tree, validate_complete


REQUIRED_ARTIFACTS = (
    "PREREGISTRATION.md",
    "INPUT_BINDING.json",
    "fresh_episode_inventory.json",
    "development_split.json",
    "confirmation_split.json",
    "perturbation_banks.npz",
    "action_bank.npz",
    "consequence_metrics.json",
    "development_rollouts.zarr",
    "predictor_metrics.csv",
    "development_quantization.csv",
    "development_controls.csv",
    "confirmation_rollouts.zarr",
    "confirmation_quantization.csv",
    "bootstrap_results.json",
    "STAGE2_REPORT.md",
)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def verify_stage2(project_root, output_root):
    checks = []

    def check(name, passed, observed=None, expected=None):
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
            }
        )

    for name in REQUIRED_ARTIFACTS:
        check("required_artifact:%s" % name, os.path.exists(os.path.join(output_root, name)))

    binding = _read_json(os.path.join(output_root, "INPUT_BINDING.json"))
    stage1_root = os.path.join(project_root, "experiments", "r13_p15_caaa_v2", "stage1")
    stage1_5_root = os.path.join(project_root, "experiments", "r13_p15_caaa_v2", "stage1_5")
    observed_stage1 = sha256_tree(stage1_root)
    check(
        "stage1_tree_unchanged",
        observed_stage1 == STAGE1_TREE_SHA256,
        observed_stage1,
        STAGE1_TREE_SHA256,
    )
    expected_stage1_5 = binding["historical_evidence"]["stage1_5"]["observed_full_tree_sha256"]
    observed_stage1_5 = sha256_tree(stage1_5_root)
    check("stage1_5_tree_unchanged", observed_stage1_5 == expected_stage1_5, observed_stage1_5, expected_stage1_5)
    observed_libero = sha256_tree(binding["simulator"]["libero_source_path"])
    check("libero_tree_unchanged", observed_libero == LIBERO_TREE_SHA256, observed_libero, LIBERO_TREE_SHA256)
    identity = (
        subprocess.call(
            [
                "git",
                "-C",
                project_root,
                "diff",
                "--quiet",
                binding["repository_input"]["commit"],
                "--",
                "experiments/r13_p15_caaa_v2/stage1",
                "experiments/r13_p15_caaa_v2/stage1_5",
            ]
        )
        == 0
    )
    check("historical_git_paths_identical_to_input_commit", identity)

    freeze = _read_json(os.path.join(output_root, "work", "freeze_validation.json"))
    check("freeze_validation_passed", freeze["passed"])
    check("snapshot_count", freeze["snapshot_count"] == 384, freeze["snapshot_count"], 384)
    check("replay_failure_count", freeze["replay_failure_count"] == 0, freeze["replay_failure_count"], 0)
    check(
        "no_exact_cross_split_directions",
        not any(freeze["support_overlap"]["exact_direction_overlap"].values()),
    )
    check(
        "no_exact_cross_split_residuals",
        not any(freeze["support_overlap"]["exact_residual_overlap"].values()),
    )
    bank = freeze["action_bank_validity"]
    check("minimum_valid_bank", bank["minimum_valid_bank_size"] >= 128, bank["minimum_valid_bank_size"], ">=128")
    check("no_target_bank_exact_match", not bank["target_residual_exact_action_bank_matches"])

    support_shards = sorted(glob.glob(os.path.join(output_root, "work", "support_shards", "*", "*", "*.npz")))
    candidate_shards = sorted(glob.glob(os.path.join(output_root, "work", "candidate_shards", "*", "*", "*.npz")))
    check("support_shard_count", len(support_shards) == 256, len(support_shards), 256)
    check("candidate_shard_count", len(candidate_shards) == 128, len(candidate_shards), 128)
    incomplete = []
    for path in support_shards + candidate_shards:
        valid, evidence = validate_complete(path)
        if not valid:
            incomplete.append({"path": path, "evidence": evidence})
    check("all_branch_shards_complete", not incomplete, incomplete, [])
    confirmation_shards = glob.glob(
        os.path.join(output_root, "work", "support_shards", "confirmation", "**", "*.npz"), recursive=True
    ) + glob.glob(
        os.path.join(output_root, "work", "candidate_shards", "confirmation", "**", "*.npz"), recursive=True
    )
    check("no_confirmation_branch_shards", not confirmation_shards, confirmation_shards, [])

    development_zarr = zarr.open_group(os.path.join(output_root, "development_rollouts.zarr"), mode="r")
    for name, expected in (("support_states", 256), ("candidate_states", 128), ("branches", 57728)):
        check("development_zarr:%s" % name, int(development_zarr.attrs[name]) == expected, int(development_zarr.attrs[name]), expected)

    gate = _read_json(os.path.join(output_root, "work", "development_gate.json"))
    disposition = gate["development_disposition"]
    check("gate_A_passed", gate["gate_A"]["passed"])
    check("gate_B_failed", gate["gate_B"]["passed"] is False)
    check("gate_C_not_run", gate["gate_C"]["status"] == "NOT_RUN_GATE_B_FAILED")
    check("confirmation_locked", gate["confirmation_unlocked"] is False)
    check(
        "exact_disposition",
        disposition == "ORACLE_ONLY_NO_DEPLOYABLE_MODEL",
        disposition,
        "ORACLE_ONLY_NO_DEPLOYABLE_MODEL",
    )
    confirmation_zarr = zarr.open_group(os.path.join(output_root, "confirmation_rollouts.zarr"), mode="r")
    check("confirmation_observations_zero", int(confirmation_zarr.attrs["observations"]) == 0)
    check("confirmation_results_not_accessed", confirmation_zarr.attrs["confirmation_results_accessed"] is False)
    check("confirmation_csv_rows_zero", _csv_rows(os.path.join(output_root, "confirmation_quantization.csv")) == 0)
    bootstrap = _read_json(os.path.join(output_root, "bootstrap_results.json"))
    check("confirmation_bootstrap_not_run", bootstrap["replicates_executed"] == 0)
    check("development_quantization_rows", _csv_rows(os.path.join(output_root, "development_quantization.csv")) == 98304)
    check("predictor_summary_rows", _csv_rows(os.path.join(output_root, "predictor_metrics.csv")) == 126)
    check("pai_jobs_zero", binding["hard_scope"]["pai_jobs_submitted"] == 0)
    check("policy_training_false", binding["hard_scope"]["policy_training"] is False)

    for test_name in ("pre_result_test_results.json", "post_result_test_results.json"):
        tests = _read_json(os.path.join(output_root, "work", test_name))
        passed = all(value.get("passed") for value in tests.values() if isinstance(value, dict) and "passed" in value)
        check("tests:%s" % test_name, passed)

    manifest = {}
    for name in REQUIRED_ARTIFACTS:
        path = os.path.join(output_root, name)
        manifest[name] = {
            "type": "directory" if os.path.isdir(path) else "file",
            "bytes": None if os.path.isdir(path) else int(os.path.getsize(path)),
            "sha256": sha256_tree(path) if os.path.isdir(path) else sha256_file(path),
        }
    atomic_json(
        os.path.join(output_root, "ARTIFACT_MANIFEST.json"),
        {
            "created_utc": utc_now(),
            "result_commit": "7e8b6bc00177a21e205442c0113ca2584cce084c",
            "disposition": disposition,
            "artifacts": manifest,
        },
    )
    failures = [row for row in checks if not row["passed"]]
    verification = {
        "created_utc": utc_now(),
        "result_commit": "7e8b6bc00177a21e205442c0113ca2584cce084c",
        "disposition": disposition,
        "checks": checks,
        "check_count": len(checks),
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
    }
    atomic_json(os.path.join(output_root, "STAGE2_RELEASE_VERIFICATION.json"), verification)
    if failures:
        raise RuntimeError("Stage 2 release verification failed: %s" % failures)
    return verification
