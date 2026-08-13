#!/usr/bin/env python3
"""Verify the complete R13-P15 Stage 1.5 evidence package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np


REQUIRED_ARTIFACTS = (
    "PREREGISTRATION.md",
    "STAGE1_INPUT_BINDING.json",
    "retrospective_diagnostics.parquet",
    "error_decomposition.csv",
    "fresh_holdout_split.json",
    "fresh_branch_rollouts.zarr",
    "method_definitions.json",
    "quantization_results_by_task.csv",
    "quantization_results_by_phase.csv",
    "mechanism_controls.csv",
    "bootstrap_results.json",
    "STAGE1_5_REPORT.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(directory for directory in dirs if directory not in (".git", "__pycache__"))
        for name in sorted(files):
            if name.endswith((".pyc", ".pyo")):
                continue
            path = Path(current, name)
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode("utf-8") + b"\0")
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            digest.update(b"\0")
    return digest.hexdigest()


def strict_json(path: Path):
    def reject(value):
        raise ValueError("non-standard JSON constant %s" % value)

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def report_hashes(report_text: str) -> dict[str, str]:
    pattern = re.compile(r"^\| ([^|]+?) \| ([0-9a-f]{64}) \|$", re.MULTILINE)
    return {name.strip(): digest for name, digest in pattern.findall(report_text)}


def validate_marker(payload: Path) -> str | None:
    marker = Path(str(payload) + ".complete.json")
    if not marker.is_file():
        return "missing marker"
    metadata = strict_json(marker)
    if metadata.get("complete") is not True:
        return "marker not complete"
    if metadata.get("payload_bytes") != payload.stat().st_size:
        return "payload size mismatch"
    if metadata.get("payload_sha256") != sha256_file(payload):
        return "payload SHA-256 mismatch"
    return None


def csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--full-stage1-hash", action="store_true")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    stage1 = repo / "experiments/r13_p15_caaa_v2/stage1"
    stage1_5 = repo / "experiments/r13_p15_caaa_v2/stage1_5"
    report = stage1_5 / "STAGE1_5_REPORT.md"
    report_text = report.read_text(encoding="utf-8")
    expected_hashes = report_hashes(report_text)
    binding = strict_json(stage1_5 / "STAGE1_INPUT_BINDING.json")
    checks: dict[str, object] = {}
    failures: list[str] = []

    for name in REQUIRED_ARTIFACTS:
        path = stage1_5 / name
        exists = path.is_dir() if name.endswith(".zarr") else path.is_file()
        checks["required:%s" % name] = {"exists": exists, "ok": exists}
        if not exists:
            failures.append("missing required artifact: %s" % name)

    for name in REQUIRED_ARTIFACTS[:-1]:
        path = stage1_5 / name
        observed = sha256_tree(path) if path.is_dir() else sha256_file(path)
        expected = expected_hashes.get(name)
        ok = observed == expected
        checks["artifact_hash:%s" % name] = {
            "expected": expected,
            "observed": observed,
            "ok": ok,
        }
        if not ok:
            failures.append("artifact hash mismatch: %s" % name)

    json_failures = []
    for path in sorted(stage1_5.rglob("*.json")):
        try:
            strict_json(path)
        except Exception as error:
            json_failures.append("%s: %s" % (path.relative_to(repo), error))
    checks["strict_json"] = {
        "files": len(list(stage1_5.rglob("*.json"))),
        "failures": json_failures,
        "ok": not json_failures,
    }
    if json_failures:
        failures.append("non-strict JSON artifacts")

    input_commit = binding["repository"]["input_commit"]
    path_diff = git(repo, "diff", "--quiet", input_commit, "--", str(stage1.relative_to(repo)))
    stage1_git_ok = path_diff.returncode == 0
    checks["stage1_git_path_identity"] = {
        "input_commit": input_commit,
        "returncode": path_diff.returncode,
        "stderr": path_diff.stderr.strip(),
        "ok": stage1_git_ok,
    }
    if not stage1_git_ok:
        failures.append("Stage 1 Git path differs from bound input commit")

    bound_stage1_files = {
        "environment_lock": (stage1 / "environment_lock.json", binding["simulator"]["environment_lock_sha256"]),
        "report": (stage1 / "STAGE1_REPORT.md", binding["stage1"]["report_sha256"]),
        "jacobian_metrics": (
            stage1 / "jacobian_metrics.parquet",
            binding["stage1"]["artifacts"]["jacobian_metrics_parquet_sha256"],
        ),
        "quantization_results": (
            stage1 / "work/quantization_results.jsonl",
            binding["stage1"]["artifacts"]["quantization_results_jsonl_sha256"],
        ),
    }
    for label, (path, expected) in bound_stage1_files.items():
        observed = sha256_file(path)
        ok = observed == expected
        checks["stage1_bound_file:%s" % label] = {
            "expected": expected,
            "observed": observed,
            "ok": ok,
        }
        if not ok:
            failures.append("Stage 1 bound file mismatch: %s" % label)

    bound_stage1_trees = {
        "branch_rollouts": (
            stage1 / "branch_rollouts.zarr",
            binding["stage1"]["artifacts"]["branch_rollouts_zarr_tree_sha256"],
        ),
        "codebooks": (
            stage1 / "alphabet_codebooks",
            binding["stage1"]["artifacts"]["alphabet_codebooks_tree_sha256"],
        ),
        "quantized_shards": (
            stage1 / "work/quantized_shards",
            binding["stage1"]["artifacts"]["quantized_shards_tree_sha256"],
        ),
    }
    for label, (path, expected) in bound_stage1_trees.items():
        observed = sha256_tree(path)
        ok = observed == expected
        checks["stage1_bound_tree:%s" % label] = {
            "expected": expected,
            "observed": observed,
            "ok": ok,
        }
        if not ok:
            failures.append("Stage 1 bound tree mismatch: %s" % label)

    if args.full_stage1_hash:
        observed = sha256_tree(stage1)
        expected = binding["stage1"]["directory_tree_sha256"]
        ok = observed == expected
        checks["stage1_full_tree"] = {"expected": expected, "observed": observed, "ok": ok}
        if not ok:
            failures.append("complete Stage 1 tree mismatch")
    else:
        checks["stage1_full_tree"] = {
            "expected": binding["stage1"]["directory_tree_sha256"],
            "observed": None,
            "ok": None,
            "skipped": True,
        }

    prereg_commit = "9a3ac1a4c774103fe618bd283909c2793ed581ec"
    prereg_paths = git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", prereg_commit)
    wanted_paths = {
        "experiments/r13_p15_caaa_v2/stage1_5/PREREGISTRATION.md",
        "experiments/r13_p15_caaa_v2/stage1_5/STAGE1_INPUT_BINDING.json",
    }
    observed_paths = set(prereg_paths.stdout.splitlines())
    prereg_ok = prereg_paths.returncode == 0 and observed_paths == wanted_paths
    checks["preregistration_first_commit"] = {
        "commit": prereg_commit,
        "expected_paths": sorted(wanted_paths),
        "observed_paths": sorted(observed_paths),
        "ok": prereg_ok,
    }
    if not prereg_ok:
        failures.append("preregistration commit path contract mismatch")

    import pandas as pd

    diagnostics = pd.read_parquet(stage1_5 / "retrospective_diagnostics.parquet")
    diagnostic_ok = len(diagnostics) == 256 and {
        "local_r2",
        "local_normalized_rmse",
        "antithetic_nonlinearity_mean",
        "radius_derivative_drift_mean",
        "condition_number",
        "pseudoinverse_operator_norm",
        "realized_effect_error",
    }.issubset(diagnostics.columns)
    checks["retrospective_diagnostics"] = {"rows": len(diagnostics), "ok": diagnostic_ok}
    if not diagnostic_ok:
        failures.append("retrospective diagnostics contract mismatch")

    expected_csv_rows = {
        "error_decomposition.csv": 59,
        "quantization_results_by_task.csv": 40,
        "quantization_results_by_phase.csv": 160,
        "mechanism_controls.csv": 45,
    }
    for name, expected in expected_csv_rows.items():
        observed = csv_rows(stage1_5 / name)
        ok = observed == expected
        checks["csv_rows:%s" % name] = {"expected": expected, "observed": observed, "ok": ok}
        if not ok:
            failures.append("CSV row mismatch: %s" % name)

    plans = sorted((stage1_5 / "work/old_test_plans").rglob("*.npz"))
    realized = sorted((stage1_5 / "work/old_test_quantized_shards").rglob("*.npz"))
    shard_failures = []
    row_count = 0
    for plan, shard in zip(plans, realized):
        if plan.name != shard.name:
            shard_failures.append("plan/shard basename mismatch: %s %s" % (plan, shard))
            continue
        for payload in (plan, shard):
            error = validate_marker(payload)
            if error:
                shard_failures.append("%s: %s" % (payload, error))
        with np.load(plan, allow_pickle=False) as planned, np.load(shard, allow_pickle=False) as observed:
            row_count += len(observed["methods"])
            if len(observed["methods"]) != 288 or not np.all(observed["k"] == 64):
                shard_failures.append("row/K mismatch: %s" % shard)
            for field in (
                "methods",
                "k",
                "candidate_row",
                "direction",
                "sign",
                "radius",
                "code_index",
                "clipped_coordinates",
                "decoded_actions",
                "original_actions",
                "original_immediate",
                "original_settled",
                "original_mask",
                "original_contact_mode",
                "original_settled_success",
                "original_immediate_progress",
                "original_settled_progress",
            ):
                left, right = np.asarray(planned[field]), np.asarray(observed[field])
                equal = (
                    np.array_equal(left, right, equal_nan=True)
                    if left.dtype.kind in "fc" and right.dtype.kind in "fc"
                    else np.array_equal(left, right)
                )
                if not equal:
                    shard_failures.append("plan binding mismatch: %s %s" % (shard, field))
            for field in ("initial", "immediate", "settled", "final_state"):
                if not np.all(np.isfinite(observed[field])):
                    shard_failures.append("nonfinite realized data: %s %s" % (shard, field))
    shard_ok = len(plans) == 64 and len(realized) == 64 and row_count == 18432 and not shard_failures
    checks["old_test_realized_shards"] = {
        "plans": len(plans),
        "realized_shards": len(realized),
        "rows": row_count,
        "failures": shard_failures,
        "ok": shard_ok,
    }
    if not shard_ok:
        failures.append("old-test realized shard validation failed")

    method_definitions = strict_json(stage1_5 / "method_definitions.json")
    k_ok = method_definitions.get("primary_k") == 64 and method_definitions.get("k32_or_k128_inspected") is False
    checks["primary_k_only"] = {
        "primary_k": method_definitions.get("primary_k"),
        "sensitivity_inspected": method_definitions.get("k32_or_k128_inspected"),
        "ok": k_ok,
    }
    if not k_ok:
        failures.append("K=64-only contract mismatch")

    screen = strict_json(stage1_5 / "work/internal_screen.json")
    screen_ok = (
        screen.get("decision") == "REJECT_P15_FAMILY"
        and screen.get("passing_methods") == []
        and screen.get("row_count") == 30720
    )
    checks["internal_screen"] = {
        "decision": screen.get("decision"),
        "passing_methods": screen.get("passing_methods"),
        "rows": screen.get("row_count"),
        "ok": screen_ok,
    }
    if not screen_ok:
        failures.append("internal-screen stopping decision mismatch")

    bootstrap = strict_json(stage1_5 / "bootstrap_results.json")
    bootstrap_ok = (
        bootstrap.get("replicates") == 10000
        and bootstrap.get("evidence_status") == "retrospective_internal_screen_not_confirmatory"
        and len(bootstrap.get("comparisons", {})) == 9
    )
    checks["bootstrap"] = {
        "replicates": bootstrap.get("replicates"),
        "comparisons": len(bootstrap.get("comparisons", {})),
        "evidence_status": bootstrap.get("evidence_status"),
        "ok": bootstrap_ok,
    }
    if not bootstrap_ok:
        failures.append("bootstrap contract mismatch")

    fresh = strict_json(stage1_5 / "fresh_holdout_split.json")
    zattrs = strict_json(stage1_5 / "fresh_branch_rollouts.zarr/.zattrs")
    zarr_files = sorted(path.name for path in (stage1_5 / "fresh_branch_rollouts.zarr").iterdir())
    fresh_ok = (
        fresh.get("status") == "NOT_COLLECTED_INTERNAL_SCREEN_FAILED"
        and fresh.get("records") == []
        and zattrs.get("status") == "NOT_COLLECTED_INTERNAL_SCREEN_FAILED"
        and zattrs.get("states") == 0
        and zarr_files == [".zattrs", ".zgroup"]
    )
    checks["fresh_holdout_stop"] = {
        "split_status": fresh.get("status"),
        "records": len(fresh.get("records", [])),
        "zarr_status": zattrs.get("status"),
        "zarr_states": zattrs.get("states"),
        "zarr_files": zarr_files,
        "ok": fresh_ok,
    }
    if not fresh_ok:
        failures.append("fresh-holdout stopping manifest mismatch")

    finalize = strict_json(stage1_5 / "work/finalize_manifest.json")
    report_digest = sha256_file(report)
    disposition_ok = (
        report_text.rstrip().endswith("FINAL_DISPOSITION: REJECT_P15_FAMILY")
        and (stage1_5 / "work/FINAL_DISPOSITION.txt").read_text(encoding="utf-8").strip()
        == "REJECT_P15_FAMILY"
        and finalize.get("status") == "STAGE1_5_COMPLETE"
        and finalize.get("disposition") == "REJECT_P15_FAMILY"
        and finalize.get("report_sha256") == report_digest
    )
    checks["final_disposition"] = {
        "expected": "REJECT_P15_FAMILY",
        "report_sha256": report_digest,
        "manifest_report_sha256": finalize.get("report_sha256"),
        "ok": disposition_ok,
    }
    if not disposition_ok:
        failures.append("final disposition/report mismatch")

    result = {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "repo_root": ".",
        "full_stage1_hash_requested": bool(args.full_stage1_hash),
        "checks": checks,
        "failures": failures,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        destination = args.output if args.output.is_absolute() else repo / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(str(destination) + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(destination)
    print(rendered, end="")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
