#!/usr/bin/env python3
"""Verify the complete published CAAA-v2 Stage 1 evidence package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


REQUIRED_ARTIFACTS = (
    "PREREGISTRATION.md",
    "environment_lock.json",
    "task_and_seed_split.json",
    "branch_replay_validation.json",
    "consequence_schema.json",
    "branch_rollouts.zarr",
    "jacobian_metrics.parquet",
    "alphabet_codebooks",
    "results_by_task.csv",
    "results_by_phase.csv",
    "bootstrap_results.json",
    "mechanism_controls.csv",
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
        dirs[:] = sorted(d for d in dirs if d not in (".git", "__pycache__"))
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


def report_hashes(report_text: str) -> dict[str, str]:
    hashes = {}
    pattern = re.compile(r"^\| ([^|]+?) \| ([0-9a-f]{64}) \|$", re.MULTILINE)
    for name, digest in pattern.findall(report_text):
        hashes[name.strip()] = digest
    return hashes


def validate_marker(payload: Path) -> str | None:
    marker = Path(str(payload) + ".complete.json")
    if not marker.is_file():
        return "missing marker"
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if metadata.get("complete") is not True:
        return "marker is not complete"
    if metadata.get("payload_bytes") != payload.stat().st_size:
        return "payload size mismatch"
    if metadata.get("payload_sha256") != sha256_file(payload):
        return "payload SHA-256 mismatch"
    return None


def csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    stage1 = repo / "experiments/r13_p15_caaa_v2/stage1"
    report = stage1 / "STAGE1_REPORT.md"
    report_text = report.read_text(encoding="utf-8")
    expected = report_hashes(report_text)
    checks: dict[str, object] = {}
    failures: list[str] = []

    for name in REQUIRED_ARTIFACTS:
        path = stage1 / name
        observed = sha256_tree(path) if path.is_dir() else sha256_file(path)
        wanted = expected.get(name)
        ok = wanted == observed
        checks[f"artifact:{name}"] = {
            "expected_sha256": wanted,
            "observed_sha256": observed,
            "ok": ok,
        }
        if not ok:
            failures.append(f"artifact hash mismatch: {name}")

    shard_groups = {
        "branch": (stage1 / "work/branch_shards", 256),
        "quantized": (stage1 / "work/quantized_shards", 128),
        "quantization_plans": (stage1 / "work/quantization_plans", 128),
        "jacobians": (stage1 / "work/jacobians", 256),
    }
    for label, (root, expected_count) in shard_groups.items():
        payloads = sorted(root.rglob("*.npz"))
        marker_failures = [
            f"{payload}: {error}"
            for payload in payloads
            if (error := validate_marker(payload)) is not None
        ]
        ok = len(payloads) == expected_count and not marker_failures
        checks[f"atomic_shards:{label}"] = {
            "expected": expected_count,
            "observed": len(payloads),
            "marker_failures": marker_failures,
            "ok": ok,
        }
        if not ok:
            failures.append(f"invalid {label} shards")

    replay = json.loads(
        (stage1 / "branch_replay_validation.json").read_text(encoding="utf-8")
    )
    replay_count = len(replay.get("tests", []))
    replay_failures = replay.get("failed_tests", [])
    replay_ok = replay_count == 256 and replay_failures == []
    checks["formal_replay"] = {
        "expected": 256,
        "observed": replay_count,
        "failed_tests": replay_failures,
        "ok": replay_ok,
    }
    if not replay_ok:
        failures.append("formal replay gate mismatch")

    expected_rows = {"results_by_task.csv": 108, "results_by_phase.csv": 432}
    for name, count in expected_rows.items():
        observed = csv_rows(stage1 / name)
        ok = observed == count
        checks[f"rows:{name}"] = {"expected": count, "observed": observed, "ok": ok}
        if not ok:
            failures.append(f"row count mismatch: {name}")

    expected_jsonl_rows = {
        "work/quantization_results.jsonl": 69120,
        "work/jacobian_metrics.jsonl": 2304,
    }
    for name, count in expected_jsonl_rows.items():
        with (stage1 / name).open("rb") as handle:
            observed = sum(1 for _ in handle)
        ok = observed == count
        checks[f"rows:{name}"] = {"expected": count, "observed": observed, "ok": ok}
        if not ok:
            failures.append(f"row count mismatch: {name}")

    codebook_payloads = sorted((stage1 / "alphabet_codebooks").glob("*.npz"))
    codebook_marker_failures = [
        f"{payload}: {error}"
        for payload in codebook_payloads
        if (error := validate_marker(payload)) is not None
    ]
    codebooks_ok = len(codebook_payloads) == 27 and not codebook_marker_failures
    checks["codebooks"] = {
        "expected": 27,
        "observed": len(codebook_payloads),
        "marker_failures": codebook_marker_failures,
        "ok": codebooks_ok,
    }
    if not codebooks_ok:
        failures.append("invalid alphabet codebooks")

    smoke = json.loads(
        (repo / "experiments/r13_p15_caaa_v2/stage1_local_smoke/LOCAL_SMOKE.json").read_text(
            encoding="utf-8"
        )
    )
    smoke_ok = smoke.get("passed") is True and smoke.get("replay_validation", {}).get("failed_tests") == []
    checks["local_smoke"] = {
        "passed": smoke.get("passed"),
        "failed_tests": smoke.get("replay_validation", {}).get("failed_tests"),
        "ok": smoke_ok,
    }
    if not smoke_ok:
        failures.append("local smoke evidence mismatch")

    evidence = (
        repo
        / "provenance/pai/artifact_evidence"
        / "r13p15-caaa-v2-stage1-20260812-f/STAGE1_COMPLETE.json"
    )
    complete = json.loads(evidence.read_text(encoding="utf-8"))
    report_digest = sha256_file(report)
    completion_ok = (
        complete.get("status") == "STAGE1_COMPLETE"
        and complete.get("report_sha256") == report_digest
        and complete.get("quantized_rows") == 69120
        and complete.get("disposition") == "REJECT_CORE_HYPOTHESIS"
    )
    checks["pai_completion"] = {**complete, "ok": completion_ok}
    if not completion_ok:
        failures.append("PAI completion evidence mismatch")

    pai_latest = repo / "provenance/pai/job_registry_runs"
    expected_pai_status = {
        "r13p15-caaa-v2-stage1-20260812-d": "Failed",
        "r13p15-caaa-v2-stage1-20260812-e": "Failed",
        "r13p15-caaa-v2-stage1-20260812-f": "Succeeded",
    }
    observed_pai_status = {}
    for run_id, wanted in expected_pai_status.items():
        record = json.loads((pai_latest / run_id / "getjob-latest.json").read_text(encoding="utf-8"))
        observed_pai_status[run_id] = record.get("Status")
    pai_status_ok = observed_pai_status == expected_pai_status
    checks["pai_attempt_statuses"] = {
        "expected": expected_pai_status,
        "observed": observed_pai_status,
        "ok": pai_status_ok,
    }
    if not pai_status_ok:
        failures.append("PAI attempt status mismatch")

    disposition_ok = (
        report_text.rstrip().endswith("FINAL_DISPOSITION: REJECT_CORE_HYPOTHESIS")
        and (stage1 / "work/FINAL_DISPOSITION.txt").read_text(encoding="utf-8").strip()
        == "REJECT_CORE_HYPOTHESIS"
    )
    checks["final_disposition"] = {
        "expected": "REJECT_CORE_HYPOTHESIS",
        "ok": disposition_ok,
    }
    if not disposition_ok:
        failures.append("final disposition mismatch")

    formal_commit = "34995e8e7c3069b22785ad04536f0d429e75c0fc"
    git_ok = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{formal_commit}^{{commit}}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    checks["formal_source_commit_present"] = {
        "commit": formal_commit,
        "ok": git_ok,
    }
    if not git_ok:
        failures.append("formal source commit is absent from Git history")

    result = {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "repo_root": ".",
        "checks": checks,
        "failures": failures,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(str(args.output) + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(rendered, end="")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
