"""Bind history/data and reproduce D1--D3 before Stage 6-A evaluation.

This module deliberately does not import the Stage 6-A evaluator.  It is the
pre-performance freeze boundary required by the preregistration.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import math
import os
import subprocess

import numpy as np

from .stage1_5 import CONSEQUENCE_GROUPS
from .stage3_analysis import _action_assign, _ensemble_embedding, _ensemble_pair_score
from .stage3_data import load_records, transformed_contexts, true_distance_matrix
from .stage3_metrics import argmin_stable, ranking_metrics, stable_fps
from .stage3_models import create_biencoder, create_pair_ranker
from .stage6a_config import (
    ATLAS_SEED,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FINAL_DISPOSITIONS,
    GATE_A,
    GATE_H,
    HISTORICAL_DISPOSITIONS,
    MIN_VALID_CANDIDATES,
    OUTPUT_RELATIVE,
    PRIMARY_K,
    STAGE1_SCRATCH_ROOT,
    STAGE3_RELATIVE,
    STAGE5_RELATIVE,
    STAGE5_SCRATCH_ROOT,
)
from .storage import atomic_json, sha256_file, validate_complete


RELATIVE_TOLERANCE = 1e-4


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _git(project_root, *args):
    return subprocess.check_output(
        ("git",) + tuple(args), cwd=project_root, text=True
    ).strip()


def _tree_hash(directory, excluded=()):
    rows = []
    excluded = set(excluded)
    for root, _, files in os.walk(directory):
        for name in sorted(files):
            path = os.path.join(root, name)
            relative = os.path.relpath(path, directory).replace(os.sep, "/")
            if relative in excluded:
                continue
            rows.append((relative, sha256_file(path), os.path.getsize(path)))
    digest = hashlib.sha256()
    for relative, value, size in rows:
        digest.update((relative + "\0" + value + "\0" + str(size) + "\n").encode())
    return digest.hexdigest(), rows


def _verify_stage5_hashes(stage5_root):
    verifier_path = os.path.join(stage5_root, "STAGE5_RELEASE_VERIFICATION.json")
    verifier = _load_json(verifier_path)
    mismatches = []
    for relative, expected in sorted(verifier["artifact_hashes"].items()):
        path = os.path.join(stage5_root, relative)
        if not os.path.isfile(path):
            mismatches.append({"path": relative, "reason": "missing"})
            continue
        actual = sha256_file(path)
        if actual != expected["sha256"] or os.path.getsize(path) != int(expected["bytes"]):
            mismatches.append(
                {
                    "path": relative,
                    "reason": "hash_or_size",
                    "expected": expected,
                    "actual": {"sha256": actual, "bytes": os.path.getsize(path)},
                }
            )
    return verifier, mismatches


def historical_binding(project_root, output_root):
    stage5_root = os.path.join(project_root, STAGE5_RELATIVE)
    published = _load_json(os.path.join(output_root, "PUBLISHED_ARTIFACT_VERIFICATION.json"))
    stage1_5 = _load_json(os.path.join(output_root, "STAGE1_5_ARTIFACT_VERIFICATION.json"))
    stage2 = _load_json(
        os.path.join(project_root, "experiments/r13_p15_ncea/stage2/STAGE2_RELEASE_VERIFICATION.json")
    )
    stage3 = _load_json(
        os.path.join(project_root, "experiments/r13_p15_ncer_aa/stage3/STAGE3_RELEASE_VERIFICATION.json")
    )
    stage4 = _load_json(
        os.path.join(project_root, "experiments/r13_p15_cr_trca/stage4/final_disposition.json")
    )
    stage5, stage5_mismatches = _verify_stage5_hashes(stage5_root)
    checks = {
        "stage1": published.get("status") == "PASS" and not published.get("failures"),
        "stage1_5": stage1_5.get("status") == "PASS" and not stage1_5.get("failures"),
        "stage2": bool(stage2.get("passed")),
        "stage3": bool(stage3.get("passed")),
        "stage4": stage4.get("final_disposition") == HISTORICAL_DISPOSITIONS["stage4"],
        "stage5": bool(stage5.get("passed")) and not stage5_mismatches,
        "historical_paths_immutable": bool(stage5.get("historical_paths_immutable")),
    }
    artifact_tree, rows = _tree_hash(
        stage5_root, excluded=("STAGE5_RELEASE_VERIFICATION.json",)
    )
    result = {
        "kind": "stage6a_historical_binding",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_commit": _git(project_root, "rev-parse", "HEAD"),
        "source_tree": _git(project_root, "rev-parse", "HEAD^{tree}"),
        "source_branch": _git(project_root, "branch", "--show-current"),
        "historical_dispositions_immutable": HISTORICAL_DISPOSITIONS,
        "checks": checks,
        "stage5_independent_hash_check": {
            "artifact_count": len(stage5.get("artifact_hashes", {})),
            "mismatches": stage5_mismatches,
            "artifact_tree_excluding_release_verifier": artifact_tree,
            "tree_file_count": len(rows),
        },
        "passed": bool(all(checks.values())),
        "failure_disposition": None
        if all(checks.values())
        else "BLOCKED_HISTORICAL_BINDING_MISMATCH",
    }
    atomic_json(os.path.join(output_root, "HISTORICAL_BINDING.json"), result)
    return result


def data_source_binding(project_root, output_root):
    manifest_path = os.path.join(project_root, STAGE5_RELATIVE, "STAGE5_CACHE_MANIFEST.json")
    manifest = _load_json(manifest_path)
    entries = []
    passed = True
    for split in ("train", "calibration", "development", "historical_exploratory"):
        source = next(row for row in manifest["caches"] if row["split"] == split)
        path = source["path"]
        valid, marker = validate_complete(path)
        with np.load(path, allow_pickle=False) as data:
            shape = list(data["true_distance"].shape)
            candidate_ids = np.asarray(data["candidate_source_index"], dtype=np.int64)
            schema = str(data["schema_version"].item())
            target_count = int(data["target_residual"].shape[0])
        actual = sha256_file(path)
        entry_passed = (
            valid
            and actual == source["sha256"]
            and shape[1] == 96
            and shape[2] >= MIN_VALID_CANDIDATES
            and len(np.unique(candidate_ids)) == shape[2]
        )
        passed = passed and entry_passed
        entries.append(
            {
                "split": split,
                "path": path,
                "schema": schema,
                "sha256": actual,
                "bytes": os.path.getsize(path),
                "complete_marker_valid": bool(valid),
                "marker_evidence": marker,
                "state_target_candidate_shape": shape,
                "target_count": target_count,
                "candidate_count": int(shape[2]),
                "candidate_source_id_min": int(candidate_ids.min()),
                "candidate_source_id_max": int(candidate_ids.max()),
                "candidate_source_ids_unique": bool(len(np.unique(candidate_ids)) == len(candidate_ids)),
                "complete_pair_coverage": bool(shape[1] == target_count and shape[2] == len(candidate_ids)),
                "passed": bool(entry_passed),
            }
        )
    result = {
        "kind": "stage6a_data_source_binding",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "chosen_source": "stage5_executed_candidate_consequence_caches",
        "selection_precedence": ["stage5_caches", "stage2_support_shards"],
        "manifest_path": os.path.relpath(manifest_path, project_root),
        "manifest_sha256": sha256_file(manifest_path),
        "index_space": {
            "state": "cache row; key/task_id/episode_id/phase identify source episode snapshot",
            "target": "0..95 frozen Stage 2 target residual",
            "candidate": "0..127 local position; candidate_source_index maps to frozen 256-bank ID",
            "lookup": "true_distance[state,target,candidate] from already-simulated settled consequence",
        },
        "entries": entries,
        "passed": bool(passed),
        "failure_disposition": None if passed else "BLOCKED_NO_EXECUTED_CANDIDATE_CACHE",
    }
    atomic_json(os.path.join(output_root, "DATA_SOURCE_BINDING.json"), result)
    return result


def _relative_match(value, expected):
    return abs(float(value) - float(expected)) <= RELATIVE_TOLERANCE * max(abs(float(expected)), 1e-12)


def reproduce_d1():
    parameter_path = os.path.join(STAGE1_SCRATCH_ROOT, "analysis_parameters.npz")
    with np.load(parameter_path, allow_pickle=False) as data:
        scale = np.asarray(data["consequence_scale"], dtype=np.float64)
    utilizations, clipped, group_values = [], [], {name: [] for name in CONSEQUENCE_GROUPS}
    paths = sorted(glob.glob(os.path.join(STAGE1_SCRATCH_ROOT, "quantized_shards", "*", "*.npz")))
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            keep = (data["methods"].astype(str) == "caaa_v2") & (data["k"] == PRIMARY_K)
            labels = np.asarray(data["code_index"][keep], dtype=np.int64)
            utilizations.append(len(np.unique(labels)) / float(PRIMARY_K))
            clipped.append(float(np.sum(data["clipped_coordinates"][keep])) / (len(labels) * 24.0))
            mask = np.asarray(data["original_mask"][keep], dtype=bool)
            delta = (
                np.asarray(data["settled"][keep], dtype=np.float64)
                - np.asarray(data["original_settled"][keep], dtype=np.float64)
            ) / scale[None, :]
            delta[~mask] = 0.0
            for name, indices in CONSEQUENCE_GROUPS.items():
                group_values[name].append(float(np.mean(np.sum(delta[:, list(indices)] ** 2, axis=1))))
    means = {name: float(np.mean(values)) for name, values in group_values.items()}
    total = sum(means.values())
    shares = {name: value / total for name, value in means.items()}
    return {
        "source_shards": len(paths),
        "parameter_sha256": sha256_file(parameter_path),
        "median_assignment_utilization": float(np.median(utilizations)),
        "median_realized_clipped_coordinate_fraction": float(np.median(clipped)),
        "group_mean_squared_normalized_error": means,
        "group_share_of_total": shares,
    }


def _load_model(path, model, device):
    import torch

    payload = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model


def _pair_score_matrix(models, context, targets, candidates, device, batch_size=8192):
    """Vectorized equivalent of the frozen Stage 3 C4 scoring loop."""
    import torch

    targets = np.asarray(targets, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    target_index = np.repeat(np.arange(len(targets), dtype=np.int64), len(candidates))
    candidate_index = np.tile(np.arange(len(candidates), dtype=np.int64), len(targets))
    by_model = []
    for model in models:
        chunks = []
        with torch.no_grad():
            for start in range(0, len(target_index), int(batch_size)):
                stop = min(start + int(batch_size), len(target_index))
                count = stop - start
                ctx = torch.as_tensor(
                    np.repeat(np.asarray(context, dtype=np.float32)[None, :], count, axis=0),
                    device=device,
                )
                tgt = torch.as_tensor(targets[target_index[start:stop]] / 0.12, device=device)
                cand = torch.as_tensor(candidates[candidate_index[start:stop]] / 0.12, device=device)
                chunks.append(model(ctx, tgt, cand).cpu().numpy())
        by_model.append(np.concatenate(chunks).reshape(len(targets), len(candidates)))
    return np.mean(np.stack(by_model), axis=0)


def reproduce_d2_checkpoint_crosscheck(project_root):
    import torch

    stage3_root = os.path.join(project_root, STAGE3_RELATIVE)
    registry = _load_json(os.path.join(stage3_root, "trained_model_registry.json"))
    with np.load(os.path.join(stage3_root, registry["scalers"]), allow_pickle=False) as data:
        scale = np.asarray(data["consequence_scale"], dtype=np.float64)
        center = np.asarray(data["context_center"], dtype=np.float64)
        context_scale = np.asarray(data["context_scale"], dtype=np.float64)
    device = torch.device("cpu")
    torch.set_num_threads(min(16, max(1, os.cpu_count() or 1)))
    models = {}
    checkpoint_hashes = {}
    for family, factory in (
        ("C3_NC_BIENCODER", create_biencoder),
        ("C4_NC_PAIR_RANKER", create_pair_ranker),
    ):
        rows = []
        for relative in registry["models"][family]["members"]:
            path = os.path.join(stage3_root, relative)
            rows.append(_load_model(path, factory(len(center)), device))
            checkpoint_hashes[relative] = sha256_file(path)
        models[family] = rows
    records = load_records(project_root, stage3_root, ("development",))
    contexts = transformed_contexts(records, center, context_scale)
    with np.load(os.path.join(stage3_root, "baseline_codebooks.npz"), allow_pickle=False) as data:
        codebooks = {name: np.asarray(data[name]).copy() for name in data.files}
    action_bank_path = os.path.join(project_root, "experiments/r13_p15_ncea/stage2/action_bank.npz")
    with np.load(action_bank_path, allow_pickle=False) as data:
        action_bank = np.asarray(data["residuals"], dtype=np.float64)
    per_method = {name: [] for name in ("B2", "C3", "C4", "C5")}
    realized = {name: [] for name in per_method}
    for state, record in enumerate(records):
        target = np.asarray(record["support"]["residual_action"][1:], dtype=np.float32)
        bank = np.asarray(record["candidate"]["residual_action"][1:], dtype=np.float32)
        truth = true_distance_matrix(record, scale)
        context = contexts[state]
        c3_bank = _ensemble_embedding(models["C3_NC_BIENCODER"], context, bank, device)
        c3_target = _ensemble_embedding(models["C3_NC_BIENCODER"], context, target, device)
        c3_atlas = stable_fps(c3_bank, PRIMARY_K)
        c3_score = np.sum((c3_target[:, None, :] - c3_bank[None, :, :]) ** 2, axis=2)
        c4_score = _pair_score_matrix(
            models["C4_NC_PAIR_RANKER"], context, target, bank, device
        )
        c3_selected = np.asarray([argmin_stable(row[c3_atlas], ids=c3_atlas) for row in c3_score])
        c4_selected = np.asarray([argmin_stable(row) for row in c4_score])
        c5_selected = np.asarray([argmin_stable(row[c3_atlas], ids=c3_atlas) for row in c4_score])
        current = str(int(bool(record["context"]["current_contact"].item())))
        b2_codes = codebooks["B2_contact_" + current]
        b2_selected = _action_assign(target, action_bank, b2_codes)
        raw_score = np.mean(((target[:, None, :] - bank[None, :, :]) / 0.12) ** 2, axis=2)
        b2_score = np.full_like(raw_score, np.max(raw_score) + max(float(np.ptp(raw_score)), 1.0) * 1000.0)
        b2_score[:, b2_codes] = raw_score[:, b2_codes]
        for method, selected, score in (
            ("B2", b2_selected, b2_score),
            ("C3", c3_selected, c3_score),
            ("C4", c4_selected, c4_score),
            ("C5", c5_selected, c4_score),
        ):
            realized[method].extend(truth[np.arange(len(target)), selected].tolist())
            per_method[method].extend(
                ranking_metrics(truth[row], score[row], int(selected[row]))
                for row in range(len(target))
            )
    output = {}
    for method in per_method:
        output[method] = {
            "oracle_regret": float(np.mean([row["oracle_regret"] for row in per_method[method]])),
            "candidate_distance_spearman": float(np.mean([row["candidate_distance_spearman"] for row in per_method[method]])),
            "ndcg_at_16": float(np.mean([row["ndcg_at_16"] for row in per_method[method]])),
            "realized_effect_error": float(np.mean(realized[method])),
        }
    return {
        "development_states": len(records),
        "rows_per_method": len(realized["C3"]),
        "checkpoint_hashes": checkpoint_hashes,
        "action_bank_sha256": sha256_file(action_bank_path),
        "metrics": output,
    }


def reproduce_d2(project_root):
    """Aggregate the frozen Stage 3 development result rows.

    Stage 3 did not persist per-query retrieval scores, so ranking statistics
    are bound to its published pooled development row.  Realized errors are
    independently averaged over all 6,144 development rows per method.  B2
    retrieval metrics are recomputed from raw executed shards because Stage 3
    did not publish a B2 retrieval summary.
    """
    stage3_root = os.path.join(project_root, STAGE3_RELATIVE)
    quantization_path = os.path.join(stage3_root, "development_quantization.csv")
    retrieval_path = os.path.join(stage3_root, "retrieval_metrics.csv")
    realized_values = {name: [] for name in ("B2", "C3", "C4", "C5")}
    name_map = {
        "B2_current_contact_kmeans": "B2",
        "C3_NC_BIENCODER": "C3",
        "C4_NC_PAIR_RANKER": "C4",
        "C5_NCER_AA": "C5",
    }
    with open(quantization_path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            method = name_map.get(row["method"])
            if method and row["split"] == "development":
                realized_values[method].append(float(row["balanced_task_effect_error"]))
    published_ranking = {}
    with open(retrieval_path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            method = name_map.get(row["method"])
            if (
                method in ("C3", "C4", "C5")
                and row["split"] == "development"
                and row["level"] == "pooled"
                and row["task_id"] == "ALL"
                and row["phase"] == "ALL"
                and row["direction_family_id"] == "ALL"
            ):
                published_ranking[method] = {
                    "oracle_regret": float(row["oracle_regret"]),
                    "candidate_distance_spearman": float(row["candidate_distance_spearman"]),
                    "ndcg_at_16": float(row["ndcg_at_16"]),
                }

    # B2 ranking was not emitted by Stage 3. Recompute it from the executed
    # true-distance matrix and the frozen current-contact action codebook.
    with np.load(os.path.join(stage3_root, "model_scalers.npz"), allow_pickle=False) as data:
        consequence_scale = np.asarray(data["consequence_scale"], dtype=np.float64)
    with np.load(os.path.join(stage3_root, "baseline_codebooks.npz"), allow_pickle=False) as data:
        codebooks = {name: np.asarray(data[name]).copy() for name in data.files}
    action_bank_path = os.path.join(project_root, "experiments/r13_p15_ncea/stage2/action_bank.npz")
    with np.load(action_bank_path, allow_pickle=False) as data:
        action_bank = np.asarray(data["residuals"], dtype=np.float64)
    records = load_records(project_root, stage3_root, ("development",))
    b2_rows = []
    for record in records:
        target = np.asarray(record["support"]["residual_action"][1:], dtype=np.float64)
        truth = true_distance_matrix(record, consequence_scale)
        current = str(int(bool(record["context"]["current_contact"].item())))
        codes = codebooks["B2_contact_" + current]
        selected = _action_assign(target, action_bank, codes)
        raw_score = np.mean(((target[:, None, :] - action_bank[None, :, :]) / 0.12) ** 2, axis=2)
        score = np.full_like(raw_score, np.max(raw_score) + max(float(np.ptp(raw_score)), 1.0) * 1000.0)
        score[:, codes] = raw_score[:, codes]
        b2_rows.extend(
            ranking_metrics(truth[row], score[row], int(selected[row]))
            for row in range(len(target))
        )
    published_ranking["B2"] = {
        "oracle_regret": float(np.mean([row["oracle_regret"] for row in b2_rows])),
        "candidate_distance_spearman": float(
            np.mean([row["candidate_distance_spearman"] for row in b2_rows])
        ),
        "ndcg_at_16": float(np.mean([row["ndcg_at_16"] for row in b2_rows])),
    }
    metrics = {}
    for method in ("B2", "C3", "C4", "C5"):
        if len(realized_values[method]) != 6144:
            raise RuntimeError("unexpected Stage 3 development row count for " + method)
        metrics[method] = dict(published_ranking[method])
        metrics[method]["realized_effect_error"] = float(np.mean(realized_values[method]))
    return {
        "development_states": 64,
        "rows_per_method": 6144,
        "ranking_source_semantics": (
            "C3/C4/C5: frozen pooled development retrieval row; "
            "B2: independently recomputed because no B2 retrieval row was published"
        ),
        "source_hashes": {
            "development_quantization.csv": sha256_file(quantization_path),
            "retrieval_metrics.csv": sha256_file(retrieval_path),
            "baseline_codebooks.npz": sha256_file(
                os.path.join(stage3_root, "baseline_codebooks.npz")
            ),
            "action_bank.npz": sha256_file(action_bank_path),
        },
        "metrics": metrics,
    }


def reproduce_d3(project_root):
    import torch

    stage5_root = os.path.join(project_root, STAGE5_RELATIVE)
    manifest = _load_json(os.path.join(stage5_root, "MODEL_TRAINING_MANIFEST.json"))
    entries = [
        row for row in manifest["entries"]
        if row["metadata"].get("method") == "P1_CONTEXT_GATED_PSD"
        and row["metadata"].get("control") == "PROPOSED"
    ]
    entries.sort(key=lambda row: int(row["metadata"]["seed"]))
    cache_path = os.path.join(STAGE5_SCRATCH_ROOT, "derived", "stage5_train_cache.npz")
    with np.load(cache_path, allow_pickle=False) as data:
        context = np.asarray(data["context"], dtype=np.float32)
    from .stage5_models import load_context_checkpoint

    device = torch.device("cpu")
    norms = []
    hashes = {}
    with torch.no_grad():
        tensor = torch.as_tensor(context, device=device)
        for entry in entries:
            path = os.path.join(stage5_root, entry["path"])
            model, _ = load_context_checkpoint(path, device)
            norms.append(float(torch.linalg.vector_norm(model.modulation(tensor), dim=1).mean().cpu()))
            hashes[entry["path"]] = sha256_file(path)
    bound = math.sqrt(24.0) * 1.25
    mean_norm = float(np.mean(norms))
    return {
        "checkpoint_hashes": hashes,
        "train_cache_sha256": sha256_file(cache_path),
        "per_seed_mean_modulation_norm": norms,
        "mean_modulation_norm": mean_norm,
        "theoretical_bound": bound,
        "fraction_of_bound": mean_norm / bound,
    }


def defect_reproduction(project_root, output_root):
    d1 = reproduce_d1()
    d2 = reproduce_d2(project_root)
    d3 = reproduce_d3(project_root)
    checks = {
        "d1_assignment_utilization": _relative_match(d1["median_assignment_utilization"], 0.015625),
        "d1_clipped_fraction": _relative_match(d1["median_realized_clipped_coordinate_fraction"], 0.834201),
        "d1_contact_and_force_share": _relative_match(d1["group_share_of_total"]["contact_and_force"], 0.999953),
        "d2_c3_regret": _relative_match(d2["metrics"]["C3"]["oracle_regret"], 0.24297),
        "d2_c4_regret": _relative_match(d2["metrics"]["C4"]["oracle_regret"], 0.31872),
        "d2_c3_ndcg": _relative_match(d2["metrics"]["C3"]["ndcg_at_16"], 0.61958),
        "d2_c4_ndcg": _relative_match(d2["metrics"]["C4"]["ndcg_at_16"], 0.4449),
        "d2_b2_realized": _relative_match(d2["metrics"]["B2"]["realized_effect_error"], 0.30817),
        "d2_c3_realized": _relative_match(d2["metrics"]["C3"]["realized_effect_error"], 0.28315),
        "d2_c4_realized": _relative_match(d2["metrics"]["C4"]["realized_effect_error"], 0.3589),
        "d2_c5_realized": _relative_match(d2["metrics"]["C5"]["realized_effect_error"], 0.37497),
        "d3_mean_norm": _relative_match(d3["mean_modulation_norm"], 5.903557),
        "d3_bound": _relative_match(d3["theoretical_bound"], 6.123724),
    }
    result = {
        "kind": "stage6a_defect_reproduction",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "relative_tolerance": RELATIVE_TOLERANCE,
        "D1_quantizer_degeneracy": d1,
        "D2_pipeline_coverage": d2,
        "D3_coverage_evidence_only": d3,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "failure_disposition": None if all(checks.values()) else "BLOCKED_DEFECT_NOT_REPRODUCED",
    }
    atomic_json(os.path.join(output_root, "DEFECT_REPRODUCTION.json"), result)
    return result


def repaired_definition(project_root, output_root):
    stage3_root = os.path.join(project_root, STAGE3_RELATIVE)
    stage5_root = os.path.join(project_root, STAGE5_RELATIVE)
    registry = _load_json(os.path.join(stage3_root, "trained_model_registry.json"))
    c3 = []
    for relative in registry["models"]["C3_NC_BIENCODER"]["members"]:
        path = os.path.join(stage3_root, relative)
        c3.append({"path": os.path.relpath(path, project_root), "sha256": sha256_file(path)})
    manifest = _load_json(os.path.join(stage5_root, "MODEL_TRAINING_MANIFEST.json"))
    stage5_checkpoints = []
    for row in manifest["entries"]:
        metadata = row["metadata"]
        if metadata.get("method") in ("B1_ACTION_ONLY", "B2_STATIC_CONSEQUENCE"):
            path = os.path.join(stage5_root, row["path"])
            stage5_checkpoints.append(
                {"method": metadata["method"], "path": os.path.relpath(path, project_root), "sha256": sha256_file(path)}
            )
    result = {
        "kind": "stage6a_repaired_definition",
        "frozen_before_stage6a_performance": True,
        "metric": {
            "name": "BALANCED_TASK_EFFECT",
            "source": "frozen Stage 2-5",
            "groups": [
                "object_pose",
                "tcp_object_relative_pose",
                "contact_mode_and_penetration",
                "gripper_and_articulation",
                "task_progress_and_constraint",
            ],
            "equal_group_weights": True,
            "train_only_robust_scales": True,
            "capped_huber": True,
            "raw_force_excluded": True,
            "refit_or_reweight": False,
        },
        "candidate_set": {
            "source": "DATA_SOURCE_BINDING.json",
            "selection_returns_executed_bank_index": True,
            "pseudo_inverse_decode": False,
            "coordinate_clipping": False,
            "action_synthesis": False,
            "target_residual_equality_required": False,
        },
        "alphabet": {
            "k": PRIMARY_K,
            "algorithm": "deterministic_id_stable_kmedoids",
            "space": "frozen_C3_biencoder_embedding",
            "seed": ATLAS_SEED,
            "tie_break": "lowest_frozen_candidate_source_id",
            "forbidden_k_before_disposition": [32, 96, 128],
        },
        "selection": {
            "intra_atlas_score": "C3_biencoder_squared_euclidean_distance_only",
            "c4_pair_ranker_in_call_graph": False,
            "listwise_reranker": False,
            "soft_mixture": False,
            "c5_or_c6_path": False,
        },
        "checkpoints": {"C3": c3, "Stage5_B1_B2": stage5_checkpoints},
        "gate_h": GATE_H,
        "gate_a": GATE_A,
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "cluster": "source_episode"},
        "final_dispositions": list(FINAL_DISPOSITIONS),
        "scope": {"training": False, "simulation": False, "pai_submission": False, "stage6b": False},
    }
    atomic_json(os.path.join(output_root, "REPAIRED_DEFINITION.json"), result)
    return result


PREREGISTRATION = """# R13-P15 Stage 6-A preregistration

## Scientific boundary

This is a defect-repair replay over already-executed LIBERO candidate consequences. It does not
train a model, run a simulator, submit PAI work, revise any historical disposition, or begin
Stage 6-B. The sole question is how much Stage 5 Gate 0 K=64 adaptive headroom is recovered after
repairing the quantizer and deleting C4 from the proposed selection call graph.

## Frozen inputs

- Tasks: `bowl_on_plate`, `plate_push`, `stove_turn_on`, `wine_rack`.
- Controller: Panda `OSC_POSE`, 20 Hz, `H=4`, three settle steps.
- Data: the hash-verified Stage 5 M=128 executed-consequence caches, 96 targets per state.
- Metric: frozen `BALANCED_TASK_EFFECT`; equal group weights, train-only robust scales, capped
  Huber, raw force excluded. No refit, retuning, or reweighting.
- Checkpoints: published Stage 3 C3 and Stage 5 B1/B2 bytes only.

## Repaired object

- Alphabet: `K=64` deterministic ID-stable K-medoids in the frozen C3 bi-encoder embedding.
- Selection: C3 squared embedding distance only inside the atlas.
- C4 is absent from the R1 import/call graph; no reranker, blend, gate, C5, C6, or synthesized
  action is permitted.
- The decoded local index is mapped through `candidate_source_index`; physical effects are looked
  up only from executed `true_distance` entries.

## Ordering firewall

1. Verify history and data hashes.
2. Reproduce D1/D2/D3 within `1e-4` relative tolerance.
3. Commit this file and `REPAIRED_DEFINITION.json`.
4. Compute Gate H without reading effect-error values.
5. Only if Gate H passes, compute the comparator ladder, controls, bootstrap, Gate A, final
   disposition, and report.

## Gate H

On development: median distinct-code utilization divided by 64 must be `>0.50`; clipped fraction
must be `<0.05`; pooled dead-code fraction `<0.10`; action RMSE no more than `1.25x` the strongest
deployable K=64 baseline; every state must expose at least 96 valid candidates. Failure freezes
`QUANTIZER_STILL_DEGENERATE` and forbids any effect-error comparison.

## Comparator ladder and controls

Recompute O_STATE_FULL/K64, O_STATIC_FULL, O_CONTACT_FULL, O_PHASE_FULL, B1/B2 FULL/K64, C3_FULL,
C5_FROZEN, R1_REPAIRED_K64, and A0_ACTUATOR_UNIFORM on identical rows. Run 20 random atlases,
task-wise shuffled C3 embedding, label shuffle, and raw-action distance. Use 10,000 paired
source-episode-clustered bootstrap replicates with seed `13150603`.

## Disposition

Apply the prompt's precedence exactly and emit one of the seven registered dispositions. A pass
only authorizes Stage 6-B review; it does not alter Stage 1-5 conclusions or establish policy/VLA
task-success evidence.
"""


def run(project_root, output_root=None):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    os.makedirs(output_root, exist_ok=True)
    history = historical_binding(project_root, output_root)
    if not history["passed"]:
        return history
    data = data_source_binding(project_root, output_root)
    if not data["passed"]:
        return data
    defects = defect_reproduction(project_root, output_root)
    if not defects["passed"]:
        return defects
    repaired_definition(project_root, output_root)
    prereg_path = os.path.join(output_root, "PREREGISTRATION.md")
    with open(prereg_path, "w", encoding="utf-8") as handle:
        handle.write(PREREGISTRATION)
    return {"passed": True, "output_root": output_root}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root, args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()
