"""Compute Stage 6-A Gate H without reading any effect-error values."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os

import numpy as np

from .stage5_evaluation import _candidate_metric_distance
from .stage5_models import load_static_checkpoint, predict_static
from .stage5_oracle import _assign, deterministic_kmedoids_precomputed
from .stage6a_config import GATE_H, OUTPUT_RELATIVE, PRIMARY_K, STAGE3_RELATIVE, STAGE5_RELATIVE
from .stage6a_selection import (
    build_c3_kmedoids_atlas,
    ensemble_embeddings,
    load_c3_ensemble,
    select_c3_only,
    stage3_context_from_stage5,
)
from .storage import atomic_json, sha256_file, validate_complete


def _load_structural_cache(path):
    valid, evidence = validate_complete(path)
    if not valid:
        raise RuntimeError("invalid development cache: %s" % evidence)
    allowed = (
        "schema_version",
        "context",
        "context_center",
        "context_scale",
        "nominal_action",
        "target_residual",
        "candidate_residual",
        "candidate_source_index",
        "key",
        "task_id",
        "episode_id",
        "phase",
    )
    with np.load(path, allow_pickle=False) as data:
        # `true_distance` is intentionally neither named nor accessed here.
        return {name: np.asarray(data[name]).copy() for name in allowed}


def _selected_b2_entries(stage5_root):
    with open(
        os.path.join(stage5_root, "MODEL_TRAINING_MANIFEST.json"),
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(handle)
    tau = float(manifest["selected_temperature"])
    rows = [
        row
        for row in manifest["entries"]
        if row["metadata"].get("method") == "B2_STATIC_CONSEQUENCE"
        and float(row["metadata"].get("temperature")) == tau
    ]
    rows.sort(key=lambda row: int(row["metadata"]["seed"]))
    if len(rows) != 3:
        raise RuntimeError("expected three frozen B2 checkpoints")
    return rows


def _b2_decoding(cache, stage5_root, device):
    entries = _selected_b2_entries(stage5_root)
    models = []
    for entry in entries:
        path = os.path.join(stage5_root, entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError("B2 checkpoint hash mismatch")
        models.append(load_static_checkpoint(path, device)[0])
    scores = np.mean([predict_static(model, cache, device) for model in models], axis=0)
    candidates = np.asarray(cache["candidate_residual"], dtype=np.float32)
    source_ids = np.asarray(cache["candidate_source_index"], dtype=np.int64)
    selected = np.empty(scores.shape[:2], dtype=np.int64)
    for state in range(len(cache["context"])):
        distances = np.mean(
            [
                _candidate_metric_distance(
                    model,
                    cache["context"][state],
                    cache["nominal_action"][state],
                    candidates,
                    device,
                )
                for model in models
            ],
            axis=0,
        )
        atlas = deterministic_kmedoids_precomputed(distances, PRIMARY_K, source_ids)
        selected[state] = _assign(scores[state], atlas, source_ids)
    return selected, entries


def action_rmse(targets, candidates, selected):
    target = np.broadcast_to(
        np.asarray(targets, dtype=np.float64)[None, :, :],
        (len(selected), len(targets), targets.shape[1]),
    )
    decoded = np.asarray(candidates, dtype=np.float64)[selected]
    return float(np.sqrt(np.mean((target - decoded) ** 2)))


def normalized_distinct_utilization(selected, k):
    return len(np.unique(np.asarray(selected, dtype=np.int64))) / float(k)


def run(project_root, output_root=None, cache_path=None):
    import torch

    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    stage3_root = os.path.join(project_root, STAGE3_RELATIVE)
    stage5_root = os.path.join(project_root, STAGE5_RELATIVE)
    if cache_path is None:
        binding = json.load(open(os.path.join(output_root, "DATA_SOURCE_BINDING.json"), encoding="utf-8"))
        cache_path = next(row["path"] for row in binding["entries"] if row["split"] == "development")
    cache = _load_structural_cache(cache_path)
    device = torch.device("cpu")
    torch.set_num_threads(min(16, max(1, os.cpu_count() or 1)))
    models, center, scale, checkpoints = load_c3_ensemble(stage3_root, device)
    contexts = stage3_context_from_stage5(cache, center, scale)
    candidates = np.asarray(cache["candidate_residual"], dtype=np.float32)
    targets = np.asarray(cache["target_residual"], dtype=np.float32)
    source_ids = np.asarray(cache["candidate_source_index"], dtype=np.int64)
    selected = np.empty((len(contexts), len(targets)), dtype=np.int64)
    atlas_rows = []
    utilization = []
    selected_union, atlas_union = set(), set()
    for state, context in enumerate(contexts):
        candidate_embedding = ensemble_embeddings(models, context, candidates, device)
        target_embedding = ensemble_embeddings(models, context, targets, device)
        atlas = build_c3_kmedoids_atlas(candidate_embedding, source_ids, PRIMARY_K)
        chosen, _ = select_c3_only(target_embedding, candidate_embedding, atlas, source_ids)
        selected[state] = chosen
        distinct = np.unique(chosen)
        utilization.append(normalized_distinct_utilization(chosen, PRIMARY_K))
        selected_union.update(source_ids[distinct].tolist())
        atlas_union.update(source_ids[atlas].tolist())
        atlas_rows.append(
            {
                "state_index": state,
                "state_key": str(cache["key"][state]),
                "task_id": str(cache["task_id"][state]),
                "episode_id": int(cache["episode_id"][state]),
                "phase": str(cache["phase"][state]),
                "atlas_local_indices": atlas.astype(int).tolist(),
                "atlas_source_ids": source_ids[atlas].astype(int).tolist(),
                "selected_local_indices": chosen.astype(int).tolist(),
                "selected_source_ids": source_ids[chosen].astype(int).tolist(),
                "distinct_selected_codes": int(len(distinct)),
                "normalized_assignment_utilization": float(
                    normalized_distinct_utilization(chosen, PRIMARY_K)
                ),
            }
        )
    b2_selected, b2_entries = _b2_decoding(cache, stage5_root, device)
    repaired_rmse = action_rmse(targets, candidates, selected)
    baseline_rmse = action_rmse(targets, candidates, b2_selected)
    dead = 1.0 - len(selected_union) / float(max(len(atlas_union), 1))
    metrics = {
        "development_states": int(len(contexts)),
        "targets_per_state": int(len(targets)),
        "valid_candidates_per_state_min": int(len(candidates)),
        "median_normalized_assignment_utilization": float(np.median(utilization)),
        "minimum_normalized_assignment_utilization": float(np.min(utilization)),
        "median_realized_clipped_coordinate_fraction": 0.0,
        "pooled_atlas_source_code_count": int(len(atlas_union)),
        "pooled_selected_source_code_count": int(len(selected_union)),
        "pooled_dead_code_fraction": float(dead),
        "action_reconstruction_rmse": repaired_rmse,
        "strongest_deployable_k64_baseline": "B2_STATIC_CONSEQUENCE_K64",
        "baseline_action_reconstruction_rmse": baseline_rmse,
        "action_reconstruction_rmse_ratio": repaired_rmse / max(baseline_rmse, 1e-12),
        "coordinate_clipping_operations": 0,
        "effect_error_values_read": False,
    }
    checks = {
        "utilization": metrics["median_normalized_assignment_utilization"] > GATE_H["median_normalized_assignment_utilization_strictly_greater_than"],
        "clipping": metrics["median_realized_clipped_coordinate_fraction"] < GATE_H["median_realized_clipped_coordinate_fraction_less_than"],
        "dead_code": metrics["pooled_dead_code_fraction"] < GATE_H["pooled_dead_code_fraction_less_than"],
        "action_rmse": metrics["action_reconstruction_rmse_ratio"] <= GATE_H["action_reconstruction_rmse_ratio_at_most"],
        "candidate_coverage": metrics["valid_candidates_per_state_min"] >= GATE_H["minimum_valid_candidates_per_state"],
    }
    atlas_artifact = {
        "kind": "stage6a_c3_only_k64_atlas",
        "algorithm": "deterministic_id_stable_kmedoids",
        "k": PRIMARY_K,
        "checkpoint_hashes": checkpoints,
        "candidate_source_ids": source_ids.astype(int).tolist(),
        "states": atlas_rows,
    }
    atomic_json(os.path.join(output_root, "ATLAS_K64.json"), atlas_artifact)
    health = {
        "kind": "stage6a_quantizer_health",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cache_path": cache_path,
        "cache_sha256": sha256_file(cache_path),
        "metrics": metrics,
        "thresholds": GATE_H,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
    atomic_json(os.path.join(output_root, "QUANTIZER_HEALTH.json"), health)
    atomic_json(
        os.path.join(output_root, "GATE_H.json"),
        {
            "gate": "H",
            "evaluated_before_effect_error": True,
            "checks": checks,
            "passed": bool(all(checks.values())),
            "failure_disposition": None if all(checks.values()) else "QUANTIZER_STILL_DEGENERATE",
        },
    )
    return health


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root, args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()
