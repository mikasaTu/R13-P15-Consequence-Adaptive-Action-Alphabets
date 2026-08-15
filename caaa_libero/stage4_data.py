"""Frozen Stage 4 matrix caches and context-reversal pair construction.

Large all-candidate matrices are derived from hash-checked simulator shards and
kept on CPFS.  The repository receives their content hashes and the compact
reversal-pair table needed to reproduce training.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict

import numpy as np

from .stage2_analysis import PHASE_TO_ID, TASK_TO_ID
from .stage3_config import SCRATCH_ROOT as HISTORICAL_SCRATCH_ROOT
from .stage3_data import (
    CONTEXT_SLICES,
    fit_context_scaler,
    load_records,
    raw_context,
    true_distance_matrix,
)
from .stage4_collection import _paths_for_key
from .stage4_config import (
    ACTION_BANK_SIZE,
    CR_PAIR_MARGIN_QUANTILE,
    CR_REVERSALS_PER_TASK_PHASE,
    HISTORICAL_REPOSITORY_ROOT,
    HISTORICAL_STAGE3_RELATIVE,
    OUTPUT_RELATIVE,
    PHASES,
    REVERSAL_PAIR_SEED,
    SCRATCH_ROOT,
    SUPPORT_TARGET_COUNT,
    TASK_IDS,
)
from .storage import atomic_json, atomic_npz, sha256_file, validate_complete


CACHE_SCHEMA = "stage4-context-distance-cache-v1"


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_checked(path):
    valid, evidence = validate_complete(path)
    if not valid:
        raise RuntimeError("incomplete Stage 4 shard %s: %s" % (path, evidence))
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]).copy() for name in data.files}


def load_expanded_records(project_root, output_root=None, scratch_root=SCRATCH_ROOT):
    """Load the 768 frozen train states in their manifest order."""
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    collection = _load_json(os.path.join(output_root, "expanded_training_collection.json"))
    if not collection.get("passed"):
        raise RuntimeError("expanded training collection has not passed verification")
    manifest = _load_json(os.path.join(output_root, "TRAINING_STATE_MANIFEST.json"))
    records = []
    for meta in manifest["records"]:
        paths = _paths_for_key(meta["key"], scratch_root)
        context = _load_checked(paths["context"])
        support = _load_checked(paths["support"])
        candidate = _load_checked(paths["candidate"])
        if support["residual_action"].shape != (SUPPORT_TARGET_COUNT + 1, 24):
            raise RuntimeError("support semantics changed for " + meta["key"])
        if candidate["residual_action"].shape != (ACTION_BANK_SIZE + 1, 24):
            raise RuntimeError("candidate semantics changed for " + meta["key"])
        if not np.array_equal(candidate["bank_index"][1:], np.arange(ACTION_BANK_SIZE)):
            raise RuntimeError("candidate order changed for " + meta["key"])
        if not np.allclose(
            context["observable_state"], support["initial"][0], rtol=0.0, atol=1e-12
        ):
            raise RuntimeError("context/support state mismatch for " + meta["key"])
        record_meta = dict(meta)
        record_meta["split"] = "train"
        records.append(
            {
                "meta": record_meta,
                "context": context,
                "support": support,
                "candidate": candidate,
                "context_path": paths["context"],
                "support_path": paths["support"],
                "candidate_path": paths["candidate"],
            }
        )
    if len(records) != int(manifest["state_count"]):
        raise RuntimeError("expanded record count changed")
    return records


def historical_records(split):
    historical_output = os.path.join(
        HISTORICAL_REPOSITORY_ROOT, HISTORICAL_STAGE3_RELATIVE
    )
    return load_records(
        HISTORICAL_REPOSITORY_ROOT,
        historical_output,
        (split,),
        HISTORICAL_SCRATCH_ROOT,
    )


def _cache_arrays(records, consequence_scale, context_center, context_scale):
    contexts = np.stack(
        [
            ((raw_context(record) - context_center) / context_scale).astype(np.float32)
            for record in records
        ]
    )
    target_reference = np.asarray(
        records[0]["support"]["residual_action"][1:], dtype=np.float64
    )
    bank_reference = np.asarray(
        records[0]["candidate"]["residual_action"][1:], dtype=np.float64
    )
    target = target_reference.astype(np.float32)
    bank = bank_reference.astype(np.float32)
    distances = np.empty(
        (len(records), SUPPORT_TARGET_COUNT, ACTION_BANK_SIZE), dtype=np.float32
    )
    keys = []
    task_ids = []
    episode_ids = []
    phases = []
    snapshot_indices = []
    family_ids = np.asarray(
        records[0]["support"]["direction_family_id"][1:], dtype=np.int8
    )
    for index, record in enumerate(records):
        if not np.array_equal(
            np.asarray(record["support"]["residual_action"][1:], dtype=np.float64),
            target_reference,
        ):
            raise RuntimeError("matched target residuals changed")
        if not np.array_equal(
            np.asarray(record["candidate"]["residual_action"][1:], dtype=np.float64),
            bank_reference,
        ):
            raise RuntimeError("common action bank changed")
        distances[index] = true_distance_matrix(record, consequence_scale)
        meta = record["meta"]
        keys.append(str(meta["key"]))
        task_ids.append(str(meta["task_id"]))
        episode_ids.append(int(meta["episode_id"]))
        phases.append(str(meta["phase"]))
        snapshot_indices.append(int(meta["snapshot_index"]))
    return {
        "schema_version": np.asarray(CACHE_SCHEMA),
        "context": contexts,
        "target_residual": target,
        "candidate_residual": bank,
        "true_distance": distances,
        "key": np.asarray(keys),
        "task_id": np.asarray(task_ids),
        "task_index": np.asarray([TASK_TO_ID[value] for value in task_ids], dtype=np.int8),
        "episode_id": np.asarray(episode_ids, dtype=np.int16),
        "phase": np.asarray(phases),
        "phase_index": np.asarray([PHASE_TO_ID[value] for value in phases], dtype=np.int8),
        "snapshot_index": np.asarray(snapshot_indices, dtype=np.int32),
        "direction_family_id": family_ids,
        "consequence_scale": np.asarray(consequence_scale, dtype=np.float64),
        "context_center": np.asarray(context_center, dtype=np.float32),
        "context_scale": np.asarray(context_scale, dtype=np.float32),
    }


def _cache_path(scratch_root, name):
    return os.path.join(scratch_root, "derived", name + ".npz")


def load_cache(path):
    valid, evidence = validate_complete(path)
    if not valid:
        raise RuntimeError("incomplete matrix cache %s: %s" % (path, evidence))
    with np.load(path, allow_pickle=False) as data:
        output = {name: np.asarray(data[name]).copy() for name in data.files}
    if str(output["schema_version"].item()) != CACHE_SCHEMA:
        raise RuntimeError("matrix cache schema changed")
    return output


def build_training_cache(project_root, output_root=None, scratch_root=SCRATCH_ROOT):
    """Fit expanded-train-only scalers and materialize all true distance matrices."""
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    records = load_expanded_records(project_root, output_root, scratch_root)
    # Preserve the frozen BALANCED_TASK_EFFECT definition used by B2/C3 and
    # bound in HISTORICAL_BINDING.json.  This scaler was fit on train episodes
    # 16-31 only; refitting it on denser Stage 4 timesteps would silently
    # redefine the primary metric and invalidate historical comparisons.
    historical_scaler = os.path.join(
        HISTORICAL_REPOSITORY_ROOT,
        HISTORICAL_STAGE3_RELATIVE,
        "model_scalers.npz",
    )
    with np.load(historical_scaler, allow_pickle=False) as data:
        consequence_scale = np.asarray(data["consequence_scale"], dtype=np.float64)
    consequence_evidence = {
        "source": "frozen Stage 3 train-only model_scalers.npz",
        "path": historical_scaler,
        "sha256": sha256_file(historical_scaler),
        "refit_for_stage4": False,
    }
    context_center, context_scale = fit_context_scaler(records)
    arrays = _cache_arrays(records, consequence_scale, context_center, context_scale)
    path = _cache_path(scratch_root, "train_matrix_cache")
    atomic_npz(path, **arrays)
    from .storage import mark_complete

    mark_complete(
        path,
        {
            "kind": "stage4_training_matrix_cache",
            "schema_version": CACHE_SCHEMA,
            "states": len(records),
            "targets_per_state": SUPPORT_TARGET_COUNT,
            "candidates_per_target": ACTION_BANK_SIZE,
        },
    )
    scaler_path = os.path.join(output_root, "stage4_scalers.npz")
    atomic_npz(
        scaler_path,
        consequence_scale=consequence_scale,
        context_center=context_center,
        context_scale=context_scale,
    )
    result = {
        "schema_version": CACHE_SCHEMA,
        "cache_path": path,
        "cache_sha256": sha256_file(path),
        "cache_bytes": os.path.getsize(path),
        "state_count": len(records),
        "query_count": len(records) * SUPPORT_TARGET_COUNT,
        "candidate_comparisons": len(records)
        * SUPPORT_TARGET_COUNT
        * ACTION_BANK_SIZE,
        "scaler_path": os.path.relpath(scaler_path, project_root),
        "scaler_sha256": sha256_file(scaler_path),
        "scaler_fit_split": "expanded train episodes 16-31 only",
        "consequence_scaling_evidence": consequence_evidence,
    }
    atomic_json(os.path.join(output_root, "expanded_training_dataset.json"), result)
    return result


def build_historical_cache(
    project_root,
    split,
    output_root=None,
    scratch_root=SCRATCH_ROOT,
):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    with np.load(os.path.join(output_root, "stage4_scalers.npz"), allow_pickle=False) as data:
        consequence_scale = np.asarray(data["consequence_scale"], dtype=np.float64)
        context_center = np.asarray(data["context_center"], dtype=np.float32)
        context_scale = np.asarray(data["context_scale"], dtype=np.float32)
    records = historical_records(split)
    arrays = _cache_arrays(records, consequence_scale, context_center, context_scale)
    path = _cache_path(scratch_root, "historical_%s_matrix_cache" % split)
    atomic_npz(path, **arrays)
    from .storage import mark_complete

    mark_complete(
        path,
        {
            "kind": "stage4_historical_matrix_cache",
            "schema_version": CACHE_SCHEMA,
            "source_split": split,
            "states": len(records),
        },
    )
    return {
        "split": split,
        "path": path,
        "sha256": sha256_file(path),
        "states": len(records),
    }


def _seed(*parts):
    value = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")


def train_robust_margins(cache):
    gaps = defaultdict(list)
    distance = cache["true_distance"]
    for state in range(len(distance)):
        key = (str(cache["task_id"][state]), str(cache["phase"][state]))
        rng = np.random.RandomState(
            _seed(
                REVERSAL_PAIR_SEED,
                key[0],
                int(cache["episode_id"][state]),
                key[1],
                int(cache["snapshot_index"][state]),
            )
        )
        for _ in range(64):
            target = int(rng.randint(distance.shape[1]))
            left, right = rng.choice(distance.shape[2], size=2, replace=False)
            gap = abs(float(distance[state, target, left] - distance[state, target, right]))
            if gap > 1e-12:
                gaps[key].append(gap)
    margins = {
        key: float(np.quantile(values, CR_PAIR_MARGIN_QUANTILE))
        for key, values in gaps.items()
    }
    return margins, gaps


def reversal_pairs(cache, margins, count_per_task_phase=CR_REVERSALS_PER_TASK_PHASE):
    """Create strict reversals, preserving evidence when a stratum has none.

    Free-space physics can be genuinely state invariant.  In that case no
    valid reversal exists and manufacturing one would leak a false label.  We
    therefore target equal quotas but return an explicitly undersupplied
    stratum rather than weakening the margin or changing the inequality.
    """
    rows = []
    distance = cache["true_distance"]
    for task in TASK_IDS:
        for phase in PHASES:
            states = np.flatnonzero(
                (cache["task_id"].astype(str) == task)
                & (cache["phase"].astype(str) == phase)
            )
            if len(states) < 2:
                raise RuntimeError("insufficient states for reversal group %s/%s" % (task, phase))
            margin = float(margins[(task, phase)])
            rng = np.random.RandomState(_seed(REVERSAL_PAIR_SEED, task, phase, len(states)))
            seen = set()
            group_rows = []
            target_counts = np.zeros(SUPPORT_TARGET_COUNT, dtype=np.int64)
            target_order = rng.permutation(SUPPORT_TARGET_COUNT)
            attempt_limit = max(20000, int(count_per_task_phase) * 200)
            for attempt in range(attempt_limit):
                if len(group_rows) >= int(count_per_task_phase):
                    break
                # Cycle targets in a frozen order. Targets with no physical
                # reversal cannot starve the remaining valid targets.
                target_id = int(target_order[attempt % SUPPORT_TARGET_COUNT])
                state_left, state_right = rng.choice(states, size=2, replace=False)
                left = distance[state_left, target_id]
                right = distance[state_right, target_id]
                candidate_i_pool = np.argsort(left, kind="stable")[:32]
                candidate_j_pool = np.argsort(right, kind="stable")[:32]
                left_gap = (
                    left[candidate_j_pool][None, :]
                    - left[candidate_i_pool][:, None]
                )
                right_gap = (
                    right[candidate_j_pool][None, :]
                    - right[candidate_i_pool][:, None]
                )
                valid = np.argwhere((left_gap > margin) & (right_gap < -margin))
                if not len(valid):
                    continue
                order = rng.permutation(len(valid))
                accepted = None
                for position in order:
                    i_position, j_position = valid[int(position)]
                    candidate_i = int(candidate_i_pool[int(i_position)])
                    candidate_j = int(candidate_j_pool[int(j_position)])
                    identity = (
                        int(state_left),
                        int(state_right),
                        target_id,
                        candidate_i,
                        candidate_j,
                    )
                    if identity not in seen:
                        accepted = (identity, candidate_i, candidate_j)
                        break
                if accepted is None:
                    continue
                identity, candidate_i, candidate_j = accepted
                seen.add(identity)
                exact_left_gap = float(left[candidate_j] - left[candidate_i])
                exact_right_gap = float(right[candidate_j] - right[candidate_i])
                group_rows.append(
                    {
                        "task_id": task,
                        "phase": phase,
                        "state_s1": int(state_left),
                        "state_s2": int(state_right),
                        "state_key_s1": str(cache["key"][state_left]),
                        "state_key_s2": str(cache["key"][state_right]),
                        "episode_s1": int(cache["episode_id"][state_left]),
                        "episode_s2": int(cache["episode_id"][state_right]),
                        "target_id": target_id,
                        "direction_family_id": int(
                            cache["direction_family_id"][target_id]
                        ),
                        "candidate_i": candidate_i,
                        "candidate_j": candidate_j,
                        "margin": margin,
                        "true_gap_s1_j_minus_i": exact_left_gap,
                        "true_gap_s2_j_minus_i": exact_right_gap,
                    }
                )
                target_counts[target_id] += 1
            rows.extend(group_rows)
    return rows


def build_reversal_artifact(project_root, output_root=None, scratch_root=SCRATCH_ROOT):
    import pandas as pd

    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    cache_path = _cache_path(scratch_root, "train_matrix_cache")
    cache = load_cache(cache_path)
    margins, samples = train_robust_margins(cache)
    rows = reversal_pairs(cache, margins)
    path = os.path.join(output_root, "CONTEXT_REVERSAL_PAIRS.parquet")
    pd.DataFrame(rows).to_parquet(path, index=False)
    metadata = {
        "source": "expanded train episodes 16-31 only",
        "pair_count": len(rows),
        "requested_pairs_per_task_phase": CR_REVERSALS_PER_TASK_PHASE,
        "realized_pairs_by_task_phase": {
            "%s/%s" % (task, phase): sum(
                row["task_id"] == task and row["phase"] == phase for row in rows
            )
            for task in TASK_IDS
            for phase in PHASES
        },
        "strictly_undersupplied_task_phases": [
            "%s/%s" % (task, phase)
            for task in TASK_IDS
            for phase in PHASES
            if sum(row["task_id"] == task and row["phase"] == phase for row in rows)
            < CR_REVERSALS_PER_TASK_PHASE
        ],
        "balanced_target_quotas_when_physics_supplies_reversals": True,
        "margin_never_relaxed_and_labels_never_fabricated": True,
        "strict_definition": "D_s1(t,i)+margin<D_s1(t,j) and D_s2(t,j)+margin<D_s2(t,i)",
        "margin_quantile": CR_PAIR_MARGIN_QUANTILE,
        "margins": {"%s/%s" % key: value for key, value in sorted(margins.items())},
        "margin_sample_counts": {
            "%s/%s" % key: len(value) for key, value in sorted(samples.items())
        },
        "sha256": sha256_file(path),
    }
    atomic_json(os.path.join(output_root, "context_reversal_metadata.json"), metadata)
    return metadata


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("build-train", "build-historical", "build-reversals")
    )
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--scratch-root", default=SCRATCH_ROOT)
    parser.add_argument(
        "--split", choices=("calibration", "development", "confirmation"), default="calibration"
    )
    args = parser.parse_args(argv)
    if args.command == "build-train":
        result = build_training_cache(args.project_root, args.output_root, args.scratch_root)
    elif args.command == "build-historical":
        result = build_historical_cache(
            args.project_root, args.split, args.output_root, args.scratch_root
        )
    else:
        result = build_reversal_artifact(args.project_root, args.output_root, args.scratch_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
