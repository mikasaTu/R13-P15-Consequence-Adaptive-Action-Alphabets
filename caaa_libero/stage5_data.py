"""Stage 5 local-bank caches and strict split-disjoint reversal tuples."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict

import numpy as np

from .stage3_data import CONTEXT_SLICES
from .stage4_collection import _paths_for_key
from .stage4_config import SCRATCH_ROOT as STAGE4_SCRATCH_ROOT
from .stage4_data import (
    _cache_path as stage4_cache_path,
    historical_records,
    load_cache as load_stage4_cache,
)
from .stage5_config import (
    LOCAL_BANK_SIZE,
    OUTPUT_RELATIVE,
    PHASES,
    REVERSAL_ATTEMPT_MULTIPLIER,
    REVERSAL_CALIBRATION_QUOTA_PER_TASK_PHASE,
    REVERSAL_DEVELOPMENT_QUOTA_PER_TASK_PHASE,
    REVERSAL_MARGIN_QUANTILE,
    REVERSAL_SEED,
    REVERSAL_TRAIN_QUOTA_PER_TASK_PHASE,
    SCRATCH_ROOT,
    TASK_IDS,
)
from .storage import atomic_json, atomic_npz, mark_complete, sha256_file, validate_complete


CACHE_SCHEMA = "stage5-local-m128-cache-v1"
SPLITS = ("train", "calibration", "development", "historical_exploratory")
STAGE4_CACHE_NAMES = {
    "train": "train_matrix_cache",
    "calibration": "historical_calibration_matrix_cache",
    "development": "historical_development_matrix_cache",
    "historical_exploratory": "historical_confirmation_matrix_cache",
}
STAGE4_RECORD_SPLITS = {
    "calibration": "calibration",
    "development": "development",
    "historical_exploratory": "confirmation",
}
REVERSAL_QUOTAS = {
    "train": REVERSAL_TRAIN_QUOTA_PER_TASK_PHASE,
    "calibration": REVERSAL_CALIBRATION_QUOTA_PER_TASK_PHASE,
    "development": REVERSAL_DEVELOPMENT_QUOTA_PER_TASK_PHASE,
}


def _seed(*parts):
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def cache_path(scratch_root, split):
    return os.path.join(scratch_root, "derived", "stage5_%s_cache.npz" % split)


def load_cache(path):
    valid, evidence = validate_complete(path)
    if not valid:
        raise RuntimeError("incomplete Stage 5 cache %s: %s" % (path, evidence))
    with np.load(path, allow_pickle=False) as data:
        output = {name: np.asarray(data[name]).copy() for name in data.files}
    if str(output["schema_version"].item()) != CACHE_SCHEMA:
        raise RuntimeError("Stage 5 cache schema changed")
    return output


def _contacts_for_train(project_root, expected_keys, local_source_indices):
    output_root = os.path.join(project_root, "experiments/r13_p15_cr_trca/stage4")
    manifest = _load_json(os.path.join(output_root, "TRAINING_STATE_MANIFEST.json"))
    by_key = {str(row["key"]): row for row in manifest["records"]}
    target, candidate = [], []
    for key in expected_keys:
        if str(key) not in by_key:
            raise RuntimeError("Stage 4 train key missing: " + str(key))
        paths = _paths_for_key(str(key), STAGE4_SCRATCH_ROOT)
        for name in ("support", "candidate"):
            valid, evidence = validate_complete(paths[name])
            if not valid:
                raise RuntimeError("incomplete Stage 4 shard %s: %s" % (paths[name], evidence))
        with np.load(paths["support"], allow_pickle=False) as data:
            target.append(np.asarray(data["contact_mode"][1:], dtype=np.int8))
        with np.load(paths["candidate"], allow_pickle=False) as data:
            values = np.asarray(data["contact_mode"][1:], dtype=np.int8)
            bank_index = np.asarray(data["bank_index"][1:], dtype=np.int64)
            if not np.array_equal(bank_index, np.arange(len(bank_index))):
                raise RuntimeError("Stage 4 candidate order changed")
            candidate.append(values[local_source_indices])
    return np.asarray(target), np.asarray(candidate)


def _contacts_for_historical(split, expected_keys, local_source_indices):
    records = historical_records(STAGE4_RECORD_SPLITS[split])
    by_key = {str(row["meta"]["key"]): row for row in records}
    target, candidate = [], []
    for key in expected_keys:
        record = by_key.get(str(key))
        if record is None:
            raise RuntimeError("historical record missing: " + str(key))
        target.append(np.asarray(record["support"]["contact_mode"][1:], dtype=np.int8))
        values = np.asarray(record["candidate"]["contact_mode"][1:], dtype=np.int8)
        bank_index = np.asarray(record["candidate"]["bank_index"][1:], dtype=np.int64)
        if not np.array_equal(bank_index, np.arange(len(bank_index))):
            raise RuntimeError("historical candidate order changed")
        candidate.append(values[local_source_indices])
    return np.asarray(target), np.asarray(candidate)


def build_cache(project_root, split, output_root=None, scratch_root=SCRATCH_ROOT):
    if split not in SPLITS:
        raise KeyError(split)
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    with np.load(os.path.join(output_root, "LOCAL_BANK.npz"), allow_pickle=False) as data:
        local_source_indices = np.asarray(data["source_indices"], dtype=np.int64)
        local_residuals = np.asarray(data["residuals"], dtype=np.float32)
    source_path = stage4_cache_path(
        STAGE4_SCRATCH_ROOT, STAGE4_CACHE_NAMES[split]
    )
    source = load_stage4_cache(source_path)
    keys = source["key"].astype(str)
    if split == "train":
        target_contact, candidate_contact = _contacts_for_train(
            project_root, keys, local_source_indices
        )
    else:
        target_contact, candidate_contact = _contacts_for_historical(
            split, keys, local_source_indices
        )
    context = np.asarray(source["context"], dtype=np.float32)
    left, right = CONTEXT_SLICES["nominal_action"]
    nominal = (
        context[:, left:right]
        * np.asarray(source["context_scale"], dtype=np.float32)[None, left:right]
        + np.asarray(source["context_center"], dtype=np.float32)[None, left:right]
    )
    current_contact_index = CONTEXT_SLICES["current_contact"][0]
    current_contact = np.rint(context[:, current_contact_index]).astype(np.int8)
    true_distance = np.asarray(
        source["true_distance"][:, :, local_source_indices], dtype=np.float32
    )
    if true_distance.shape[2] != LOCAL_BANK_SIZE:
        raise AssertionError(true_distance.shape)
    arrays = {
        "schema_version": np.asarray(CACHE_SCHEMA),
        "source_stage4_cache_sha256": np.asarray(sha256_file(source_path)),
        "split": np.asarray(split),
        "context": context,
        "nominal_action": nominal.astype(np.float32),
        "target_residual": np.asarray(source["target_residual"], dtype=np.float32),
        "candidate_residual": local_residuals,
        "candidate_source_index": local_source_indices,
        "true_distance": true_distance,
        "target_contact_mode": target_contact,
        "candidate_contact_mode": candidate_contact,
        "current_contact": current_contact,
        "key": keys,
        "task_id": source["task_id"].astype(str),
        "task_index": np.asarray(source["task_index"], dtype=np.int8),
        "episode_id": np.asarray(source["episode_id"], dtype=np.int16),
        "phase": source["phase"].astype(str),
        "phase_index": np.asarray(source["phase_index"], dtype=np.int8),
        "snapshot_index": np.asarray(source["snapshot_index"], dtype=np.int32),
        "direction_family_id": np.asarray(source["direction_family_id"], dtype=np.int8),
        "consequence_scale": np.asarray(source["consequence_scale"], dtype=np.float64),
        "context_center": np.asarray(source["context_center"], dtype=np.float32),
        "context_scale": np.asarray(source["context_scale"], dtype=np.float32),
    }
    destination = cache_path(scratch_root, split)
    atomic_npz(destination, **arrays)
    mark_complete(
        destination,
        {
            "kind": "stage5_local_bank_matrix_cache",
            "schema_version": CACHE_SCHEMA,
            "split": split,
            "states": len(context),
            "targets": int(true_distance.shape[1]),
            "candidates": int(true_distance.shape[2]),
        },
    )
    return {
        "split": split,
        "path": destination,
        "sha256": sha256_file(destination),
        "bytes": int(os.path.getsize(destination)),
        "states": int(len(context)),
        "targets": int(true_distance.shape[1]),
        "candidates": int(true_distance.shape[2]),
        "minimum_valid_candidates": LOCAL_BANK_SIZE,
    }


def build_all_caches(project_root, output_root=None, scratch_root=SCRATCH_ROOT):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    rows = [
        build_cache(project_root, split, output_root, scratch_root) for split in SPLITS
    ]
    manifest = {
        "schema_version": CACHE_SCHEMA,
        "local_bank_sha256": sha256_file(os.path.join(output_root, "LOCAL_BANK.npz")),
        "caches": rows,
        "development_values_inspected_during_build": False,
    }
    atomic_json(os.path.join(output_root, "STAGE5_CACHE_MANIFEST.json"), manifest)
    return manifest


def _train_margins(train):
    samples = defaultdict(list)
    distance = train["true_distance"]
    for state in range(len(distance)):
        task = str(train["task_id"][state])
        phase = str(train["phase"][state])
        rng = np.random.RandomState(
            _seed(
                REVERSAL_SEED,
                "margin",
                task,
                phase,
                int(train["episode_id"][state]),
                int(train["snapshot_index"][state]),
            )
        )
        for _ in range(64):
            target = int(rng.randint(distance.shape[1]))
            left, right = rng.choice(distance.shape[2], size=2, replace=False)
            gap = abs(float(distance[state, target, left] - distance[state, target, right]))
            if gap > 1e-12:
                samples[(task, phase)].append(gap)
    margins = {}
    for task in TASK_IDS:
        for phase in PHASES:
            values = samples[(task, phase)]
            if not values:
                raise RuntimeError("no train margin observations for %s/%s" % (task, phase))
            margins[(task, phase)] = float(
                np.quantile(values, REVERSAL_MARGIN_QUANTILE)
            )
    return margins, samples


def _same_contact_pair(rng, states, current_contact):
    categories = []
    for value in (0, 1):
        group = states[current_contact[states] == value]
        if len(group) >= 2:
            categories.append(group)
    if categories:
        group = categories[int(rng.randint(len(categories)))]
        left, right = rng.choice(group, size=2, replace=False)
        return int(left), int(right), True
    left, right = rng.choice(states, size=2, replace=False)
    return int(left), int(right), False


def _sample_reversal_rate(cache, margins, attempts_per_task_phase=8192):
    rows = []
    distance = cache["true_distance"]
    for task in TASK_IDS:
        for phase in PHASES:
            states = np.flatnonzero(
                (cache["task_id"].astype(str) == task)
                & (cache["phase"].astype(str) == phase)
            )
            if len(states) < 2:
                continue
            rng = np.random.RandomState(
                _seed(REVERSAL_SEED, "rate", str(cache["split"].item()), task, phase)
            )
            valid = 0
            eligible = 0
            same_contact = 0
            for attempt in range(int(attempts_per_task_phase)):
                target = int(attempt % distance.shape[1])
                s1, s2, matched = _same_contact_pair(
                    rng, states, cache["current_contact"]
                )
                same_contact += int(matched)
                left_pool = np.argsort(distance[s1, target], kind="stable")[:32]
                right_pool = np.argsort(distance[s2, target], kind="stable")[:32]
                i = int(left_pool[int(rng.randint(len(left_pool)))])
                j = int(right_pool[int(rng.randint(len(right_pool)))])
                if i == j:
                    continue
                eligible += 1
                margin = margins[(task, phase)]
                if (
                    distance[s1, target, i] + margin < distance[s1, target, j]
                    and distance[s2, target, j] + margin < distance[s2, target, i]
                ):
                    valid += 1
            rows.append(
                {
                    "split": str(cache["split"].item()),
                    "task_id": task,
                    "phase": phase,
                    "attempts": int(attempts_per_task_phase),
                    "eligible": int(eligible),
                    "valid": int(valid),
                    "reversal_rate": float(valid / max(eligible, 1)),
                    "same_contact_pair_fraction": float(
                        same_contact / float(attempts_per_task_phase)
                    ),
                }
            )
    return rows


def _construct_pairs(cache, margins, quota):
    split = str(cache["split"].item())
    distance = cache["true_distance"]
    rows = []
    undersupplied = []
    for task in TASK_IDS:
        for phase in PHASES:
            states = np.flatnonzero(
                (cache["task_id"].astype(str) == task)
                & (cache["phase"].astype(str) == phase)
            )
            rng = np.random.RandomState(
                _seed(REVERSAL_SEED, "pairs", split, task, phase)
            )
            group = []
            seen = set()
            target_offset = int(rng.randint(distance.shape[1]))
            attempt_limit = max(20000, int(quota) * REVERSAL_ATTEMPT_MULTIPLIER)
            for attempt in range(attempt_limit):
                if len(group) >= int(quota):
                    break
                target = int((target_offset + attempt) % distance.shape[1])
                s1, s2, matched = _same_contact_pair(
                    rng, states, cache["current_contact"]
                )
                left = distance[s1, target]
                right = distance[s2, target]
                i_pool = np.argsort(left, kind="stable")[:32]
                j_pool = np.argsort(right, kind="stable")[:32]
                left_gap = left[j_pool][None, :] - left[i_pool][:, None]
                right_gap = right[j_pool][None, :] - right[i_pool][:, None]
                margin = margins[(task, phase)]
                valid = np.argwhere((left_gap > margin) & (right_gap < -margin))
                if not len(valid):
                    continue
                order = rng.permutation(len(valid))
                accepted = None
                for position in order:
                    i_pos, j_pos = valid[int(position)]
                    candidate_i = int(i_pool[int(i_pos)])
                    candidate_j = int(j_pool[int(j_pos)])
                    identity = (
                        str(cache["key"][s1]),
                        str(cache["key"][s2]),
                        target,
                        int(cache["candidate_source_index"][candidate_i]),
                        int(cache["candidate_source_index"][candidate_j]),
                    )
                    if identity not in seen:
                        accepted = (identity, candidate_i, candidate_j)
                        break
                if accepted is None:
                    continue
                identity, candidate_i, candidate_j = accepted
                seen.add(identity)
                row = {
                    "split": split,
                    "task_id": task,
                    "phase": phase,
                    "current_contact_s1": int(cache["current_contact"][s1]),
                    "current_contact_s2": int(cache["current_contact"][s2]),
                    "same_current_contact": bool(matched),
                    "state_s1": int(s1),
                    "state_s2": int(s2),
                    "state_key_s1": str(cache["key"][s1]),
                    "state_key_s2": str(cache["key"][s2]),
                    "episode_s1": int(cache["episode_id"][s1]),
                    "episode_s2": int(cache["episode_id"][s2]),
                    "target_id": target,
                    "direction_family_id": int(cache["direction_family_id"][target]),
                    "candidate_i": candidate_i,
                    "candidate_j": candidate_j,
                    "candidate_source_i": int(
                        cache["candidate_source_index"][candidate_i]
                    ),
                    "candidate_source_j": int(
                        cache["candidate_source_index"][candidate_j]
                    ),
                    "margin": float(margin),
                    "true_gap_s1_j_minus_i": float(
                        left[candidate_j] - left[candidate_i]
                    ),
                    "true_gap_s2_j_minus_i": float(
                        right[candidate_j] - right[candidate_i]
                    ),
                }
                row["tuple_sha256"] = hashlib.sha256(
                    json.dumps(identity, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                group.append(row)
            if len(group) < int(quota):
                undersupplied.append(
                    {
                        "split": split,
                        "task_id": task,
                        "phase": phase,
                        "requested": int(quota),
                        "realized": len(group),
                    }
                )
            rows.extend(group)
    return rows, undersupplied


def build_reversals(project_root, output_root=None, scratch_root=SCRATCH_ROOT):
    import pandas as pd

    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    caches = {
        split: load_cache(cache_path(scratch_root, split))
        for split in ("train", "calibration", "development")
    }
    margins, samples = _train_margins(caches["train"])
    rows = []
    undersupplied = []
    rates = []
    for split, cache in caches.items():
        built, missing = _construct_pairs(cache, margins, REVERSAL_QUOTAS[split])
        rows.extend(built)
        undersupplied.extend(missing)
        rates.extend(_sample_reversal_rate(cache, margins))
    frame = pd.DataFrame(rows)
    path = os.path.join(output_root, "CONTEXT_REVERSAL_PAIRS.parquet")
    frame.to_parquet(path, index=False)
    tuple_counts = Counter(frame["tuple_sha256"].astype(str).tolist())
    overlap = [key for key, count in tuple_counts.items() if count != 1]
    episode_sets = {
        split: set(caches[split]["episode_id"].astype(int).tolist()) for split in caches
    }
    episode_overlap = {
        "%s__%s" % (left, right): sorted(episode_sets[left] & episode_sets[right])
        for i, left in enumerate(episode_sets)
        for right in list(episode_sets)[i + 1 :]
    }
    phase_rates = {}
    for split in caches:
        for phase in PHASES:
            chosen = [
                row for row in rates if row["split"] == split and row["phase"] == phase
            ]
            numerator = sum(row["valid"] for row in chosen)
            denominator = sum(row["eligible"] for row in chosen)
            phase_rates["%s/%s" % (split, phase)] = float(
                numerator / max(denominator, 1)
            )
    metadata = {
        "kind": "stage5_strict_context_reversal_benchmark",
        "source_splits": {
            "train": "episodes 16-31",
            "calibration": "episodes 32-35",
            "development": "episodes 36-39",
        },
        "pair_count": len(rows),
        "pair_counts_by_split": dict(Counter(row["split"] for row in rows)),
        "pair_counts_by_split_task_phase": dict(
            Counter(
                "%s/%s/%s" % (row["split"], row["task_id"], row["phase"])
                for row in rows
            )
        ),
        "requested_quota_per_task_phase": dict(REVERSAL_QUOTAS),
        "undersupplied_strata": undersupplied,
        "strict_definition": (
            "D_s1(target,i)+margin<D_s1(target,j) and "
            "D_s2(target,j)+margin<D_s2(target,i)"
        ),
        "margin_quantile": REVERSAL_MARGIN_QUANTILE,
        "margin_fit_split": "train episodes 16-31 only",
        "margins": {
            "%s/%s" % key: value for key, value in sorted(margins.items())
        },
        "margin_sample_counts": {
            "%s/%s" % key: len(value) for key, value in sorted(samples.items())
        },
        "rate_sampling": {
            "definition": (
                "same-contact state pair when available; i sampled from s1 top32, "
                "j sampled from s2 top32; fixed 8192 attempts per split/task/phase"
            ),
            "rows": rates,
            "phase_rates": phase_rates,
        },
        "same_contact_preferred": True,
        "margin_relaxed": False,
        "labels_fabricated": False,
        "exact_tuple_overlap_count": len(overlap),
        "exact_tuple_overlap_hashes": overlap,
        "episode_overlap": episode_overlap,
        "split_disjoint": not overlap and not any(episode_overlap.values()),
        "parquet_sha256": sha256_file(path),
    }
    atomic_json(os.path.join(output_root, "CONTEXT_REVERSAL_METADATA.json"), metadata)
    return metadata


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build-caches", "build-reversals"))
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--scratch-root", default=SCRATCH_ROOT)
    args = parser.parse_args(argv)
    if args.command == "build-caches":
        result = build_all_caches(args.project_root, args.output_root, args.scratch_root)
    else:
        result = build_reversals(args.project_root, args.output_root, args.scratch_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
