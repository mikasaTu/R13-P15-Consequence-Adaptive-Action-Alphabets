"""Stage 5 true-effect adaptivity decomposition on development episodes."""

from __future__ import annotations

import json
import os
import time
from collections import Counter

import numpy as np

from .stage2_analysis import PRIMARY_GROUPS, balanced_error
from .stage3_data import effect
from .stage3_metrics import realized_rows, write_csv
from .stage4_data import historical_records
from .stage5_config import (
    CONTACT_SENSITIVE_TASKS,
    GATES,
    LOCAL_BANK_SIZE,
    OUTPUT_RELATIVE,
    PHASES,
    PRIMARY_K,
    SCRATCH_ROOT,
    TASK_IDS,
)
from .stage5_data import cache_path, load_cache
from .storage import atomic_json, sha256_file


METHODS = (
    "O_STATE_FULL",
    "O_STATE_K64",
    "O_STATIC_FULL",
    "O_CONTACT_FULL",
    "O_PHASE_FULL",
)


def _stable_argmin(values, source_ids):
    values = np.asarray(values, dtype=np.float64)
    source_ids = np.asarray(source_ids, dtype=np.int64)
    minimum = float(np.min(values))
    tied = np.flatnonzero(np.isclose(values, minimum, rtol=0.0, atol=1e-15))
    return int(tied[np.argmin(source_ids[tied])])


def _assign(distance, medoids, source_ids):
    return np.asarray(
        [
            int(medoids[_stable_argmin(row[medoids], source_ids[medoids])])
            for row in np.asarray(distance)
        ],
        dtype=np.int64,
    )


def deterministic_kmedoids_precomputed(distance, k, source_ids):
    """Deterministic PAM-like medoids with original-ID tie breaking."""
    distance = np.asarray(distance, dtype=np.float64)
    source_ids = np.asarray(source_ids, dtype=np.int64)
    if distance.shape != (len(source_ids), len(source_ids)):
        raise ValueError("distance matrix shape mismatch")
    if int(k) <= 0 or int(k) > len(source_ids):
        raise ValueError("invalid k")
    first = int(np.argmin(source_ids))
    medoids = [first]
    minimum = distance[first].copy()
    minimum[first] = -1.0
    while len(medoids) < int(k):
        maximum = float(np.max(minimum))
        tied = np.flatnonzero(np.isclose(minimum, maximum, rtol=0.0, atol=1e-15))
        choice = int(tied[np.argmin(source_ids[tied])])
        medoids.append(choice)
        minimum = np.minimum(minimum, distance[choice])
        minimum[np.asarray(medoids, dtype=np.int64)] = -1.0
    medoids = np.asarray(medoids, dtype=np.int64)
    for _ in range(100):
        assignment = np.empty(len(distance), dtype=np.int64)
        for row_id, row in enumerate(distance[:, medoids]):
            local = _stable_argmin(row, source_ids[medoids])
            assignment[row_id] = int(medoids[local])
        updated = []
        for old in medoids:
            members = np.flatnonzero(assignment == int(old))
            if not len(members):
                updated.append(int(old))
                continue
            objective = np.sum(distance[np.ix_(members, members)], axis=1)
            updated.append(int(members[_stable_argmin(objective, source_ids[members])]))
        updated = np.asarray(updated, dtype=np.int64)
        # Duplicate updates can only arise from numerical tie pathologies. Fill
        # deterministically with farthest nonmedoids without synthesizing data.
        unique = []
        for value in sorted(set(updated.tolist()), key=lambda i: int(source_ids[i])):
            unique.append(int(value))
        while len(unique) < int(k):
            keep = np.asarray(unique, dtype=np.int64)
            minimum = np.min(distance[:, keep], axis=1)
            minimum[keep] = -1.0
            maximum = float(np.max(minimum))
            tied = np.flatnonzero(
                np.isclose(minimum, maximum, rtol=0.0, atol=1e-15)
            )
            unique.append(int(tied[np.argmin(source_ids[tied])]))
        updated = np.asarray(unique, dtype=np.int64)
        updated = updated[np.argsort(source_ids[updated], kind="stable")]
        previous = medoids[np.argsort(source_ids[medoids], kind="stable")]
        if np.array_equal(updated, previous):
            medoids = updated
            break
        medoids = updated
    return medoids


def _candidate_distance(record, source_ids, consequence_scale):
    candidate_effect = effect(record["candidate"])[1:][source_ids]
    candidate_mask = np.asarray(record["candidate"]["mask"][1:], dtype=bool)[
        source_ids
    ]
    candidate_mode = np.asarray(
        record["candidate"]["contact_mode"][1:], dtype=np.int64
    )[source_ids]
    output = np.empty((len(source_ids), len(source_ids)), dtype=np.float64)
    for row in range(len(source_ids)):
        output[row] = balanced_error(
            np.repeat(candidate_effect[row][None, :], len(source_ids), axis=0),
            candidate_effect,
            np.repeat(candidate_mask[row][None, :], len(source_ids), axis=0),
            candidate_mask,
            np.full(len(source_ids), candidate_mode[row], dtype=np.int64),
            candidate_mode,
            consequence_scale,
        )
    output = 0.5 * (output + output.T)
    np.fill_diagonal(output, 0.0)
    return output


def _summaries(rows):
    metric_names = (
        "balanced_task_effect_error",
        "object_pose_error",
        "tcp_object_relative_pose_error",
        "contact_mode_preserved",
        "task_progress_abs_error",
        "action_reconstruction_rmse",
    ) + tuple("error_group_" + name for name in PRIMARY_GROUPS)
    partitions = [("pooled", "ALL", "ALL", "ALL")]
    partitions += [("task", task, "ALL", "ALL") for task in TASK_IDS]
    partitions += [("phase", "ALL", phase, "ALL") for phase in PHASES]
    families = sorted({int(row["direction_family_id"]) for row in rows})
    partitions += [
        ("direction_family", "ALL", "ALL", str(family)) for family in families
    ]
    output = []
    for method in METHODS:
        for level, task, phase, family in partitions:
            selected = [
                row
                for row in rows
                if row["method"] == method
                and (task == "ALL" or row["task_id"] == task)
                and (phase == "ALL" or row["phase"] == phase)
                and (
                    family == "ALL"
                    or str(row["direction_family_id"]) == str(family)
                )
            ]
            if not selected:
                continue
            summary = {
                "row_type": "summary",
                "method": method,
                "level": level,
                "task_id": task,
                "phase": phase,
                "direction_family_id": family,
                "n": len(selected),
                "mean_valid_bank_size": float(
                    np.mean([row["valid_bank_size"] for row in selected])
                ),
                "mean_atlas_size": float(
                    np.mean([row["atlas_size"] for row in selected])
                ),
            }
            for metric in metric_names:
                summary[metric] = float(
                    np.mean([float(row[metric]) for row in selected])
                )
            output.append(summary)
    return output


def _mean(summary, method, level="pooled", task="ALL", phase="ALL"):
    row = next(
        value
        for value in summary
        if value["method"] == method
        and value["level"] == level
        and value["task_id"] == task
        and value["phase"] == phase
    )
    return float(row["balanced_task_effect_error"])


def run(project_root, output_root=None, scratch_root=SCRATCH_ROOT):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    train = load_cache(cache_path(scratch_root, "train"))
    development = load_cache(cache_path(scratch_root, "development"))
    records = historical_records("development")
    if [str(row["meta"]["key"]) for row in records] != development["key"].astype(str).tolist():
        raise RuntimeError("development record/cache order mismatch")
    source_ids = np.asarray(development["candidate_source_index"], dtype=np.int64)
    consequence_scale = np.asarray(development["consequence_scale"], dtype=np.float64)
    static = np.mean(train["true_distance"], axis=0)
    by_contact = {
        contact: np.mean(
            train["true_distance"][train["current_contact"] == contact], axis=0
        )
        for contact in (0, 1)
    }
    by_phase = {
        phase: np.mean(
            train["true_distance"][train["phase"].astype(str) == phase], axis=0
        )
        for phase in PHASES
    }
    all_rows = []
    atlas_rows = []
    for state, record in enumerate(records):
        started = time.perf_counter()
        truth = np.asarray(development["true_distance"][state], dtype=np.float64)
        selected = {
            "O_STATE_FULL": np.asarray(
                [_stable_argmin(row, source_ids) for row in truth], dtype=np.int64
            ),
            "O_STATIC_FULL": np.asarray(
                [_stable_argmin(row, source_ids) for row in static], dtype=np.int64
            ),
            "O_CONTACT_FULL": np.asarray(
                [
                    _stable_argmin(row, source_ids)
                    for row in by_contact[int(development["current_contact"][state])]
                ],
                dtype=np.int64,
            ),
            "O_PHASE_FULL": np.asarray(
                [
                    _stable_argmin(row, source_ids)
                    for row in by_phase[str(development["phase"][state])]
                ],
                dtype=np.int64,
            ),
        }
        candidate_distance = _candidate_distance(record, source_ids, consequence_scale)
        medoids = deterministic_kmedoids_precomputed(
            candidate_distance, PRIMARY_K, source_ids
        )
        selected["O_STATE_K64"] = _assign(truth, medoids, source_ids)
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        atlas_rows.append(
            {
                "state_key": str(development["key"][state]),
                "task_id": str(development["task_id"][state]),
                "episode_id": int(development["episode_id"][state]),
                "phase": str(development["phase"][state]),
                "local_medoids": medoids.tolist(),
                "source_medoids": source_ids[medoids].tolist(),
                "atlas_size": len(medoids),
            }
        )
        for method, local_decoded in selected.items():
            original_decoded = source_ids[local_decoded]
            rows = realized_rows(
                record,
                original_decoded,
                method,
                consequence_scale,
                latency_ms=elapsed_ms / len(METHODS),
                extra={
                    "evidence": "STAGE5_DEVELOPMENT_ORACLE_ADAPTIVITY",
                    "valid_bank_size": LOCAL_BANK_SIZE,
                    "atlas_size": PRIMARY_K if method == "O_STATE_K64" else LOCAL_BANK_SIZE,
                },
            )
            for target_id, row in enumerate(rows):
                row["local_bank_index"] = int(local_decoded[target_id])
                row["source_bank_index"] = int(original_decoded[target_id])
                row["direction_family_id"] = int(
                    development["direction_family_id"][target_id]
                )
                row["target_id"] = int(target_id)
            all_rows.extend(rows)
    summary = _summaries(all_rows)
    path = os.path.join(output_root, "ORACLE_ADAPTIVITY_AUDIT.csv")
    write_csv(path, summary)
    from .stage3_metrics import write_csv as write_rows

    raw_path = os.path.join(output_root, "oracle_adaptivity_rows.parquet")
    import pandas as pd

    pd.DataFrame(all_rows).to_parquet(raw_path, index=False)
    atlas_path = os.path.join(output_root, "oracle_state_k64_atlases.json")
    atomic_json(atlas_path, {"states": atlas_rows})

    state_error = _mean(summary, "O_STATE_FULL")
    static_errors = {
        method: _mean(summary, method)
        for method in ("O_STATIC_FULL", "O_CONTACT_FULL")
    }
    strongest = min(static_errors, key=lambda name: (static_errors[name], name))
    strongest_error = static_errors[strongest]
    pooled_gain = (strongest_error - state_error) / max(strongest_error, 1e-12)
    contact_state_rows = [
        row
        for row in all_rows
        if row["method"] == "O_STATE_FULL"
        and row["phase"] in ("contact_onset", "post_contact")
    ]
    contact_base_rows = [
        row
        for row in all_rows
        if row["method"] == strongest
        and row["phase"] in ("contact_onset", "post_contact")
    ]
    contact_state = float(
        np.mean([row["balanced_task_effect_error"] for row in contact_state_rows])
    )
    contact_base = float(
        np.mean([row["balanced_task_effect_error"] for row in contact_base_rows])
    )
    contact_gain = (contact_base - contact_state) / max(contact_base, 1e-12)
    task_gains = {}
    for task in CONTACT_SENSITIVE_TASKS:
        state_value = _mean(summary, "O_STATE_FULL", level="task", task=task)
        base_value = _mean(summary, strongest, level="task", task=task)
        task_gains[task] = (base_value - state_value) / max(base_value, 1e-12)
    metadata = json.load(
        open(os.path.join(output_root, "CONTEXT_REVERSAL_METADATA.json"), "r", encoding="utf-8")
    )
    train_phase_rates = {
        phase: float(metadata["rate_sampling"]["phase_rates"]["train/" + phase])
        for phase in PHASES
    }
    contact_phases_passing = [
        phase
        for phase in ("pre_contact", "contact_onset", "post_contact")
        if train_phase_rates[phase]
        >= GATES["oracle_adaptivity"]["reversal_rate_min"]
    ]
    checks = {
        "pooled_gain": {
            "value": pooled_gain,
            "threshold": GATES["oracle_adaptivity"]["pooled_gain_min"],
            "passed": pooled_gain >= GATES["oracle_adaptivity"]["pooled_gain_min"],
        },
        "contact_onset_post_contact_gain": {
            "value": contact_gain,
            "threshold": GATES["oracle_adaptivity"]["contact_phase_gain_min"],
            "passed": contact_gain
            >= GATES["oracle_adaptivity"]["contact_phase_gain_min"],
        },
        "contact_sensitive_tasks_improved": {
            "value": int(sum(value > 0.0 for value in task_gains.values())),
            "threshold": GATES["oracle_adaptivity"][
                "contact_sensitive_tasks_improved_min"
            ],
            "passed": sum(value > 0.0 for value in task_gains.values())
            >= GATES["oracle_adaptivity"]["contact_sensitive_tasks_improved_min"],
        },
        "strict_pair_count": {
            "value": int(metadata["pair_count"]),
            "threshold": GATES["oracle_adaptivity"]["strict_pair_count_min"],
            "passed": int(metadata["pair_count"])
            >= GATES["oracle_adaptivity"]["strict_pair_count_min"],
        },
        "contact_phases_with_reversal_rate": {
            "value": len(contact_phases_passing),
            "threshold": GATES["oracle_adaptivity"][
                "contact_phases_with_reversal_rate_min"
            ],
            "passed": len(contact_phases_passing)
            >= GATES["oracle_adaptivity"][
                "contact_phases_with_reversal_rate_min"
            ],
        },
    }
    gate_passed = all(value["passed"] for value in checks.values())
    gate = {
        "gate": "ORACLE_ADAPTIVITY_GATE_0",
        "split": "development episodes 36-39",
        "primary_bank_size": LOCAL_BANK_SIZE,
        "strongest_static_or_contact_method": strongest,
        "pooled_errors": {
            "O_STATE_FULL": state_error,
            **static_errors,
            "O_STATE_K64": _mean(summary, "O_STATE_K64"),
            "O_PHASE_FULL": _mean(summary, "O_PHASE_FULL"),
        },
        "state_specific_vs_static_or_contact_gain": pooled_gain,
        "contact_onset_post_contact_gain": contact_gain,
        "contact_sensitive_task_gains": task_gains,
        "full_to_k64_compression_loss": (
            _mean(summary, "O_STATE_K64") - state_error
        )
        / max(state_error, 1e-12),
        "state_vs_privileged_phase_gain": (
            _mean(summary, "O_PHASE_FULL") - state_error
        )
        / max(_mean(summary, "O_PHASE_FULL"), 1e-12),
        "strict_reversal": {
            "pair_count": int(metadata["pair_count"]),
            "train_phase_rates": train_phase_rates,
            "contact_phases_passing": contact_phases_passing,
            "exact_tuple_overlap_count": metadata["exact_tuple_overlap_count"],
            "split_disjoint": metadata["split_disjoint"],
        },
        "checks": checks,
        "passed": gate_passed,
        "failure_disposition": GATES["oracle_adaptivity"]["failure_disposition"],
        "remaining_registered_experiments_execute_after_failure": True,
        "artifact_hashes": {
            "audit_csv": sha256_file(path),
            "raw_rows_parquet": sha256_file(raw_path),
            "atlas_json": sha256_file(atlas_path),
        },
    }
    atomic_json(os.path.join(output_root, "ORACLE_ADAPTIVITY_GATE.json"), gate)
    return gate


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--scratch-root", default=SCRATCH_ROOT)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(args.project_root, args.output_root, args.scratch_root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
