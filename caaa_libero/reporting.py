"""Aggregate realized rollouts, bootstrap episodes, and write Stage 1 report."""

from __future__ import annotations

import csv
import glob
import json
import math
import os

import numpy as np

from . import config
from .analysis import BASELINE_METHODS
from .pipeline import utc_now
from .storage import atomic_json, atomic_text, sha256_file, sha256_tree, validate_complete


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write empty CSV %s" % path)
    fieldnames = list(rows[0].keys())
    temporary = path + ".incomplete"
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_quantized_rows(output_root):
    parameter_path = os.path.join(output_root, "work", "analysis_parameters.npz")
    with np.load(parameter_path, allow_pickle=False) as parameters:
        scale = np.asarray(parameters["consequence_scale"], dtype=np.float64)
    paths = sorted(glob.glob(os.path.join(output_root, "work", "quantized_shards", "*", "*.npz")))
    expected = len(config.TASKS) * len(config.CALIBRATION_EPISODES + config.TEST_EPISODES) * config.N_PHASES
    if len(paths) != expected:
        raise RuntimeError("expected %d quantized shards, found %d" % (expected, len(paths)))
    rows = []
    for path in paths:
        valid, evidence = validate_complete(path)
        if not valid:
            raise RuntimeError("invalid quantized shard %s: %s" % (path, evidence))
        with np.load(path, allow_pickle=False) as data:
            count = len(data["methods"])
            task_id = str(data["task_id"].item())
            split = str(data["split"].item())
            phase = str(data["phase"].item())
            episode_id = int(data["episode_id"].item())
            decoded_cont = data["decoded_actions"][:, :, config.CONTINUOUS_ACTION_INDICES].reshape(count, -1)
            original_cont = data["original_actions"][:, :, config.CONTINUOUS_ACTION_INDICES].reshape(count, -1)
            mask = np.asarray(data["original_mask"], dtype=bool)
            immediate_delta = (np.asarray(data["immediate"]) - np.asarray(data["original_immediate"])) / scale[None, :]
            settled_delta = (np.asarray(data["settled"]) - np.asarray(data["original_settled"])) / scale[None, :]
            immediate_delta[~mask] = 0.0
            settled_delta[~mask] = 0.0
            immediate_error = np.linalg.norm(immediate_delta, axis=1)
            settled_error = np.linalg.norm(settled_delta, axis=1)
            action_error = np.linalg.norm(decoded_cont - original_cont, axis=1) / math.sqrt(config.CHUNK_CONTINUOUS_DIM)
            progress_error = np.abs(
                np.asarray(data["settled_progress"], dtype=np.float64)
                - np.asarray(data["original_settled_progress"], dtype=np.float64)
            )
            for index in range(count):
                rows.append(
                    {
                        "task_id": task_id,
                        "episode_id": episode_id,
                        "split": split,
                        "phase": phase,
                        "method": str(data["methods"][index]),
                        "k": int(data["k"][index]),
                        "direction": int(data["direction"][index]),
                        "sign": int(data["sign"][index]),
                        "radius": float(data["radius"][index]),
                        "code_index": int(data["code_index"][index]),
                        "clipped_coordinates": int(data["clipped_coordinates"][index]),
                        "settled_effect_error": float(settled_error[index]),
                        "immediate_effect_error": float(immediate_error[index]),
                        "action_reconstruction_error": float(action_error[index]),
                        "contact_preserved": int(data["contact_mode"][index])
                        == int(data["original_contact_mode"][index]),
                        "progress_absolute_error": float(progress_error[index]),
                        "progress_preserved_005": bool(progress_error[index] <= 0.05),
                        "success_preserved": int(data["settled_success"][index])
                        == int(data["original_settled_success"][index]),
                        "source_shard": path,
                    }
                )
    return rows


def _group(rows, fields):
    output = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        output.setdefault(key, []).append(row)
    return output


def aggregate_rows(rows, group_fields):
    output = []
    for key, values in sorted(_group(rows, group_fields).items()):
        row = dict(zip(group_fields, key))
        k = int(values[0]["k"])
        method = values[0]["method"]
        if method == "phase_conditioned_kmeans" and "phase" not in group_fields:
            used = len(set((value["phase"], value["code_index"]) for value in values))
            capacity = k * len(set(value["phase"] for value in values))
        else:
            used = len(set(value["code_index"] for value in values))
            capacity = k
        row.update(
            {
                "n": len(values),
                "settled_effect_error_mean": float(np.mean([x["settled_effect_error"] for x in values])),
                "immediate_effect_error_mean": float(np.mean([x["immediate_effect_error"] for x in values])),
                "contact_mode_preservation": float(np.mean([x["contact_preserved"] for x in values])),
                "task_progress_preservation_005": float(np.mean([x["progress_preserved_005"] for x in values])),
                "task_progress_mae": float(np.mean([x["progress_absolute_error"] for x in values])),
                "action_reconstruction_error_mean": float(
                    np.mean([x["action_reconstruction_error"] for x in values])
                ),
                "codebook_utilization": float(used) / float(max(capacity, 1)),
                "dead_code_ratio": 1.0 - float(used) / float(max(capacity, 1)),
                "clipped_coordinate_rate": float(
                    np.sum([x["clipped_coordinates"] for x in values])
                )
                / float(len(values) * config.CHUNK_CONTINUOUS_DIM),
            }
        )
        output.append(row)
    return output


def choose_baseline(rows):
    calibration = [row for row in rows if row["split"] == "calibration" and row["k"] == config.PRIMARY_K]
    means = {}
    for method in BASELINE_METHODS:
        values = [row["settled_effect_error"] for row in calibration if row["method"] == method]
        means[method] = float(np.mean(values))
    method = min(means, key=lambda name: (means[name], name))
    return method, means


def _episode_method_means(rows, methods):
    values = {}
    selected = [row for row in rows if row["split"] == "test" and row["k"] == config.PRIMARY_K]
    for (task_id, episode_id, method), group in _group(selected, ("task_id", "episode_id", "method")).items():
        if method in methods:
            values[(task_id, int(episode_id), method)] = float(
                np.mean([row["settled_effect_error"] for row in group])
            )
    return values


def bootstrap_comparison(rows, baseline, replicates=config.BOOTSTRAP_REPLICATES):
    methods = (baseline, "caaa_v2", "permuted_j", "random_spd")
    episode = _episode_method_means(rows, methods)
    tasks = [task["task_id"] for task in config.TASKS]
    episodes_by_task = {
        task: sorted({episode_id for (name, episode_id, method) in episode if name == task})
        for task in tasks
    }
    for task, ids in episodes_by_task.items():
        if ids != list(config.TEST_EPISODES):
            raise RuntimeError("incomplete test episodes for %s: %r" % (task, ids))
    rng = np.random.RandomState(config.GLOBAL_SEED + 707)

    def statistic(task_subset, draws):
        base_values, caaa_values = [], []
        for task in task_subset:
            for episode_id in draws[task]:
                base_values.append(episode[(task, int(episode_id), baseline)])
                caaa_values.append(episode[(task, int(episode_id), "caaa_v2")])
        base = float(np.mean(base_values))
        caaa = float(np.mean(caaa_values))
        return (base - caaa) / max(base, 1e-12), base - caaa, base, caaa

    identity_draws = dict((task, episodes_by_task[task]) for task in tasks)
    point = statistic(tasks, identity_draws)
    pooled = np.empty((int(replicates), 4), dtype=np.float64)
    per_task = dict((task, np.empty((int(replicates), 4), dtype=np.float64)) for task in tasks)
    for index in range(int(replicates)):
        draws = {
            task: rng.choice(episodes_by_task[task], size=len(episodes_by_task[task]), replace=True).tolist()
            for task in tasks
        }
        pooled[index] = statistic(tasks, draws)
        for task in tasks:
            per_task[task][index] = statistic([task], draws)

    def summarize(point_values, boot):
        names = ("relative_improvement", "absolute_improvement", "baseline_error", "caaa_error")
        return {
            name: {
                "estimate": float(point_values[i]),
                "ci95": [float(np.percentile(boot[:, i], 2.5)), float(np.percentile(boot[:, i], 97.5))],
            }
            for i, name in enumerate(names)
        }

    result = {
        "created_utc": utc_now(),
        "cluster_unit": "episode",
        "resampling": "within-task paired episode clusters with replacement",
        "replicates": int(replicates),
        "primary_k": config.PRIMARY_K,
        "frozen_calibration_baseline": baseline,
        "pooled": summarize(point, pooled),
        "per_task": {},
    }
    for task in tasks:
        task_point = statistic([task], identity_draws)
        result["per_task"][task] = summarize(task_point, per_task[task])
    return result


def _mean_error(rows, method, split="test", k=config.PRIMARY_K, task_id=None):
    values = [
        row["settled_effect_error"]
        for row in rows
        if row["method"] == method
        and row["split"] == split
        and row["k"] == k
        and (task_id is None or row["task_id"] == task_id)
    ]
    return float(np.mean(values))


def mechanism_controls(rows, baseline, bootstrap):
    baseline_error = _mean_error(rows, baseline)
    caaa_error = _mean_error(rows, "caaa_v2")
    gain = baseline_error - caaa_error
    output = []
    for scope in ["pooled"] + [task["task_id"] for task in config.TASKS]:
        task_id = None if scope == "pooled" else scope
        base = _mean_error(rows, baseline, task_id=task_id)
        caaa = _mean_error(rows, "caaa_v2", task_id=task_id)
        local_gain = base - caaa
        for method in ("caaa_v2", "permuted_j", "random_spd"):
            error = _mean_error(rows, method, task_id=task_id)
            control_gain = base - error
            output.append(
                {
                    "scope": scope,
                    "method": method,
                    "primary_k": config.PRIMARY_K,
                    "frozen_baseline": baseline,
                    "baseline_error": base,
                    "method_error": error,
                    "absolute_gain": control_gain,
                    "gain_retention_vs_caaa": control_gain / local_gain if abs(local_gain) > 1e-12 else float("nan"),
                }
            )
    return output


def choose_disposition(rows, baseline, bootstrap, controls, jacobian_rows, task_results):
    pooled = bootstrap["pooled"]["relative_improvement"]
    per_task = {
        task: bootstrap["per_task"][task]["relative_improvement"]["estimate"]
        for task in bootstrap["per_task"]
    }
    pooled_controls = {row["method"]: row for row in controls if row["scope"] == "pooled"}
    permuted_retention = pooled_controls["permuted_j"]["gain_retention_vs_caaa"]
    random_retention = pooled_controls["random_spd"]["gain_retention_vs_caaa"]
    test_primary = [row for row in task_results if row["k"] == config.PRIMARY_K]
    action = {
        row["method"]: float(np.mean([x["action_reconstruction_error_mean"] for x in test_primary if x["method"] == row["method"]]))
        for row in test_primary
    }
    caaa_values = [row for row in test_primary if row["method"] == "caaa_v2"]
    caaa_dead = float(np.mean([row["dead_code_ratio"] for row in caaa_values]))
    sensitive_positive = sum(per_task[name] > 0 for name in ("plate_push", "stove_turn_on", "wine_rack"))
    local_test = [row for row in jacobian_rows if row["split"] == "test" and row["method"] == "caaa_v2"]
    stable = (
        float(np.nanmedian([row["local_r2"] for row in local_test])) >= 0.0
        and float(np.nanmedian([row["condition_number"] for row in local_test])) <= 1e8
    )
    go = (
        pooled["estimate"] >= 0.10
        and pooled["ci95"][0] > 0.0
        and sensitive_positive >= 2
        and per_task["bowl_on_plate"] >= -0.05
        and permuted_retention <= 0.25
        and random_retention < 0.75
        and action["caaa_v2"] <= 1.10 * action[baseline]
        and caaa_dead < 0.20
        and stable
    )
    if go:
        return "GO_TO_SMALL_BC"
    kmeans_error = min(_mean_error(rows, "global_kmeans"), _mean_error(rows, "phase_conditioned_kmeans"))
    reject = (
        pooled["estimate"] < 0.05
        or kmeans_error < _mean_error(rows, "caaa_v2")
        or permuted_retention > 0.50
        or random_retention >= 0.75
        or not stable
    )
    return "REJECT_CORE_HYPOTHESIS" if reject else "REVISE_ALPHABET"


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jacobian_rows(output_root):
    rows = []
    with open(os.path.join(output_root, "work", "jacobian_metrics.jsonl"), "r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _fmt(value):
    if isinstance(value, (float, np.floating)):
        return "%.5g" % float(value)
    return str(value)


def write_report(output_root, rows, task_results, phase_results, bootstrap, controls, disposition, jacobian_rows):
    environment = _load_json(os.path.join(output_root, "environment_lock.json"))
    replay = _load_json(os.path.join(output_root, "branch_replay_validation.json"))
    selection = _load_json(os.path.join(output_root, "work", "model_selection.json"))
    baseline = bootstrap["frozen_calibration_baseline"]
    unique_j = [row for row in jacobian_rows if row["method"] == "caaa_v2"]
    locality = []
    for (task, phase), values in sorted(_group(unique_j, ("task_id", "phase")).items()):
        locality.append(
            [
                task,
                phase,
                _fmt(np.nanmedian([x["local_r2"] for x in values])),
                _fmt(np.nanmedian([x["local_normalized_rmse"] for x in values])),
                _fmt(np.nanmedian([x["effective_rank"] for x in values])),
                _fmt(np.nanmedian([x["condition_number"] for x in values])),
                _fmt(np.nanmedian([x["metric_to_consequence_spearman"] for x in values])),
            ]
        )
    primary = [row for row in task_results if row["k"] == config.PRIMARY_K]
    outcome = []
    for (task, method), values in sorted(_group(primary, ("task_id", "method")).items()):
        value = values[0]
        if method not in (baseline, "old_diagonal_sensitivity", "permuted_j", "random_spd", "caaa_v2"):
            continue
        outcome.append(
            [
                task,
                method,
                _fmt(value["settled_effect_error_mean"]),
                _fmt(value["immediate_effect_error_mean"]),
                _fmt(value["contact_mode_preservation"]),
                _fmt(value["task_progress_preservation_005"]),
                _fmt(value["action_reconstruction_error_mean"]),
                _fmt(value["codebook_utilization"]),
            ]
        )
    ci_rows = []
    for scope, value in [("pooled", bootstrap["pooled"])] + sorted(bootstrap["per_task"].items()):
        rel = value["relative_improvement"]
        ci_rows.append([scope, _fmt(rel["estimate"]), "[%s, %s]" % (_fmt(rel["ci95"][0]), _fmt(rel["ci95"][1]))])

    artifact_names = [
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
    ]
    hashes = []
    for name in artifact_names:
        path = os.path.join(output_root, name)
        digest = sha256_tree(path) if os.path.isdir(path) else sha256_file(path)
        hashes.append([name, digest])

    next_experiment = {
        "GO_TO_SMALL_BC": (
            "Run a separately approved, small behavior-cloning comparison on the same four LIBERO tasks, "
            "freezing K=64 CAAA-v2 versus the calibration-selected baseline; no such training was started here."
        ),
        "REVISE_ALPHABET": (
            "Revise only the alphabet mechanism using calibration diagnostics (especially unstable phases and dead "
            "codes), preregister the revision, then repeat this Stage 1 simulator audit before any policy training."
        ),
        "REJECT_CORE_HYPOTHESIS": (
            "Do not start policy training for CAAA-v2. Test a narrower diagnostic that separates local linear-model "
            "failure from state-dependent codebook alignment, using the same frozen replay snapshots."
        ),
    }[disposition]
    development_failure = (
        "The first local smoke initially failed for bowl_on_plate/e0/free_space: repeated A had "
        "final-state max |Δ|=0.0122775, immediate consequence max |Δ|=0.0119254, and settled max "
        "|Δ|=0.00229436. Cause: Panda gripper.current_action is an integrated hidden command omitted by "
        "MuJoCo's flattened state. Snapshotting gripper history plus solver/control auxiliaries reduced all "
        "formal A/A and A/B/A differences to zero."
    )
    failed_formal = replay.get("failed_tests", [])
    report = """# R13-P15-v2 Stage 1 Report — LIBERO CAAA-v2

## Executive result

**{disposition}**

The frozen calibration baseline was `{baseline}`. The pooled test relative improvement of CAAA-v2 was
{pooled_estimate}, with episode-clustered 95% CI {pooled_ci}. This is a mechanism-only oracle audit: no ACT,
Diffusion Policy, SmolVLA, π0.5, DINO-WM, behavior cloning, or other policy training was launched.

## Scope and frozen environment

This authorized LIBERO adaptation uses standard `libero_goal`, Panda `OSC_POSE`, 20 Hz, 7D normalized actions,
H=4, and an alphabet over the 24 pose coordinates while copying gripper commands unchanged. It freezes 16
successful official demonstrations per task with episode split 8/4/4 and four snapshots per episode.

- Project commit: `{project_commit}`
- Project tree SHA-256: `{project_hash}`
- LIBERO upstream commit: `{libero_commit}`
- LIBERO source tree SHA-256: `{libero_hash}`
- Python: `{python_version}`
- MuJoCo: `{mujoco}`; robosuite: `{robosuite}`; PyTorch: `{torch}`; CUDA build: `{cuda}`
- Formal demonstration SHA-256 values are recorded individually in `environment_lock.json`.

## Replay validation and all failures

Formal replay gate: **{replay_gate}** ({n_tests} tests, {n_failed} failures, tolerance {tolerance}).
The formal failed-test array contains: `{formal_failures}`.

Development incident retained for completeness: {development_failure}

## Frozen consequence model and calibration

The task-generic continuous schema has 46 masked dimensions: object/TCP/relative poses in continuous rotation-6D,
gripper width, articulation, task progress, three task-relevant contact-force channels, penetration and joint-limit
violation. Immediate effects are measured after H=4; settled effects add three zero-pose steps holding the final
gripper command. Train-only robust scales were used. Calibration-only selections were ridge={ridge}, singular
cutoff={cutoff}, metric regularization={metric_reg}, covariance regularization={cov_reg}, PCA rank={pca_rank}.

### Per-task and per-phase locality (median across episodes)

{locality_table}

## Realized held-out quantization results (K=64)

Errors below come from executing every decoded action from its identical restored simulator snapshot. Progress
preservation means settled progress differs by at most 0.05. Codebook utilization is measured on test assignments.

{outcome_table}

Full per-task results, including K=32/128 sensitivity, are in `results_by_task.csv`; per-phase results are in
`results_by_phase.csv`. All nine methods' metric-to-consequence Spearman correlations, local linearity, effective
rank and condition number are in `jacobian_metrics.parquet`.

## Episode-clustered confidence intervals

{ci_table}

Bootstrap uses 10,000 paired episode-cluster resamples within task. Calibration episodes selected the baseline;
test episodes were not used for model or baseline selection.

## Mechanism controls and disposition logic

The permuted-J and random-SPD realized controls are recorded in `mechanism_controls.csv`. The final gate also checks
improvement on contact-sensitive tasks, bowl-control degradation, action reconstruction, dead-code ratio, local
stability, and whether k-means or geometry-destroying controls reproduce the gain. Applying the preregistered gates
returns exactly:

**{disposition}**

## Next recommended experiment

{next_experiment}

## Artifact hashes

{hash_table}

FINAL_DISPOSITION: {disposition}
""".format(
        disposition=disposition,
        baseline=baseline,
        pooled_estimate=_fmt(bootstrap["pooled"]["relative_improvement"]["estimate"]),
        pooled_ci="[%s, %s]" % tuple(_fmt(x) for x in bootstrap["pooled"]["relative_improvement"]["ci95"]),
        project_commit=environment["project_git_commit"],
        project_hash=environment["project_source_tree_sha256"],
        libero_commit=environment["libero"]["upstream_commit"],
        libero_hash=environment["libero"]["source_tree_sha256"],
        python_version=environment["python"]["version"].splitlines()[0],
        mujoco=environment["packages"].get("mujoco"),
        robosuite=environment["packages"].get("robosuite"),
        torch=environment["torch"].get("version"),
        cuda=environment["torch"].get("cuda_build"),
        replay_gate=replay["gate"],
        n_tests=replay["n_tests"],
        n_failed=replay["n_failed"],
        tolerance=replay["tolerance"],
        formal_failures=json.dumps(failed_formal, ensure_ascii=False, sort_keys=True),
        development_failure=development_failure,
        ridge=selection["ridge"],
        cutoff=selection["singular_cutoff"],
        metric_reg=selection["metric_regularization"],
        cov_reg=selection["covariance_regularization"],
        pca_rank=selection["pca_rank"],
        locality_table=_markdown_table(
            ["task", "phase", "R²", "NRMSE", "eff. rank", "condition", "CAAA Spearman"], locality
        ),
        outcome_table=_markdown_table(
            ["task", "method", "settled err", "immediate err", "contact", "progress", "action err", "util."],
            outcome,
        ),
        ci_table=_markdown_table(["scope", "relative improvement", "95% CI"], ci_rows),
        next_experiment=next_experiment,
        hash_table=_markdown_table(["artifact", "SHA-256"], hashes),
    )
    atomic_text(os.path.join(output_root, "STAGE1_REPORT.md"), report)


def finalize(output_root):
    rows = load_quantized_rows(output_root)
    raw_path = os.path.join(output_root, "work", "quantization_results.jsonl")
    atomic_text(raw_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    test = [row for row in rows if row["split"] == "test"]
    task_results = aggregate_rows(test, ("task_id", "method", "k"))
    phase_results = aggregate_rows(test, ("task_id", "phase", "method", "k"))
    _write_csv(os.path.join(output_root, "results_by_task.csv"), task_results)
    _write_csv(os.path.join(output_root, "results_by_phase.csv"), phase_results)
    baseline, calibration_means = choose_baseline(rows)
    bootstrap = bootstrap_comparison(rows, baseline)
    bootstrap["calibration_method_errors"] = calibration_means
    atomic_json(os.path.join(output_root, "bootstrap_results.json"), bootstrap)
    controls = mechanism_controls(rows, baseline, bootstrap)
    _write_csv(os.path.join(output_root, "mechanism_controls.csv"), controls)
    jacobian_rows = _load_jacobian_rows(output_root)
    disposition = choose_disposition(rows, baseline, bootstrap, controls, jacobian_rows, task_results)
    atomic_text(os.path.join(output_root, "work", "FINAL_DISPOSITION.txt"), disposition + "\n")
    write_report(
        output_root,
        rows,
        task_results,
        phase_results,
        bootstrap,
        controls,
        disposition,
        jacobian_rows,
    )
    result = {
        "created_utc": utc_now(),
        "quantized_rows": len(rows),
        "frozen_calibration_baseline": baseline,
        "disposition": disposition,
        "status": "STAGE1_COMPLETE",
    }
    atomic_json(os.path.join(output_root, "work", "finalize_manifest.json"), result)
    return result
