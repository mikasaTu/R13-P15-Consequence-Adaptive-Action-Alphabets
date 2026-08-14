"""Plain Markdown reporting and release checks for the completed Stage 3 audit."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys

import numpy as np

from .pipeline import utc_now
from .stage3_config import GATES, PRIMARY_K, TASK_IDS
from .storage import atomic_json, atomic_text, sha256_file


REQUIRED_ARTIFACTS = (
    "PREREGISTRATION.md",
    "INPUT_BINDING.json",
    "episode_split.json",
    "support_codebooks.npz",
    "action_bank_binding.json",
    "model_definitions.json",
    "training_pairs.parquet",
    "predictor_metrics.csv",
    "retrieval_metrics.csv",
    "development_quantization.csv",
    "mechanism_controls.csv",
    "development_gate.json",
    "confirmation_quantization.csv",
    "bootstrap_results.json",
    "STAGE3_REPORT.md",
)


EXPECTED_METHODS = {
    "B1_centered_covariance",
    "B2_current_contact_kmeans",
    "B2_PRIV_hard_phase_kmeans",
    "B3_dynamic_action_medoids",
    "B4_state_action_vq",
    "B5_local_knn_consequence",
    "C0_stage2_ncea_reproduction",
    "C1_NC_VECTOR",
    "C2_NC_TEMPORAL_VECTOR",
    "C3_NC_BIENCODER",
    "C4_NC_PAIR_RANKER",
    "C5_NCER_AA",
    "C6_SOFT_MIXTURE_NCER_AA",
    "O_FULL_true_effect_full_bank",
    "O_K64_true_effect_atlas",
    "no_nominal_action",
    "nominal_action_shuffled_within_task",
    "state_shuffled_within_task",
    "joint_state_nominal_shuffled_within_task",
    "history_shuffled",
    "consequence_labels_shuffled",
    "soft_routing_labels_shuffled",
    "action_only_pair_ranker",
    "candidate_order_permutation",
}


def _json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _csv(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value, digits=5):
    value = _number(value)
    if not math.isfinite(value):
        return "NA"
    return ("%%.%dg" % int(digits)) % value


def _pct(value, digits=3):
    value = _number(value)
    return "NA" if not math.isfinite(value) else ("%%.%df%%%%" % digits) % (100.0 * value)


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _summary(
    rows,
    method,
    level="pooled",
    task="ALL",
    phase="ALL",
    split=None,
    family="ALL",
):
    for row in rows:
        if (
            row.get("method") == method
            and row.get("level") == level
            and row.get("task_id", "ALL") == task
            and row.get("phase", "ALL") == phase
            and row.get("direction_family_id", "ALL") == family
            and (split is None or row.get("split") == split)
        ):
            return row
    raise KeyError((method, level, task, phase, split, family))


def _relative_improvement(reference, method):
    reference = _number(reference)
    method = _number(method)
    return (reference - method) / reference if reference > 0 else float("nan")


def _command(args, env=None):
    try:
        return subprocess.check_output(args, env=env, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as error:
        return "ERROR:%s" % error


def capture_execution_environment(project_root, output_root):
    """Capture the two local runtimes used; this does not submit any job."""
    analysis = {
        "python_executable": sys.executable,
        "python_version": sys.version,
    }
    try:
        import pandas
        import pyarrow
        import scipy
        import torch
        import zarr

        analysis.update(
            {
                "numpy": np.__version__,
                "pandas": pandas.__version__,
                "pyarrow": pyarrow.__version__,
                "scipy": scipy.__version__,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "zarr": zarr.__version__,
            }
        )
    except Exception as error:
        analysis["capture_error"] = repr(error)
    sim_python = "/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero-original/bin/python"
    sim_code = (
        "import json,sys,numpy; import mujoco_py,robosuite; "
        "print(json.dumps({'python_executable':sys.executable,'python_version':sys.version,"
        "'numpy':numpy.__version__,'mujoco_py':getattr(mujoco_py,'__version__','unknown'),"
        "'robosuite':getattr(robosuite,'__version__','unknown')}))"
    )
    sim_env = dict(os.environ)
    sim_env.update({"MUJOCO_GL": "glx", "CUDA_VISIBLE_DEVICES": ""})
    sim_raw = _command([sim_python, "-c", sim_code], env=sim_env)
    try:
        simulation = json.loads(sim_raw.splitlines()[-1])
    except Exception:
        simulation = {"python_executable": sim_python, "capture_output": sim_raw}
    binding = _json(os.path.join(output_root, "INPUT_BINDING.json"))
    registry = _json(os.path.join(output_root, "trained_model_registry.json"))
    cleanliness_scope = (
        "README.md",
        "TESTING.md",
        "caaa_libero",
        "config",
        "tests",
    )
    dirty_paths = _command(
        [
            "git",
            "-C",
            project_root,
            "diff",
            "--name-only",
            "HEAD",
            "--",
            *cleanliness_scope,
        ]
    )
    payload = {
        "created_utc": utc_now(),
        "repository_code_commit": _command(["git", "-C", project_root, "rev-parse", "HEAD"]),
        "repository_dirty": bool(dirty_paths),
        "repository_dirty_paths": dirty_paths.splitlines() if dirty_paths else [],
        "repository_cleanliness_scope": list(cleanliness_scope),
        "libero_commit": binding["libero"]["upstream_commit"],
        "libero_tree_sha256": binding["libero"]["observed_tree_sha256"],
        "analysis_runtime": analysis,
        "simulation_runtime": simulation,
        "gpu_training": {
            "device": registry["device"],
            "visible_gpu_count": registry["visible_gpu_count"],
            "gpu_name": registry["gpu_name"],
            "torch_version": registry["torch_version"],
            "cuda_version": registry["cuda_version"],
        },
        "local_gpu_limit_respected": registry["visible_gpu_count"] <= 1,
        "pai_jobs_submitted": 0,
        "policy_training_performed": False,
    }
    path = os.path.join(output_root, "execution_environment.json")
    atomic_json(path, payload)
    return payload


def validate_pair_invariants(output_root):
    """Empirically confirm the architectural symmetry/self-distance contract."""
    import torch

    from .stage3_analysis import load_trained_models

    device = torch.device("cpu")
    _, scalers, models = load_trained_models(output_root, device)
    generator = torch.Generator(device="cpu").manual_seed(13150377)
    context = torch.randn(
        32, len(scalers["context_center"]), generator=generator
    )
    left = torch.randn(32, 24, generator=generator)
    right = torch.randn(32, 24, generator=generator)
    families = (
        "C3_NC_BIENCODER",
        "C4_NC_PAIR_RANKER",
        "C6_SOFT_MIXTURE_NCER_AA",
        "no_nominal_action",
        "nominal_action_shuffled_within_task",
        "state_shuffled_within_task",
        "joint_state_nominal_shuffled_within_task",
        "history_shuffled",
        "consequence_labels_shuffled",
        "action_only_pair_ranker",
        "soft_routing_labels_shuffled",
    )
    rows = []
    with torch.no_grad():
        for family in families:
            for member, model in enumerate(models[family]):
                forward = model(context, left, right)
                reverse = model(context, right, left)
                self_distance = model(context, left, left)
                rows.append(
                    {
                        "family": family,
                        "member": member,
                        "symmetry_max_abs": float(
                            torch.max(torch.abs(forward - reverse)).item()
                        ),
                        "self_distance_max_abs": float(
                            torch.max(torch.abs(self_distance)).item()
                        ),
                    }
                )
    passed = all(
        row["symmetry_max_abs"] == 0.0
        and row["self_distance_max_abs"] == 0.0
        for row in rows
    )
    payload = {
        "created_utc": utc_now(),
        "seed": 13150377,
        "samples_per_model": len(context),
        "passed": passed,
        "rows": rows,
    }
    atomic_json(os.path.join(output_root, "pair_invariant_validation.json"), payload)
    if not passed:
        raise RuntimeError("pair symmetry/self-distance validation failed")
    return payload


def _mechanism_audit(output_root, gate, bootstrap, retrieval_rows):
    split = "development"
    realized = gate["development_realized_summary"]
    c5_r = _summary(retrieval_rows, "C5_NCER_AA", split=split)
    c5_q = _summary(realized, "C5_NCER_AA")
    c3_r = _summary(retrieval_rows, "C3_NC_BIENCODER", split=split)
    c3_q = _summary(realized, "C3_NC_BIENCODER")
    baseline_q = _summary(realized, "B2_current_contact_kmeans")

    def comparison(label, control, code_path):
        control_r = _summary(retrieval_rows, control, split=split)
        control_q = _summary(realized, control)
        return (
            label,
            control,
            _pct(
                _relative_improvement(
                    control_r["oracle_regret"], c5_r["oracle_regret"]
                )
            ),
            _pct(
                _relative_improvement(
                    control_q["balanced_task_effect_error"],
                    c5_q["balanced_task_effect_error"],
                )
            ),
            code_path,
        )

    rows = [
        comparison("Nominal chunk", "no_nominal_action", "raw_context nominal slice → C4 symmetric scorer"),
        comparison("Current state", "state_shuffled_within_task", "state+mask+contact bundle permutation"),
        comparison("Short history", "history_shuffled", "two deltas/actions and masks permutation"),
        comparison("Correct labels", "consequence_labels_shuffled", "true-distance row permutation"),
        comparison("State + nominal jointly", "joint_state_nominal_shuffled_within_task", "joint within-task bundle permutation"),
        comparison("All context", "action_only_pair_ranker", "constant context; target/candidate only"),
    ]
    best_vector = min(
        ("C1_NC_VECTOR", "C2_NC_TEMPORAL_VECTOR"),
        key=lambda name: _number(_summary(retrieval_rows, name, split=split)["oracle_regret"]),
    )
    c4 = _summary(retrieval_rows, "C4_NC_PAIR_RANKER", split=split)
    vector = _summary(retrieval_rows, best_vector, split=split)
    c6_r = _summary(retrieval_rows, "C6_SOFT_MIXTURE_NCER_AA", split=split)
    c6_q = _summary(realized, "C6_SOFT_MIXTURE_NCER_AA")
    c4_q = _summary(realized, "C4_NC_PAIR_RANKER")
    nominal_shuffle_r = _summary(
        retrieval_rows,
        "nominal_action_shuffled_within_task",
        split=split,
    )
    state_shuffle_r = _summary(
        retrieval_rows,
        "state_shuffled_within_task",
        split=split,
    )
    history_shuffle_r = _summary(
        retrieval_rows,
        "history_shuffled",
        split=split,
    )
    label_shuffle_r = _summary(
        retrieval_rows,
        "consequence_labels_shuffled",
        split=split,
    )
    family_names = {
        "0": "smooth DCT (train overrepresented)",
        "1": "suffix-localized contact",
        "2": "low-rank temporal-action",
    }
    family_rows = []
    for family, label in family_names.items():
        c3_family = _summary(
            retrieval_rows,
            "C3_NC_BIENCODER",
            level="direction_family",
            split=split,
            family=family,
        )
        c4_family = _summary(
            retrieval_rows,
            "C4_NC_PAIR_RANKER",
            level="direction_family",
            split=split,
            family=family,
        )
        family_rows.append(
            (
                label,
                _fmt(c3_family["oracle_regret"]),
                _fmt(c4_family["oracle_regret"]),
                _fmt(c3_family["ndcg_at_16"]),
                _fmt(c4_family["ndcg_at_16"]),
            )
        )
    audit = f"""# Stage 3 mechanism reverse audit

This is a code-to-result localization audit of the frozen NCER-AA implementation. It does not propose a new idea.

## Executed path

`stage3_collection._context_arrays` binds the observable context to the exact support branch initial state. `stage3_data.raw_context` concatenates current observable state/mask, two history deltas and masks, two previous actions and masks, current contact, nominal chunk and task ID. `stage3_models.create_biencoder` embeds the 256 residual bank; deterministic ID-stable FPS selects K=64. `create_pair_ranker` computes

`||dt-dc||₂ × softplus(MLP(h, mean(dt,dc), |dt-dc|, dt·dc))`,

which gives exact symmetry and exact zero self-distance by construction. `stage3_analysis.evaluate_records` carries frozen bank IDs through FPS/reranking and looks up the actually simulated candidate consequence; predictor scores never substitute for simulator outcomes.

## Controlled localization on development

Positive percentages mean C5 is better (lower error) than the named control.

{_table(["Information/mechanism", "Frozen comparator", "Oracle-regret improvement", "Realized-effect improvement", "Implementation isolation"], rows)}

Direct pair ranking versus the best vector regressor ({best_vector}) changes oracle regret from {_fmt(vector['oracle_regret'])} to {_fmt(c4['oracle_regret'])} ({_pct(_relative_improvement(vector['oracle_regret'], c4['oracle_regret']))}). At realized execution, full-bank C4 error is {_fmt(c4_q['balanced_task_effect_error'])}; K=64 C5 error is {_fmt(c5_q['balanced_task_effect_error'])}. The difference between C4 and C5 localizes loss introduced by C3/FPS alphabet compression, not the pair scorer.

The dominant signed transition is C3→C4, not vector→ranker: C3 regret {_fmt(c3_r['oracle_regret'])}, Spearman {_fmt(c3_r['candidate_distance_spearman'])}, NDCG@16 {_fmt(c3_r['ndcg_at_16'])}, and realized error {_fmt(c3_q['balanced_task_effect_error'])}; C4 changes these to {_fmt(c4['oracle_regret'])}, {_fmt(c4['candidate_distance_spearman'])}, {_fmt(c4['ndcg_at_16'])}, and {_fmt(c4_q['balanced_task_effect_error'])}. C3 alone improves realized error over B2 from {_fmt(baseline_q['balanced_task_effect_error'])} to {_fmt(c3_q['balanced_task_effect_error'])}, but the frozen C5 implementation discards C3 target-candidate distance after using C3 only to choose the 64-code atlas. C4 then reranks that atlas and reverses the C3 gain. Compression adds a smaller second loss: C4 {_fmt(c4_q['balanced_task_effect_error'])} → C5 {_fmt(c5_q['balanced_task_effect_error'])}, while normalized utilization collapses from {_fmt(c3_q['normalized_code_utilization'])} (C3) to {_fmt(c5_q['normalized_code_utilization'])} (C5).

{_table(["Fresh support family", "C3 regret", "C4 regret", "C3 NDCG@16", "C4 NDCG@16"], family_rows)}

Training reuse supplied 48/24/24 branches per state for smooth/suffix/low-rank families, while fresh development is 32/32/32. The largest C3→C4 regret increase is on suffix-localized contact support. Because C3 sees the same training distribution and remains strong across all three families, imbalance can amplify but cannot by itself explain the cross-encoder failure.

The soft mixture changes K=64 realized error from {_fmt(c5_q['balanced_task_effect_error'])} (C5) to {_fmt(c6_q['balanced_task_effect_error'])} (C6), and oracle regret from {_fmt(c5_r['oracle_regret'])} to {_fmt(c6_r['oracle_regret'])}. Its router sees only permitted observable context and uses no hard demonstration phase.

The pair scorer does not show mechanism-specific context dependence: C5 regret is {_fmt(c5_r['oracle_regret'])}; nominal-shuffled={_fmt(nominal_shuffle_r['oracle_regret'])}, state-shuffled={_fmt(state_shuffle_r['oracle_regret'])}, history-shuffled={_fmt(history_shuffle_r['oracle_regret'])}, and label-shuffled={_fmt(label_shuffle_r['oracle_regret'])}. Several destructive controls are equal or better, so the absence of a positive Gate-B denominator—not a numerical division bug—is why gain retention is frozen to 1e9.

## What the code can and cannot establish

- Nominal conditioning can help because the same residual has different consequences under different base chunks; the nominal slice enters every proposed scorer before target/candidate comparison. A loss under no-nominal/shuffle controls supports this mechanism only if it is larger than run variance.
- Pair/listwise training optimizes candidate ordering directly, whereas C1/C2 first reconstruct a masked multi-group consequence vector and only then induce distances. A ranking gain therefore localizes avoidance of vector-reconstruction error accumulation.
- Here, that expected ranking advantage does not materialize consistently: C4 has slightly lower regret than the best vector regressor but worse NDCG/Recall, and it is much worse than the jointly trained C3 metric. All four calibration objective tuples produced poor C3-atlas+C4 regret, so the failure is not caused by one post-development objective choice.
- History can matter near contact because two observable deltas and previous actions distinguish approach, sustained contact and departure without using a future phase label. The history control permutes values and masks as one bundle.
- C3/FPS can lower performance when its learned embedding spreads candidates along directions irrelevant to the target-specific C4 scorer. C4 sees all 256 candidates; C5 sees only the 64 retained by C3, so C4→C5 degradation is the clean compression bottleneck.
- C6 can help only when the observable router separates regimes that need different pairwise scalings. If C6 or its shuffled-route control matches C5, the extra experts did not provide mechanism-specific routing.
- The controls have one member while primary ensembles have three, as preregistered. Large differences are informative, but small differences cannot be attributed solely to the ablated input because ensemble size is a remaining confound.
- C0 reproduces the prior hard-phase implementation and is used as a conservative comparator; it is not permitted as the proposed deployable method.

## Confirmation-boundary consequence

The holdout label is `{bootstrap['evidence_label']}`. The pre-result replay incident means it is not untouched confirmation and cannot unlock BC, even if its counterfactual statistics pass.
"""
    atomic_text(os.path.join(output_root, "MECHANISM_REVERSE_AUDIT.md"), audit)
    return audit


def write_stage3_report(project_root, output_root):
    binding = _json(os.path.join(output_root, "INPUT_BINDING.json"))
    split = _json(os.path.join(output_root, "episode_split.json"))
    selection = _json(os.path.join(output_root, "calibration_selection.json"))
    gate = _json(os.path.join(output_root, "development_gate.json"))
    bootstrap = _json(os.path.join(output_root, "bootstrap_results.json"))
    registry = _json(os.path.join(output_root, "trained_model_registry.json"))
    environment = capture_execution_environment(project_root, output_root)
    predictor = _csv(os.path.join(output_root, "predictor_metrics.csv"))
    retrieval = _csv(os.path.join(output_root, "retrieval_metrics.csv"))
    k_sensitivity = _csv(os.path.join(output_root, "k_sensitivity.csv"))
    mechanism = _mechanism_audit(output_root, gate, bootstrap, retrieval)
    del mechanism

    gates = gate["gates"]
    final = bootstrap["final_disposition"]
    dev = gate["development_realized_summary"]
    hold = bootstrap["holdout_realized_summary"]
    baseline = selection["strongest_deployable_baseline"]
    rank_baseline = selection["strongest_gate_b_learned_or_action_baseline"]
    ranker = selection["selected_learned_ranker"]
    breakdown_specs = [("pooled", "ALL", "ALL", "ALL", "ALL")]
    breakdown_specs.extend(
        ("task", task, task, "ALL", "ALL") for task in TASK_IDS
    )
    breakdown_specs.extend(
        ("phase", phase, "ALL", phase, "ALL")
        for phase in ("free_space", "pre_contact", "contact_onset", "post_contact")
    )
    breakdown_specs.extend(
        ("direction_family", family, "ALL", "ALL", family)
        for family in ("0", "1", "2")
    )
    predictor_breakdown_rows = []
    for method in (
        "C0_stage2_ncea_reproduction",
        "C1_NC_VECTOR",
        "C2_NC_TEMPORAL_VECTOR",
    ):
        for level, label, task, phase, family in breakdown_specs:
            row = _summary(
                predictor,
                method,
                level=level,
                task=task,
                phase=phase,
                family=family,
                split="development",
            )
            predictor_breakdown_rows.append(
                (
                    method,
                    level,
                    label,
                    _fmt(row["normalized_effect_rmse"]),
                    _fmt(row["balanced_prediction_error"]),
                    _fmt(row["contact_accuracy"]),
                )
            )
    retrieval_breakdown_rows = []
    for method in ("C3_NC_BIENCODER", "C4_NC_PAIR_RANKER", "C5_NCER_AA"):
        for level, label, task, phase, family in breakdown_specs:
            row = _summary(
                retrieval,
                method,
                level=level,
                task=task,
                phase=phase,
                family=family,
                split="development",
            )
            retrieval_breakdown_rows.append(
                (
                    method,
                    level,
                    label,
                    _fmt(row["oracle_regret"]),
                    _fmt(row["ndcg_at_16"]),
                    _fmt(row["oracle_neighbor_recall_at_8"]),
                )
            )
    task_rows = []
    for task in TASK_IDS:
        base = _summary(dev, baseline, level="task", task=task)
        c5 = _summary(dev, "C5_NCER_AA", level="task", task=task)
        oracle = _summary(dev, "O_K64_true_effect_atlas", level="task", task=task)
        task_rows.append(
            (task, _fmt(base["balanced_task_effect_error"]), _fmt(c5["balanced_task_effect_error"]), _pct(_relative_improvement(base["balanced_task_effect_error"], c5["balanced_task_effect_error"])), _fmt(oracle["balanced_task_effect_error"]))
        )
    phase_rows = []
    for phase in ("free_space", "pre_contact", "contact_onset", "post_contact"):
        base = _summary(dev, baseline, level="phase", phase=phase)
        c5 = _summary(dev, "C5_NCER_AA", level="phase", phase=phase)
        oracle = _summary(dev, "O_K64_true_effect_atlas", level="phase", phase=phase)
        phase_rows.append((phase, _fmt(base["balanced_task_effect_error"]), _fmt(c5["balanced_task_effect_error"]), _fmt(oracle["balanced_task_effect_error"]), _fmt(c5["contact_mode_preserved"])))
    retrieval_rows = []
    for method in (rank_baseline, "C1_NC_VECTOR", "C2_NC_TEMPORAL_VECTOR", "C3_NC_BIENCODER", "C4_NC_PAIR_RANKER", "C5_NCER_AA", "C6_SOFT_MIXTURE_NCER_AA"):
        row = _summary(retrieval, method, split="development")
        retrieval_rows.append((method, _fmt(row["oracle_regret"]), _fmt(row["candidate_distance_spearman"]), _fmt(row["kendall_tau"]), _fmt(row["ndcg_at_16"]), _fmt(row["oracle_neighbor_recall_at_8"]), _fmt(row["inference_latency_ms"])))
    control_rows = []
    controls = _csv(os.path.join(output_root, "mechanism_controls.csv"))
    for method in sorted({row["method"] for row in controls if row["split"] == "development"}):
        row = _summary(controls, method, split="development")
        control_rows.append((method, _fmt(row["balanced_task_effect_error"]), _fmt(row.get("oracle_regret")), _fmt(row.get("ndcg_at_16")), row.get("exact_selected_index_invariance", ""), row.get("index_mismatches", "")))
    bootstrap_rows = [
        ("pooled", _fmt(bootstrap["paired_episode_cluster_bootstrap"]["pooled"]["point"]), "[" + ", ".join(_fmt(x) for x in bootstrap["paired_episode_cluster_bootstrap"]["pooled"]["ci95"]) + "]")
    ]
    for task in TASK_IDS:
        row = bootstrap["paired_episode_cluster_bootstrap"]["by_task"][task]
        bootstrap_rows.append((task, _fmt(row["point"]), "[" + ", ".join(_fmt(x) for x in row["ci95"]) + "]"))
    dev_base = _summary(dev, baseline)
    dev_c5 = _summary(dev, "C5_NCER_AA")
    hold_base = _summary(hold, baseline)
    hold_c5 = _summary(hold, "C5_NCER_AA")
    deployable_rows = []
    for method in (
        baseline,
        "C3_NC_BIENCODER",
        "C4_NC_PAIR_RANKER",
        "C5_NCER_AA",
        "C6_SOFT_MIXTURE_NCER_AA",
    ):
        row = _summary(dev, method)
        deployable_rows.append(
            (
                method,
                _fmt(row["balanced_task_effect_error"]),
                _fmt(row["action_reconstruction_rmse"]),
                _fmt(row["contact_mode_preserved"]),
                _fmt(row["normalized_code_utilization"]),
                _fmt(row["clipped"]),
            )
        )
    privileged_rows = []
    for method in ("B2_PRIV_hard_phase_kmeans", "C0_stage2_ncea_reproduction"):
        row = _summary(dev, method)
        privileged_rows.append(
            (
                method,
                _fmt(row["balanced_task_effect_error"]),
                _fmt(row["action_reconstruction_rmse"]),
                _fmt(row["contact_mode_preserved"]),
                _fmt(row["normalized_code_utilization"]),
            )
        )
    k_rows = []
    for row in sorted(
        (row for row in k_sensitivity if row["level"] == "pooled"),
        key=lambda row: (int(row["alphabet_k"]), row["method"]),
    ):
        k_rows.append(
            (
                row["alphabet_k"],
                row["method"],
                _fmt(row["balanced_task_effect_error"]),
                _fmt(row["action_reconstruction_rmse"]),
                _fmt(row["contact_mode_preserved"]),
                _fmt(row["normalized_code_utilization"]),
            )
        )
    gate_rows = []
    for name in ("A", "B", "C"):
        row = gates[name]
        gate_rows.append(
            (
                name,
                "PASS" if row["passed"] else "FAIL",
                GATES[name]["failure_disposition"],
            )
        )
    support = binding["support_separation"]
    max_cosine = max(support["maximum_cross_split_absolute_cosine_similarity"].values())
    answers = [
        ("1", "Nominal a0 materially improves prediction?", "YES" if gates["B"]["gain_retention"]["nominal_action_shuffled_within_task"] <= 0.5 else "NO", "Development nominal-shuffle gain retention=" + _fmt(gates["B"]["gain_retention"]["nominal_action_shuffled_within_task"])),
        ("2", "Pair/listwise ranking beats vector regression?", "YES" if (_number(_summary(retrieval, "C4_NC_PAIR_RANKER", split="development")["oracle_regret"]) < min(_number(_summary(retrieval, name, split="development")["oracle_regret"]) for name in ("C1_NC_VECTOR", "C2_NC_TEMPORAL_VECTOR")) and _number(_summary(retrieval, "C4_NC_PAIR_RANKER", split="development")["ndcg_at_16"]) > max(_number(_summary(retrieval, name, split="development")["ndcg_at_16"]) for name in ("C1_NC_VECTOR", "C2_NC_TEMPORAL_VECTOR"))) else "NO", "C4 regret is slightly lower, but NDCG/Recall are worse and C3 is decisively better."),
        ("3", "Short observable history necessary?", "YES" if _number(_summary(retrieval, "history_shuffled", split="development")["oracle_regret"]) > _number(_summary(retrieval, "C5_NCER_AA", split="development")["oracle_regret"]) else "NO", "History bundle includes values and availability masks."),
        ("4", "Soft mixture beats one global model?", "MARGINALLY YES, NOT SUFFICIENT" if _number(_summary(dev, "C6_SOFT_MIXTURE_NCER_AA")["balanced_task_effect_error"]) < _number(dev_c5["balanced_task_effect_error"]) else "NO", "C6 is slightly better than C5 but remains worse than B2 and C3."),
        ("5", "Learned retrieval recovers meaningful oracle fraction?", "YES" if gates["B"]["passed"] else "NO", "Frozen Gate B."),
        ("6", "Gain survives K=64 compression?", "YES" if gates["C"]["passed"] else "NO", "Frozen Gate C."),
        ("7", "State/nominal/label shuffles destroy gain?", "YES" if (gates["B"]["gain_retention"]["joint_state_nominal_shuffled_within_task"] <= 0.25 and gates["B"]["gain_retention"]["state_shuffled_within_task"] <= 0.50 and gates["B"]["gain_retention"]["nominal_action_shuffled_within_task"] <= 0.50 and gates["B"]["gain_retention"]["consequence_labels_shuffled"] <= 0.25) else "NO", json.dumps(gates["B"]["gain_retention"], sort_keys=True)),
        ("8", "Confirmed on episodes 40–49?", "NO", "Executed as " + bootstrap["evidence_label"] + "; strict untouched confirmation is false."),
        ("9", "Ready for small state-based BC?", "NO", "go_to_small_bc_available=false; final disposition=" + final),
    ]
    report = f"""# R13-P15 Stage 3 — NCER-AA experiment report

## Exact disposition

`{final}`

`GO_TO_SMALL_BC` is unavailable. Stage 1 remains `REJECT_CORE_HYPOTHESIS`, Stage 1.5 remains `REJECT_P15_FAMILY`, and Stage 2 remains `ORACLE_ONLY_NO_DEPLOYABLE_MODEL`. No policy or VLA was trained.

## Historical evidence (read-only)

{_table(["Stage", "Frozen disposition", "Published commit"], [
    ("Stage 1", binding['historical_evidence']['stage1_disposition'], binding['historical_evidence']['stage1_published_commit']),
    ("Stage 1.5", binding['historical_evidence']['stage1_5_disposition'], binding['historical_evidence']['stage1_5_result_commit']),
    ("Stage 2", binding['historical_evidence']['stage2_disposition'], binding['historical_evidence']['stage2_published_commit']),
])}

These artifacts are inputs only. Stage 3 neither rewrites nor relabels them.

## Stage 3 evidence integrity and scope

- LIBERO tasks: bowl_on_plate, plate_push, stove_turn_on, wine_rack; Panda OSC_POSE at 20 Hz; H=4; three settle steps.
- Episodes: historical 0–15, train 16–31, calibration 32–35, development 36–39, holdout 40–49. All demonstrations were successful; {split['snapshot_count']} four-phase snapshots were frozen.
- Fresh support split overlap is zero; worst cross-split absolute cosine is {_fmt(max_cosine)} (limit 0.90); target-bank exact matches are zero; clipping validity passed.
- The user-required holdout was fully executed after settings were frozen even if gates failed. It is labeled `{bootstrap['evidence_label']}`, not untouched confirmation.
- Pre-result incident `stage3-pre-result-confirmation-replay-001` executed one fixed confirmation support direction twice per snapshot for replay checking before gates. No method/result was computed then, but the literal untouched rule was broken; the incident is preserved in `PRE_RESULT_PROTOCOL_INCIDENT.json`.
- Local execution only: {environment['gpu_training']['visible_gpu_count']} visible training GPU ({environment['gpu_training']['gpu_name']}); PAI jobs=0; policy training=false.

## Exact bindings

{_table(["Component", "Commit/hash"], [
    ("Stage 3 input repository", binding['stage3_input_commit']),
    ("Published Stage 2", binding['historical_evidence']['stage2_published_commit']),
    ("LIBERO upstream", binding['libero']['upstream_commit']),
    ("LIBERO source tree SHA-256", binding['libero']['observed_tree_sha256']),
    ("Stage 2 M=256 bank SHA-256", binding['stage2_reuse']['action_bank_sha256']),
    ("Stage 3 code used for report", environment['repository_code_commit']),
])}

The simulator and analysis package versions are frozen in `execution_environment.json`; all 256 reused training states and all 544 final branch states are hash-checked in the validation JSON files.

## Reused training and calibration evidence

- Reused Stage 2 support: episodes 16–31 after exact simulator/hash verification. Missing candidate-bank outcomes for 16–23 were collected; 24–31 were reused.
- Training states: {registry['train_states']}; calibration states: {registry['calibration_states']}; training pairs: {registry['training_pairs']['rows']} rows.
- Calibration alone selected B5 neighbors={selection['b5_selected']['neighbors']}, bandwidth={selection['b5_selected']['bandwidth']}; strongest deployable comparator=`{baseline}`; Gate-B comparator=`{rank_baseline}`; selected learned ranker=`{ranker}`; ranking objective index={registry['selected_ranking_objective_index']}.
- Candidate-order invariance passed on calibration: {selection['candidate_permutation_all_passed']}.

## Development gates (episodes 36–39)

{_table(["Gate", "Result", "Failure disposition if first failure"], gate_rows)}

- Gate A: oracle gain={_pct(gates['A']['pooled_relative_gain'])}, tasks improved={gates['A']['tasks_improved']}/4, contact tasks={gates['A']['contact_sensitive_tasks_improved']}/3.
- Gate B: regret gain={_pct(gates['B']['oracle_regret_relative_gain'])}, NDCG@16 gain={_fmt(gates['B']['ndcg_at_16_absolute_gain'])}, Recall@8={_fmt(gates['B']['recall_at_8'])}, tasks={gates['B']['tasks_improved']}/4, contact tasks={gates['B']['contact_sensitive_tasks_improved']}/3, exact permutation={gates['B']['candidate_permutation_exact']}.
- Gate C: realized gain={_pct(gates['C']['realized_relative_gain'])}, oracle gap closed={_pct(gates['C']['oracle_gap_fraction_closed'])}, utilization={_fmt(gates['C']['normalized_utilization'])}, clipping={_fmt(gates['C']['clipping_rate'])}, action-RMSE degradation={_pct(gates['C']['action_rmse_degradation'])}, contact drop={_fmt(gates['C']['contact_preservation_drop_points'])}.

### Oracle-only result

Gate A is a simulator-outcome upper bound, not a deployable method. The K=64 true-effect atlas improves all four tasks and all three contact-sensitive tasks, but it consumes candidate consequences unavailable at deployment.

## K=64 alphabet and deployable results

{_table(["Method", "Effect error", "Action RMSE", "Contact preserved", "Normalized utilization", "Clipping"], deployable_rows)}

### Realized K=64 effect by task

{_table(["Task", baseline, "C5", "C5 improvement", "True-effect K64 oracle"], task_rows)}

### Realized K=64 effect by phase

{_table(["Phase", baseline, "C5", "Oracle K64", "C5 contact preservation"], phase_rows)}

Pooled development realized error is {_fmt(dev_base['balanced_task_effect_error'])} for `{baseline}` and {_fmt(dev_c5['balanced_task_effect_error'])} for C5. C5 action RMSE={_fmt(dev_c5['action_reconstruction_rmse'])}, contact preservation={_fmt(dev_c5['contact_mode_preserved'])}, normalized utilization={_fmt(dev_c5['normalized_code_utilization'])}, code perplexity={_fmt(dev_c5['code_perplexity'])}, clipping={_fmt(dev_c5['clipped'])}.

## Privileged diagnostic upper bounds

{_table(["Diagnostic", "Effect error", "Action RMSE", "Contact preserved", "Normalized utilization"], privileged_rows)}

`B2_PRIV` and C0 consume the frozen demonstration hard-phase construction and are diagnostics only. Neither is the deployable NCER-AA result.

## Learned prediction and retrieval

{_table(["Method", "Oracle regret", "Spearman", "Kendall", "NDCG@16", "Recall@8", "ms/target"], retrieval_rows)}

`predictor_metrics.csv` additionally reports balanced vector error/contact accuracy per task, phase and support family. `retrieval_metrics.csv` reports pairwise accuracy, Spearman, Kendall tau, NDCG@16, Recall@1/8, regret and latency for every learned method and direction family.

### Vector predictor breakdown

{_table(["Method", "Level", "Slice", "Normalized RMSE", "Balanced prediction error", "Contact accuracy"], predictor_breakdown_rows)}

### Learned retrieval breakdown

{_table(["Method", "Level", "Slice", "Oracle regret", "NDCG@16", "Recall@8"], retrieval_breakdown_rows)}

## Mechanism controls

{_table(["Control", "Realized effect", "Oracle regret", "NDCG@16", "Order invariant", "Index mismatches"], control_rows)}

Symmetry error and self-distance error are exactly zero by architecture and unit test. Candidate permutation carries immutable bank IDs through FPS and reranking. The full code-to-result explanation, including improvements, degradations and confounds, is in `MECHANISM_REVERSE_AUDIT.md`.

## Holdout episodes 40–49 (not untouched confirmation)

Pooled realized error: `{baseline}`={_fmt(hold_base['balanced_task_effect_error'])}; C5={_fmt(hold_c5['balanced_task_effect_error'])}; relative gain={_pct(_relative_improvement(hold_base['balanced_task_effect_error'], hold_c5['balanced_task_effect_error']))}. The paired difference is baseline minus C5.

{_table(["Cluster", "Paired difference", "95% bootstrap CI"], bootstrap_rows)}

Replicates={bootstrap['paired_episode_cluster_bootstrap']['replicates']}. Counterfactual statistical/mechanism GO criteria passed={bootstrap['go_audit']['statistical_and_mechanism_criteria_passed']}; confirmation integrity criterion passed=false; BC remains locked.

## K sensitivity

K=32 and K=128 were evaluated only after `{final}` was frozen. Results are in `k_sensitivity.csv`; they cannot change the primary K=64 disposition.

{_table(["K", "Method", "Effect error", "Action RMSE", "Contact preserved", "Normalized utilization"], k_rows)}

## Direct answers

{_table(["#", "Question", "Answer", "Basis"], answers)}

## Negative runs and limitations

- All failed gates and all negative control runs remain reported; no gate stopped later experiments.
- One initial no-result local training process was interrupted before development inspection to make state/history shuffle controls permute their masks and contact indicator with the semantic bundle. It was rerun from scratch; no model-selection result was read from that interrupted process.
- Constant-score predictors make rank correlation undefined; the frozen metric implementation records Spearman/Kendall as zero. A diagnostic warning in the first development run led only to an explicit constant-range check with identical numeric semantics.
- Controls use one member versus three-member primary ensembles, so small ablation differences are not cleanly attributable to one input.
- C0 is a privileged hard-phase reproduction and only a conservative diagnostic/comparator; C5/C6 never consume the phase label.
- Raw branch arrays are too large for ordinary Git and remain at the bound scratch paths; repository JSON records their byte sizes and SHA-256 markers, while all row-level decoded-action results are committed as CSV.

## Recommended next experiment

Do not start BC from this audit. The bounded next validation is a preregistered C3-only decoder diagnostic on genuinely untouched episodes: retain the already implemented bi-encoder distance through final action selection, compare it directly with B2 and the true-effect oracle, and separately audit the existing C4 hard-negative/objective path on suffix-localized support. This tests the localized C3-to-C4 reversal without introducing a new policy or claiming a new idea. Do not launch ACT, Diffusion Policy, SmolVLA or pi0.5 automatically.
"""
    atomic_text(os.path.join(output_root, "STAGE3_REPORT.md"), report)
    feishu_report = f"""# 实验报告

## 中文结论摘要

最终处置：**`{final}`**。

- Stage 3 的严格支持 oracle 仍然有价值：Gate A 提升 {_pct(gates['A']['pooled_relative_gain'])}，4/4 任务、3/3 接触敏感任务改善。
- 学习排序没有通过：C4 相对 `{rank_baseline}` 的 oracle regret 变化为 {_pct(gates['B']['oracle_regret_relative_gain'])}，NDCG@16 变化 {_fmt(gates['B']['ndcg_at_16_absolute_gain'])}，Recall@8={_fmt(gates['B']['recall_at_8'])}，所以 Gate B 失败。
- 主方法 C5 没有保住 C3 的收益：相对 `{baseline}` 的真实物理效应误差变化为 {_pct(gates['C']['realized_relative_gain'])}，K=64 利用率 {_fmt(gates['C']['normalized_utilization'])}，action RMSE 退化 {_pct(gates['C']['action_rmse_degradation'])}；Gate C 失败。
- episodes 40–49 已按要求全部执行，但证据标签是 `{bootstrap['evidence_label']}`，不是 untouched confirmation，不能解锁 BC。
- 没有训练 ACT、Diffusion Policy、SmolVLA、pi0.5 或任何策略；PAI 作业数为 0；预测器训练仅使用一张本地 A800。

## 机理反解

代码与控制实验共同指向同一条下降链：C3 bi-encoder 本身能学到较强的效果几何，并把 development 真实误差从 B2 的 {_fmt(dev_base['balanced_task_effect_error'])} 降到 {_fmt(_summary(dev, 'C3_NC_BIENCODER')['balanced_task_effect_error'])}。但是 C5 只用 C3 做 K=64 FPS 覆盖，最终选择完全由表现较差的 C4 pair ranker 决定；C4 把 C3 的排序收益逆转，K=64 压缩再带来较小的二次损失。nominal/state/history/label shuffle 没有摧毁正收益，因为 C4/C5 相对 C3 本来就是负收益。详细逐代码路径、support-family 分解和限制见下方完整报告及 `MECHANISM_REVERSE_AUDIT.md`。

## 代码与制品

GitHub 主分支：<https://github.com/mikasaTu/R13-P15-Consequence-Adaptive-Action-Alphabets/tree/main>

本页以下内容是仓库内冻结的 `STAGE3_REPORT.md` 完整正文。

---

{report}
"""
    atomic_text(
        os.path.join(output_root, "FEISHU_EXPERIMENT_REPORT.md"),
        feishu_report,
    )
    return {"path": os.path.join(output_root, "STAGE3_REPORT.md"), "disposition": final}


def verify_stage3_release(project_root, output_root):
    checks = []

    def check(name, passed, observed=None, expected=None):
        checks.append({"name": name, "passed": bool(passed), "observed": observed, "expected": expected})

    for name in REQUIRED_ARTIFACTS:
        check("required_artifact:" + name, os.path.isfile(os.path.join(output_root, name)))
    split = _json(os.path.join(output_root, "episode_split.json"))
    check("snapshot_count", split["snapshot_count"] == 544, split["snapshot_count"], 544)
    check("all_episodes_successful", split["all_episodes_successful"])
    check("support_separation", split["support_separation"]["passed"])
    reuse = _json(os.path.join(output_root, "training_reuse_validation.json"))
    check("training_reuse", reuse["passed"] and reuse["states"] == 256, reuse.get("states"), 256)
    collection = _json(os.path.join(output_root, "branch_collection_validation.json"))
    check("full_collection", collection["passed"] and collection["states"] == 544, collection.get("states"), 544)
    invariants = _json(os.path.join(output_root, "pair_invariant_validation.json"))
    check("pair_invariants", invariants["passed"])
    registry = _json(os.path.join(output_root, "trained_model_registry.json"))
    check("one_visible_gpu", registry["visible_gpu_count"] == 1, registry["visible_gpu_count"], 1)
    check("settings_frozen_before_development", registry["method_settings_frozen_before_development"])
    import pyarrow.parquet as pq

    pair_rows = pq.ParquetFile(os.path.join(output_root, "training_pairs.parquet")).metadata.num_rows
    check("training_pair_rows", pair_rows == registry["training_pairs"]["rows"], pair_rows, registry["training_pairs"]["rows"])
    for filename, expected_rows, expected_split in (
        ("development_quantization.csv", 64 * 96 * len(EXPECTED_METHODS), "development"),
        ("confirmation_quantization.csv", 160 * 96 * len(EXPECTED_METHODS), "confirmation"),
    ):
        count = 0
        methods = set()
        splits = set()
        with open(os.path.join(output_root, filename), "r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                count += 1
                methods.add(row["method"])
                splits.add(row["split"])
        check(filename + ":rows", count == expected_rows, count, expected_rows)
        check(filename + ":methods", methods == EXPECTED_METHODS, sorted(methods), sorted(EXPECTED_METHODS))
        check(filename + ":split", splits == {expected_split}, sorted(splits), [expected_split])
    gate = _json(os.path.join(output_root, "development_gate.json"))
    check("all_development_methods", gate["all_planned_development_methods_executed"])
    check("settings_frozen_before_holdout", gate["method_settings_frozen_before_holdout"])
    check("strict_confirmation_false", gate["strict_untouched_confirmation_available"] is False)
    check("go_locked_at_development", gate["go_to_small_bc_available"] is False)
    permutation = gate["candidate_permutation_checks"]
    check("candidate_permutation_exact", all(row["identical_selected_bank_indices"] and row["mismatches"] == 0 for row in permutation))
    bootstrap = _json(os.path.join(output_root, "bootstrap_results.json"))
    check("bootstrap_replicates", bootstrap["paired_episode_cluster_bootstrap"]["replicates"] == 10000, bootstrap["paired_episode_cluster_bootstrap"]["replicates"], 10000)
    check("all_holdout_methods", bootstrap["all_planned_holdout_methods_executed"])
    check("untouched_confirmation_false", bootstrap["strict_untouched_confirmation"] is False)
    check("go_to_small_bc_false", bootstrap["go_audit"]["go_to_small_bc_available"] is False)
    check("k_after_disposition", bootstrap["final_disposition_frozen_before_k_sensitivity"])
    k_rows = _csv(os.path.join(output_root, "k_sensitivity.csv"))
    check("k_sensitivity_values", {int(row["alphabet_k"]) for row in k_rows} == {32, 128})
    environment = _json(os.path.join(output_root, "execution_environment.json"))
    check("pai_jobs_zero", environment["pai_jobs_submitted"] == 0)
    check("policy_training_false", environment["policy_training_performed"] is False)
    check("local_gpu_limit", environment["local_gpu_limit_respected"])
    failures = [row for row in checks if not row["passed"]]
    payload = {
        "created_utc": utc_now(),
        "final_disposition": bootstrap["final_disposition"],
        "checks": checks,
        "check_count": len(checks),
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
        "required_artifact_sha256": {
            name: sha256_file(os.path.join(output_root, name))
            for name in REQUIRED_ARTIFACTS
        },
    }
    atomic_json(os.path.join(output_root, "STAGE3_RELEASE_VERIFICATION.json"), payload)
    if failures:
        raise RuntimeError("Stage 3 release verification failed: %s" % failures)
    return payload


def finalize_stage3(project_root, output_root):
    validate_pair_invariants(output_root)
    report = write_stage3_report(project_root, output_root)
    verification = verify_stage3_release(project_root, output_root)
    return {
        "report": report["path"],
        "final_disposition": report["disposition"],
        "release_checks": verification["check_count"],
        "release_passed": verification["passed"],
    }
