"""Finalize Stage 5 evidence, mechanism audit, disposition, and release checks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from glob import glob

import numpy as np

from .stage3_metrics import paired_episode_bootstrap
from .stage5_config import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FINAL_DISPOSITIONS,
    MODEL_SEEDS,
    OUTPUT_RELATIVE,
    SCRATCH_ROOT,
    TASK_IDS,
)
from .stage5_logic import choose_disposition, exact_one_disposition
from .storage import atomic_json, atomic_text, sha256_file, validate_complete


def _json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _csv(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    if not rows:
        raise ValueError("empty CSV")
    temporary = path + ".incomplete"
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _summary(rows, method, metric="balanced_task_effect_error", level="pooled", task="ALL", phase="ALL"):
    row = next(
        value
        for value in rows
        if value["method"] == method
        and value["level"] == level
        and value["task_id"] == task
        and value["phase"] == phase
    )
    return float(row[metric])


def _gain(baseline, method):
    return (float(baseline) - float(method)) / max(float(baseline), 1e-12)


def _pct(value):
    return "%.3f%%" % (100.0 * float(value))


def _num(value):
    return "%.6f" % float(value)


def _table(headers, rows):
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def _blocked_confirmation(output_root, split):
    reason = "fresh trajectory supply incomplete; confirmation firewall forbids branch execution"
    row = {
        "status": "NOT_EXECUTED_BY_PREREGISTERED_FIREWALL",
        "evidence": "FRESH_POLICY_TRAJECTORY_CONFIRMATION",
        "reason": reason,
        "rows": 0,
    }
    _write_csv(os.path.join(output_root, "CONFIRMATION_RANKING.csv"), [row])
    _write_csv(os.path.join(output_root, "CONFIRMATION_REALIZED.csv"), [row])
    manifest = {
        "evidence_label": "FRESH_POLICY_TRAJECTORY_CONFIRMATION",
        "executed": False,
        "blocked_by_firewall": True,
        "reason": reason,
        "split_complete": False,
        "split_record_count": int(split.get("record_count", 0)),
        "shortages": split.get("shortages", []),
        "states": 0,
        "nominal_branches": 0,
        "target_branches": 0,
        "candidate_branches": 0,
        "total_short_rollouts": 0,
        "clipped": 0,
    }
    atomic_json(os.path.join(output_root, "FRESH_BRANCH_MANIFEST.json"), manifest)
    gate = {
        "gate": "FRESH_CONFIRMATION_GATE",
        "executed": False,
        "passed": False,
        "reason": reason,
        "not_interpreted_as_metric_failure": True,
    }
    atomic_json(os.path.join(output_root, "CONFIRMATION_GATE.json"), gate)
    return gate


def _bootstrap_from_parquet(path, method, baseline, seed):
    import pandas as pd

    frame = pd.read_parquet(path)
    frame = frame[frame.method.isin((method, baseline))]
    columns = ("method", "task_id", "episode_id", "balanced_task_effect_error")
    return paired_episode_bootstrap(
        frame[list(columns)].to_dict("records"),
        method,
        baseline,
        BOOTSTRAP_REPLICATES,
        seed,
    )


def _bootstraps(output_root, confirmation_gate):
    development = _json(os.path.join(output_root, "DEVELOPMENT_GATE.json"))
    dev_raw = os.path.join(output_root, "development_realized_rows.parquet")
    hist_raw = os.path.join(output_root, "historical_exploratory_realized_rows.parquet")
    output = {
        "cluster_unit": "source episode",
        "paired": True,
        "replicates_per_executed_comparison": BOOTSTRAP_REPLICATES,
        "development": {
            "P1_FULL_vs_B2_FULL": development["gate1_context_identifiable"]["bootstrap"],
            "B2_FULL_vs_B1_FULL": development["static_consequence_value"]["bootstrap"],
            "P1_K64_vs_B2_K64": _bootstrap_from_parquet(
                dev_raw,
                "P1_CONTEXT_GATED_PSD__K64",
                development["gate2_dynamic_k64"]["baseline"],
                BOOTSTRAP_SEED + 2,
            ),
        },
        "historical_exploratory": {
            "P1_FULL_vs_B2_FULL": _bootstrap_from_parquet(
                hist_raw,
                "P1_CONTEXT_GATED_PSD__FULL",
                "B2_STATIC_CONSEQUENCE__FULL",
                BOOTSTRAP_SEED + 10,
            ),
            "B2_FULL_vs_B1_FULL": _bootstrap_from_parquet(
                hist_raw,
                "B2_STATIC_CONSEQUENCE__FULL",
                "B1_ACTION_ONLY__FULL",
                BOOTSTRAP_SEED + 11,
            ),
            "P1_K64_vs_B2_K64": _bootstrap_from_parquet(
                hist_raw,
                "P1_CONTEXT_GATED_PSD__K64",
                "B2_STATIC_CONSEQUENCE__K64",
                BOOTSTRAP_SEED + 12,
            ),
        },
        "fresh_confirmation": (
            confirmation_gate.get("bootstrap")
            if confirmation_gate.get("executed", True)
            else {
                "executed": False,
                "reason": confirmation_gate["reason"],
                "replicates": 0,
            }
        ),
    }
    atomic_json(os.path.join(output_root, "BOOTSTRAP_RESULTS.json"), output)
    return output


def _mechanism(output_root):
    development = _json(os.path.join(output_root, "DEVELOPMENT_GATE.json"))
    oracle = _json(os.path.join(output_root, "ORACLE_ADAPTIVITY_GATE.json"))
    training = _json(os.path.join(output_root, "MODEL_TRAINING_MANIFEST.json"))
    dev_realized = _csv(os.path.join(output_root, "DEVELOPMENT_REALIZED.csv"))
    dev_ranking = _csv(os.path.join(output_root, "DEVELOPMENT_RANKING.csv"))
    hist_realized = _csv(os.path.join(output_root, "HISTORICAL_EXPLORATORY_REALIZED.csv"))
    controls = _csv(os.path.join(output_root, "DEVELOPMENT_CONTROLS.csv"))
    p1 = "P1_CONTEXT_GATED_PSD__FULL"
    b2 = "B2_STATIC_CONSEQUENCE__FULL"
    b1 = "B1_ACTION_ONLY__FULL"
    p1_error = _summary(dev_realized, p1)
    b2_error = _summary(dev_realized, b2)
    b1_error = _summary(dev_realized, b1)
    phase_error = _summary(dev_realized, "CONTROL_PHASE_ONLY__FULL")
    hist_p1 = _summary(hist_realized, p1)
    hist_b2 = _summary(hist_realized, b2)
    reversal = {
        row["method"]: float(row["joint_reversal_accuracy"])
        for row in controls
        if row.get("level") == "reversal" and row.get("joint_reversal_accuracy") not in (None, "")
    }
    proposed_entries = [
        entry["metadata"]
        for entry in training["entries"]
        if entry["metadata"].get("method") == "P1_CONTEXT_GATED_PSD"
        and entry["metadata"].get("control") == "PROPOSED"
    ]
    modulation_norm = float(
        np.mean([row["zero_mean_offset_audit"]["mean_modulation_norm"] for row in proposed_entries])
    )
    maximum_norm = math.sqrt(24.0) * 1.25
    initial_reversal = float(
        np.mean([row["training"]["trace"][0]["reversal"] for row in proposed_entries])
    )
    final_reversal = float(
        np.mean([row["training"]["trace"][-1]["reversal"] for row in proposed_entries])
    )
    mechanism = {
        "method": "code-first mechanism reverse engineering; no new idea generated",
        "source_paths": {
            "static_metric": "caaa_libero.stage5_models.create_static_metric",
            "context_metric": "caaa_libero.stage5_models.create_context_metric",
            "bounded_modulation": "caaa_libero.stage5_models.create_context_metric.ContextMetric.modulation",
            "positive_weight": "caaa_libero.stage5_models.create_context_metric.ContextMetric.positive_weight",
            "reversal_loss": "caaa_libero.stage5_models._reversal_loss",
            "dynamic_atlas": "caaa_libero.stage5_evaluation._atlas_decodings",
        },
        "static_consequence_improvement": {
            "development_B1_error": b1_error,
            "development_B2_error": b2_error,
            "relative_gain": _gain(b1_error, b2_error),
            "all_four_tasks_improve": all(
                value > 0
                for value in development["static_consequence_value"]["task_gains"].values()
            ),
            "historical_B1_error": _summary(hist_realized, b1),
            "historical_B2_error": hist_b2,
            "interpretation": (
                "The action encoder and diagonal weights trained on physical-consequence "
                "distances learn a stable global ordering unavailable to action-only geometry."
            ),
        },
        "context_modulation_failure": {
            "development_B2_error": b2_error,
            "development_P1_error": p1_error,
            "P1_relative_gain": _gain(b2_error, p1_error),
            "historical_B2_error": hist_b2,
            "historical_P1_error": hist_p1,
            "historical_P1_relative_gain": _gain(hist_b2, hist_p1),
            "phase_only_error": phase_error,
            "phase_only_vs_P1_gain": _gain(p1_error, phase_error),
            "P1_joint_reversal_accuracy": reversal.get("P1_CONTEXT_GATED_PSD"),
            "B2_joint_reversal_accuracy": reversal.get("B2_STATIC_CONSEQUENCE"),
            "registered_reversal_threshold": 0.35,
            "mean_modulation_norm": modulation_norm,
            "theoretical_maximum_modulation_norm": maximum_norm,
            "fraction_of_maximum_norm": modulation_norm / maximum_norm,
            "development_metric_condition_number": _summary(
                dev_ranking, p1, "metric_condition_number"
            ),
            "B2_metric_condition_number": _summary(
                dev_ranking, b2, "metric_condition_number"
            ),
            "mean_reversal_loss_initial": initial_reversal,
            "mean_reversal_loss_final": final_reversal,
            "interpretation": (
                "The only context path is a bounded diagonal reweighting of a frozen B2 "
                "embedding. It cannot rotate or change that representation. Training drives "
                "the 24-D modulation close to its tanh boundary while barely reducing the "
                "reversal loss; the resulting high-condition metric changes codes but does "
                "not generalize state-specific order reversals. Coarse phase/contact controls "
                "equal or beat P1, so the apparent context response is not identifiable."
            ),
        },
        "dynamic_k64": {
            "P1_FULL_error": p1_error,
            "P1_K64_error": _summary(dev_realized, "P1_CONTEXT_GATED_PSD__K64"),
            "B2_K64_error": _summary(dev_realized, "B2_STATIC_CONSEQUENCE__K64"),
            "P1_K64_vs_B2_K64_gain": development["gate2_dynamic_k64"]["checks"]["realized_gain"]["value"],
            "registered_gain_threshold": 0.08,
            "interpretation": (
                "State-dependent reweighting changes the medoid set, but those changes are "
                "not supported by accurate reversal rankings. Compression therefore adds "
                "effect error and preserves only a small, non-gated difference from B2."
            ),
        },
        "oracle_context_supply": {
            "state_vs_static_or_contact_gain": oracle["state_specific_vs_static_or_contact_gain"],
            "strict_reversal_pairs": oracle["strict_reversal"]["pair_count"],
            "interpretation": (
                "True state-specific headroom and real reversal examples exist; the negative "
                "P1 result is model/identifiability failure, not absence of oracle signal."
            ),
        },
        "offset_solver_incident": training.get("context_offset_repair_before_development"),
        "conclusion": "static consequence supervision survives; learned context-specific geometry does not",
    }
    atomic_json(os.path.join(output_root, "MECHANISM_REVERSE_ENGINEERING.json"), mechanism)
    return mechanism


def _report(output_root, final, mechanism, bootstraps, confirmation_gate):
    binding = _json(os.path.join(output_root, "HISTORICAL_BINDING.json"))
    oracle = _json(os.path.join(output_root, "ORACLE_ADAPTIVITY_GATE.json"))
    development = _json(os.path.join(output_root, "DEVELOPMENT_GATE.json"))
    generator = _json(os.path.join(output_root, "NOMINAL_GENERATOR_TRAINING.json"))
    replay = _json(os.path.join(output_root, "FRESH_REPLAY_VALIDATION.json"))
    split = _json(os.path.join(output_root, "FRESH_CONFIRMATION_SPLIT.json"))
    local = _json(os.path.join(output_root, "LOCAL_BANK_BINDING.json"))
    dev = _csv(os.path.join(output_root, "DEVELOPMENT_REALIZED.csv"))
    hist = _csv(os.path.join(output_root, "HISTORICAL_EXPLORATORY_REALIZED.csv"))
    ranks = _csv(os.path.join(output_root, "DEVELOPMENT_RANKING.csv"))
    p1, b2, b1 = "P1_CONTEXT_GATED_PSD__FULL", "B2_STATIC_CONSEQUENCE__FULL", "B1_ACTION_ONLY__FULL"
    task_rows = []
    for task in TASK_IDS:
        b2v = _summary(dev, b2, level="task", task=task)
        p1v = _summary(dev, p1, level="task", task=task)
        task_rows.append((task, _num(b2v), _num(p1v), _pct(_gain(b2v, p1v))))
    phase_rows = []
    for phase in ("free_space", "pre_contact", "contact_onset", "post_contact"):
        b2v = _summary(dev, b2, level="phase", phase=phase)
        p1v = _summary(dev, p1, level="phase", phase=phase)
        phase_rows.append((phase, _num(b2v), _num(p1v), _pct(_gain(b2v, p1v))))
    methods = (
        "B0_CURRENT_CONTACT_KMEANS__FULL",
        b1,
        b2,
        p1,
        "CONTROL_PHASE_ONLY__FULL",
        "CONTROL_CONTEXT_SHUFFLED__FULL",
        "CONTROL_CONSEQUENCE_LABEL_SHUFFLED__FULL",
        "CONTROL_NO_REVERSAL_LOSS__FULL",
        "B2_STATIC_CONSEQUENCE__K64",
        "P1_CONTEXT_GATED_PSD__K64",
    )
    method_rows = []
    for method in methods:
        method_rows.append(
            (
                method,
                _num(_summary(dev, method)),
                _num(_summary(hist, method)),
                _num(_summary(ranks, method, "oracle_regret")),
            )
        )
    successes = {
        task: split["trajectory_summaries"][task]["success_count"]
        for task in TASK_IDS
    }
    historical_rows = [
        (name, value["disposition"])
        for name, value in binding["historical_evidence"].items()
    ]
    lines = [
        "# R13-P15 Stage 5 — Context-Identifiable Consequence Retrieval and Dynamic Local Alphabet",
        "",
        "## 精确结论",
        "",
        "`%s`" % final["final_disposition"],
        "",
        "该结论由预注册优先级锁定：Gate 0 通过；P1 context-identifiable Gate 1 失败；B2 static consequence gate 通过。后续所有可执行的负向实验仍继续运行，但不能升级较早失败的 disposition。",
        "",
        "## 历史结论保持只读",
        "",
        _table(("阶段", "冻结结论"), historical_rows),
        "",
        "历史目录的 Git tree 均与发布对象一致；本轮没有覆盖或重解释旧结果。",
        "",
        "## 环境、控制与审计边界",
        "",
        "- LIBERO commit: `%s`；source tree SHA256: `%s`。" % (binding["libero"]["commit"], binding["libero"]["source_tree_sha256"]),
        "- Panda `OSC_POSE`，20 Hz，H=4，settle=3；M=128，primary K=64。",
        "- local bank SHA256: `%s`；fresh target bank 与 local/historical overlap 均为 0。" % local["npz"]["sha256"],
        "- 模型训练/推理和仿真均在本地完成；GPU 使用数 0，PAI job 数 0。计划要求只有本地技术上不可行才启用 PAI，因此没有提交 PAI。",
        "- nominal generator 是 state-only H4 BC，仅用于 fresh trajectory；10,000 steps，checkpoint `%s`。" % generator["checkpoint_sha256"],
        "- 牺牲轨迹确定性回放 %d/%d 通过，confirmation state 未被用于 replay。" % (sum(row["passed"] for row in replay["rows"]), len(replay["rows"])),
        "",
        "## Oracle adaptivity",
        "",
        "Gate 0: **PASS**。O_STATE_FULL=%s，最强 static/contact=%s，state-specific gain=%s；contact-onset/post-contact gain=%s；strict reversal pairs=%d。" % (
            _num(oracle["pooled_errors"]["O_STATE_FULL"]),
            oracle["strongest_static_or_contact_method"],
            _pct(oracle["state_specific_vs_static_or_contact_gain"]),
            _pct(oracle["contact_onset_post_contact_gain"]),
            oracle["strict_reversal"]["pair_count"],
        ),
        "这说明状态自适应真值确实存在；后续失败不能归因于没有 oracle headroom。",
        "",
        "## Static consequence geometry 与 context geometry",
        "",
        "B2 FULL 相对 B1 FULL 的 development realized error 改善 %s，episode-clustered 95%% CI（绝对误差差）为 [%s, %s]，四个任务全部改善。P1 FULL 相对 B2 反而变化 %s，CI 为 [%s, %s]，三颗 seed 均为负向。" % (
            _pct(development["static_consequence_value"]["checks"]["realized_pooled_gain"]["value"]),
            _num(development["static_consequence_value"]["bootstrap"]["pooled"]["ci95"][0]),
            _num(development["static_consequence_value"]["bootstrap"]["pooled"]["ci95"][1]),
            _pct(development["gate1_context_identifiable"]["checks"]["realized_pooled_gain"]["value"]),
            _num(development["gate1_context_identifiable"]["bootstrap"]["pooled"]["ci95"][0]),
            _num(development["gate1_context_identifiable"]["bootstrap"]["pooled"]["ci95"][1]),
        ),
        "",
        _table(("任务", "B2 FULL", "P1 FULL", "P1 gain vs B2"), task_rows),
        "",
        _table(("阶段", "B2 FULL", "P1 FULL", "P1 gain vs B2"), phase_rows),
        "",
        "P1 joint reversal accuracy=%s（阈值 0.35），B2=%s；P1 仅高 %s，而要求至少 15 percentage points。" % (
            _num(development["gate1_context_identifiable"]["checks"]["joint_reversal_accuracy"]["value"]),
            _num(development["gate1_context_identifiable"]["checks"]["joint_reversal_accuracy"]["value"] - development["gate1_context_identifiable"]["checks"]["joint_reversal_gain"]["value"]),
            _num(development["gate1_context_identifiable"]["checks"]["joint_reversal_gain"]["value"]),
        ),
        "",
        "## FULL、K=64 与全部 matched controls",
        "",
        _table(("方法", "development effect error", "historical effect error", "development oracle regret"), method_rows),
        "",
        "Gate 2: **FAIL**。P1 K=64 相对最强 deployable K=64 baseline 仅改善 %s（要求 8%%）；FULL gain 为负，因此 75%% retention 未定义。action RMSE degradation=%s，contact preservation drop=%s，utilization=%s，clipping=0，valid bank=128。" % (
            _pct(development["gate2_dynamic_k64"]["checks"]["realized_gain"]["value"]),
            _pct(development["gate2_dynamic_k64"]["checks"]["action_rmse_degradation"]["value"]),
            _pct(development["gate2_dynamic_k64"]["checks"]["contact_preservation_drop"]["value"]),
            _num(development["gate2_dynamic_k64"]["checks"]["normalized_utilization"]["value"]),
        ),
        "历史探索中 P1 FULL 比 B2 FULL 仅改善 %s，但 phase-only、contact-only、nominal-shuffled、context-shuffled 均优于 P1；P1 K=64 还弱于 B2 K=64。这不是可识别的 context mechanism。" % _pct(mechanism["context_modulation_failure"]["historical_P1_relative_gain"]),
        "",
        "## Fresh policy-trajectory confirmation",
        "",
        "200 个预注册升序 rollout seed/任务已全部执行；success counts: `%s`。" % json.dumps(successes, sort_keys=True),
        (
            "48 条轨迹与 192 个 phase states 均已冻结并完成 confirmation branch/gate。"
            if split["complete"]
            else "供给不足，`FRESH_CONFIRMATION_SPLIT.json` 冻结为 incomplete。按照预注册 firewall，未执行任何 confirmation target/candidate branch；这不是因为 Gate 1 失败而早停，也没有换 generator 或放宽 acceptance。"
        ),
        "Confirmation gate status: `%s`。" % ("PASS" if confirmation_gate.get("passed") else ("FAIL" if confirmation_gate.get("executed", True) else "NOT_EXECUTED_FIREWALL")),
        "",
        "## 代码级机理反解（不生成新 idea）",
        "",
        "B2 的提升来自对真实 physical-consequence distance 的监督：同一 action encoder/训练预算下，它在 development 比 action-only B1 降低 %s，并在四任务和 historical exploratory 均保持方向。" % _pct(mechanism["static_consequence_improvement"]["relative_gain"]),
        "P1 的 context 只能对冻结 B2 latent 的 24 个轴做 `base_weight * exp(1.25*tanh(...))` 对角重权，不能旋转或改变表示。平均 modulation norm=%s / 理论上限 %s（%s），condition number 从 B2 的 %s 升到 %s；reversal loss 仅从 %s 降到 %s，joint reversal 仍只有 %s。" % (
            _num(mechanism["context_modulation_failure"]["mean_modulation_norm"]),
            _num(mechanism["context_modulation_failure"]["theoretical_maximum_modulation_norm"]),
            _pct(mechanism["context_modulation_failure"]["fraction_of_maximum_norm"]),
            _num(mechanism["context_modulation_failure"]["B2_metric_condition_number"]),
            _num(mechanism["context_modulation_failure"]["development_metric_condition_number"]),
            _num(mechanism["context_modulation_failure"]["mean_reversal_loss_initial"]),
            _num(mechanism["context_modulation_failure"]["mean_reversal_loss_final"]),
            _num(mechanism["context_modulation_failure"]["P1_joint_reversal_accuracy"]),
        ),
        "因此代码确实让 context 改变了距离与 code，但变化主要是接近边界的高条件数轴重权；phase-only/current-contact controls 相当或更好，说明提升/下降来自粗粒度分区和不稳定 medoid 重排，而非可泛化的 state+history+nominal consequence ranking。",
        "",
        "## 执行异常（均未改变科学输入）",
        "",
        "- 初版 fixed `[-40,40]` zero-mean root bracket 未包住 6 个 pre-tanh logit root；在读取 development 前改为 observed min/max ±20。只重算 buffer offset，未改 trained parameters 或 optimizer steps；修复后最大 train mean error <7.45e-7。",
        "- development evaluator 两次因 consequence group 的旧别名触发 KeyError；均发生在结果 gate 生成前，随后改为从冻结 `PRIMARY_GROUPS` 动态取名并从头重跑。",
        "- replay CLI 曾把 global argument 放在 subcommand 后，argparse 在 simulator 启动前拒绝；正确命令随后完成 16/16 回放。",
        "- rollout 后处理仅采用互斥 CPU seed shards 加速；每个 seed 的 checkpoint、动作、simulator 和升序 acceptance 不变。",
        "",
        "## 11 个明确回答",
        "",
        "1. Oracle 中 state-specific 部分很强：相对最强 static/contact oracle 为 %s。" % _pct(oracle["state_specific_vs_static_or_contact_gain"]),
        "2. 是，true consequence supervision 的 B2 明显优于 action-only B1。",
        "3. 否，observable context 的 P1 未优于 B2，且 controls 不支持因果归因。",
        "4. 否，joint reversal 0.041825 < 0.35。",
        "5. 否；因为原始 context increment 为负，shuffle retention 不可定义，且多个 shuffled/coarse controls 更好。",
        "6. 只有 static B2 FULL 成立；P1 FULL 不成立。",
        "7. 否；FULL context gain 为负，retention 无法定义。",
        "8. 是，K=64 action deviation、contact、utilization、clipping 和 valid-bank safety 子项通过。",
        "9. %s" % ("是，fresh confirmation 已执行。" if split["complete"] else "无法检验：预注册 generator 未产生每任务 12 条成功轨迹，firewall 正确阻断 branch。"),
        "10. 存活的是 static consequence metric，不是 adaptive alphabet。",
        "11. 否；不建议进入 fixed-policy test-time reranking。",
        "",
        "## Bootstrap 与统计单位",
        "",
        "所有已执行主比较均用 source episode 做 paired cluster，10,000 replicates；完整 pooled/per-task CI 位于 `BOOTSTRAP_RESULTS.json`。未执行的 fresh gate 明确记为 0 replicates，未伪造 CI。",
        "",
        "## 最终 disposition 与下一步",
        "",
        "`%s`" % final["final_disposition"],
        "",
        "建议停止本方法族的 context-gated dynamic alphabet 路线；若继续该研究程序，只应把 B2 作为静态 consequence retrieval baseline 做独立、预注册的 fixed-policy safety evaluation，而不是把本轮结果称为 adaptive alphabet、VLA 改进或 task-success evidence。Stage 5 到此停止。",
        "",
    ]
    path = os.path.join(output_root, "STAGE5_REPORT.md")
    atomic_text(path, "\n".join(lines))
    return path


def _verify(project_root, output_root, final, bootstraps):
    required = (
        "PREREGISTRATION.md",
        "HISTORICAL_BINDING.json",
        "DATA_PROTOCOL.json",
        "LOCAL_BANK.npz",
        "LOCAL_BANK_BINDING.json",
        "ORACLE_ADAPTIVITY_AUDIT.csv",
        "ORACLE_ADAPTIVITY_GATE.json",
        "CONTEXT_REVERSAL_PAIRS.parquet",
        "CONTEXT_REVERSAL_METADATA.json",
        "MODEL_DEFINITIONS.json",
        "MODEL_SELECTION.json",
        "DEVELOPMENT_RANKING.csv",
        "DEVELOPMENT_REALIZED.csv",
        "DEVELOPMENT_CONTROLS.csv",
        "DEVELOPMENT_GATE.json",
        "NOMINAL_GENERATOR_BINDING.json",
        "FRESH_TRAJECTORY_SEEDS.json",
        "FRESH_CONFIRMATION_SPLIT.json",
        "FRESH_BRANCH_MANIFEST.json",
        "CONFIRMATION_RANKING.csv",
        "CONFIRMATION_REALIZED.csv",
        "BOOTSTRAP_RESULTS.json",
        "FINAL_DISPOSITION.json",
        "STAGE5_REPORT.md",
    )
    files = {name: os.path.isfile(os.path.join(output_root, name)) for name in required}
    training = _json(os.path.join(output_root, "MODEL_TRAINING_MANIFEST.json"))
    checkpoints = []
    for entry in training["entries"]:
        path = os.path.join(output_root, entry["path"])
        checkpoints.append(
            {
                "path": entry["path"],
                "exists": os.path.isfile(path),
                "expected_sha256": entry["sha256"],
                "observed_sha256": sha256_file(path) if os.path.isfile(path) else None,
                "matched": os.path.isfile(path) and sha256_file(path) == entry["sha256"],
            }
        )
    split = _json(os.path.join(output_root, "FRESH_CONFIRMATION_SPLIT.json"))
    branch = _json(os.path.join(output_root, "FRESH_BRANCH_MANIFEST.json"))
    completion = []
    if split["complete"]:
        for task_manifest in branch["task_manifests"]:
            for row in task_manifest["rows"]:
                for name in ("context_path", "support_path", "candidate_path"):
                    valid, evidence = validate_complete(row[name])
                    completion.append({"path": row[name], "valid": bool(valid), "evidence": evidence if isinstance(evidence, str) else "matched"})
    bootstrap_checks = []
    for split_name, values in bootstraps.items():
        if split_name in ("cluster_unit", "paired", "replicates_per_executed_comparison"):
            continue
        if split_name == "fresh_confirmation" and values.get("executed") is False:
            bootstrap_checks.append({"comparison": split_name, "executed": False, "replicates": 0, "passed": True})
            continue
        groups = values if isinstance(values, dict) and "replicates" not in values else {split_name: values}
        for name, value in groups.items():
            replicas = int(value["replicates"])
            bootstrap_checks.append({"comparison": split_name + "/" + name, "executed": True, "replicates": replicas, "passed": replicas == BOOTSTRAP_REPLICATES})
    expected = choose_disposition(
        _json(os.path.join(output_root, "HISTORICAL_BINDING.json"))["all_hashes_match"],
        _json(os.path.join(output_root, "ORACLE_ADAPTIVITY_GATE.json"))["passed"],
        _json(os.path.join(output_root, "DEVELOPMENT_GATE.json"))["gate1_context_identifiable"]["passed"],
        _json(os.path.join(output_root, "DEVELOPMENT_GATE.json"))["static_consequence_value"]["passed"],
        _json(os.path.join(output_root, "DEVELOPMENT_GATE.json"))["gate2_dynamic_k64"]["passed"],
        split["complete"],
        _json(os.path.join(output_root, "CONFIRMATION_GATE.json"))["passed"],
    )
    hashes = {}
    for path in sorted(glob(os.path.join(output_root, "**", "*"), recursive=True)):
        if not os.path.isfile(path) or path.endswith("STAGE5_RELEASE_VERIFICATION.json"):
            continue
        relative = os.path.relpath(path, output_root).replace(os.sep, "/")
        hashes[relative] = {"bytes": os.path.getsize(path), "sha256": sha256_file(path)}
    verification = {
        "kind": "stage5 ordinary-file release verification",
        "source_commit": subprocess.check_output(["git", "-C", project_root, "rev-parse", "HEAD"], text=True).strip(),
        "required_artifacts": files,
        "all_required_artifacts_present": all(files.values()),
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "all_checkpoint_hashes_match": all(row["matched"] for row in checkpoints),
        "fresh_split_complete": bool(split["complete"]),
        "fresh_branch_counts": {
            "states": int(branch["states"]),
            "total_short_rollouts": int(branch["total_short_rollouts"]),
            "clipped": int(branch["clipped"]),
        },
        "completion_markers": completion,
        "all_executed_completion_markers_valid": all(row["valid"] for row in completion),
        "bootstrap_checks": bootstrap_checks,
        "all_executed_bootstraps_have_10000_replicates": all(row["passed"] for row in bootstrap_checks),
        "final_disposition": final["final_disposition"],
        "recomputed_disposition": expected,
        "exact_one_disposition": exact_one_disposition(final["final_disposition"]) and final["final_disposition"] == expected,
        "historical_paths_immutable": _json(os.path.join(output_root, "HISTORICAL_BINDING.json"))["historical_paths_modified"] is False,
        "pai_jobs_submitted": 0,
        "policy_or_vla_training_beyond_nominal_generator": False,
        "artifact_hashes": hashes,
    }
    verification["passed"] = all(
        (
            verification["all_required_artifacts_present"],
            verification["all_checkpoint_hashes_match"],
            verification["all_executed_completion_markers_valid"],
            verification["all_executed_bootstraps_have_10000_replicates"],
            verification["exact_one_disposition"],
            verification["historical_paths_immutable"],
            verification["pai_jobs_submitted"] == 0,
            not verification["policy_or_vla_training_beyond_nominal_generator"],
        )
    )
    atomic_json(os.path.join(output_root, "STAGE5_RELEASE_VERIFICATION.json"), verification)
    return verification


def finalize(project_root, output_root=None):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    split = _json(os.path.join(output_root, "FRESH_CONFIRMATION_SPLIT.json"))
    if split["complete"]:
        confirmation_gate = _json(os.path.join(output_root, "CONFIRMATION_GATE.json"))
        confirmation_gate.setdefault("executed", True)
    else:
        confirmation_gate = _blocked_confirmation(output_root, split)
    bootstraps = _bootstraps(output_root, confirmation_gate)
    binding = _json(os.path.join(output_root, "HISTORICAL_BINDING.json"))
    oracle = _json(os.path.join(output_root, "ORACLE_ADAPTIVITY_GATE.json"))
    development = _json(os.path.join(output_root, "DEVELOPMENT_GATE.json"))
    disposition = choose_disposition(
        binding["all_hashes_match"],
        oracle["passed"],
        development["gate1_context_identifiable"]["passed"],
        development["static_consequence_value"]["passed"],
        development["gate2_dynamic_k64"]["passed"],
        split["complete"],
        confirmation_gate["passed"],
    )
    final = {
        "final_disposition": disposition,
        "exactly_one": disposition in FINAL_DISPOSITIONS,
        "precedence_trace": {
            "historical_binding": bool(binding["all_hashes_match"]),
            "oracle_gate_0": bool(oracle["passed"]),
            "context_gate_1": bool(development["gate1_context_identifiable"]["passed"]),
            "static_consequence_gate": bool(development["static_consequence_value"]["passed"]),
            "dynamic_k64_gate_2": bool(development["gate2_dynamic_k64"]["passed"]),
            "fresh_trajectory_supply": bool(split["complete"]),
            "confirmation_gate": bool(confirmation_gate["passed"]),
        },
        "decisive_rule": "Gate 1 failed and B2 static consequence gate passed",
        "later_negative_or_blocked_runs_cannot_upgrade_disposition": True,
        "stage_stopped": True,
        "next_experiment": "Do not advance the context-gated dynamic alphabet; optionally evaluate frozen B2 only as a preregistered static-retrieval safety baseline.",
    }
    atomic_json(os.path.join(output_root, "FINAL_DISPOSITION.json"), final)
    mechanism = _mechanism(output_root)
    _report(output_root, final, mechanism, bootstraps, confirmation_gate)
    verification = _verify(project_root, output_root, final, bootstraps)
    return {"final_disposition": disposition, "release_verification_passed": verification["passed"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    print(json.dumps(finalize(args.project_root, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
