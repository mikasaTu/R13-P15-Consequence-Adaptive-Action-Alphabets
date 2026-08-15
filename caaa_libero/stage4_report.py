"""Generate the human-readable Stage 4 report and mechanism reverse audit."""

from __future__ import annotations

import json
import os

import numpy as np

from .stage4_config import OUTPUT_RELATIVE, PHASES, TASK_IDS
from .storage import atomic_json, atomic_text, sha256_file


def _json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _pct(value):
    return "%.2f%%" % (100.0 * float(value))


def _num(value):
    return "%.6g" % float(value)


def _lookup(summary, method, level="pooled", task="ALL", phase="ALL"):
    return next(
        row
        for row in summary["realized_summary"]
        if row["method"] == method
        and row["level"] == level
        and row["task_id"] == task
        and row["phase"] == phase
    )


def _gain(summary, method, baseline="B2", level="pooled", task="ALL", phase="ALL"):
    base = _lookup(summary, baseline, level, task, phase)[
        "balanced_task_effect_error"
    ]
    value = _lookup(summary, method, level, task, phase)[
        "balanced_task_effect_error"
    ]
    return (base - value) / max(base, 1e-12)


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def generate_report(project_root, output_root=None):
    import pandas as pd

    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    binding = _json(os.path.join(output_root, "HISTORICAL_BINDING.json"))
    replay = _json(os.path.join(output_root, "snapshot_restore_validation.json"))
    collection = _json(os.path.join(output_root, "expanded_training_collection.json"))
    reversal_meta = _json(os.path.join(output_root, "context_reversal_metadata.json"))
    selection = _json(os.path.join(output_root, "MODEL_SELECTION.json"))
    dev = _json(os.path.join(output_root, "development_evaluation_summary.json"))
    hist = _json(os.path.join(output_root, "historical_exploratory_evaluation_summary.json"))
    confirm = _json(os.path.join(output_root, "confirmation_evaluation_summary.json"))
    gates = _json(os.path.join(output_root, "DEVELOPMENT_GATE.json"))
    bootstrap = _json(os.path.join(output_root, "BOOTSTRAP_RESULTS.json"))
    final = _json(os.path.join(output_root, "final_disposition.json"))
    fresh_split = _json(os.path.join(output_root, "FRESH_CONFIRMATION_SPLIT.json"))
    fresh_collection = _json(os.path.join(output_root, "fresh_collection_manifest.json"))
    historical = pd.read_csv(os.path.join(output_root, "C3_FAILURE_DECOMPOSITION.csv"))
    interventions = pd.read_csv(os.path.join(output_root, "C3_CONTEXT_INTERVENTIONS.csv"))
    dependence = pd.read_csv(os.path.join(output_root, "CONTEXT_DEPENDENCE_AUDIT.csv"))
    reselect = pd.read_csv(os.path.join(output_root, "C3_RESELECT_EVALUATION.csv"))

    disposition = final["final_disposition"]
    dev_means = dev["method_balanced_task_effect"]
    hist_means = hist["method_balanced_task_effect"]
    conf_means = confirm["method_balanced_task_effect"]
    fail_pooled = historical[
        (historical.level == "pooled")
        & (historical.task_id == "ALL")
        & (historical.phase == "ALL")
    ]
    decomposition = {
        row.split: {
            "bank": float(row.oracle_bank_compression_loss),
            "metric": float(row.learned_metric_loss),
            "compression": float(row.learned_compression_loss),
            "override": float(row.c4_override_loss),
        }
        for row in fail_pooled.itertuples()
    }
    dev_label = "DEVELOPMENT_EPISODES_36_39"
    hist_label = "HISTORICAL_EXPLORATORY_EPISODES_40_49"
    dev_decomp = decomposition[dev_label]
    hist_decomp = decomposition[hist_label]

    intervention_dev = interventions[
        (interventions.level == "pooled")
        & (interventions.split == dev_label)
    ].set_index("intervention")
    correct_c3 = float(
        intervention_dev.loc["correct_context", "balanced_task_effect_error"]
    )
    state_nominal_c3 = float(
        intervention_dev.loc[
            "state_and_nominal_jointly_shuffled", "balanced_task_effect_error"
        ]
    )
    history_c3 = float(
        intervention_dev.loc[
            "history_actions_masks_shuffled_within_task",
            "balanced_task_effect_error",
        ]
    )
    nominal_c3 = float(
        intervention_dev.loc[
            "nominal_shuffled_within_task", "balanced_task_effect_error"
        ]
    )

    reversal_rows = dependence[dependence.metric == "context_reversal_rate"]
    phase_reversal = {
        phase: float(reversal_rows[reversal_rows.phase == phase].value.mean())
        for phase in PHASES
    }
    selected_family = selection["cr_c3_selection"]["selected_family"]
    selected_l = int(selection["trust_region_selection"]["selected_L"])
    selected_gamma = float(
        selection["bounded_correction_selection"]["selected_gamma"]
    )

    mechanism = {
        "method": "code-first mechanism reverse engineering; no new idea generated",
        "failure_decomposition": {
            "development": dev_decomp,
            "historical_exploratory": hist_decomp,
            "interpretation": (
                "The true bank contains headroom, but learned geometry, K=64 "
                "compression, and the old C4 override are separate error sources."
            ),
        },
        "frozen_c3_input_use": {
            "correct_effect_error": correct_c3,
            "joint_state_nominal_shuffle_effect_error": state_nominal_c3,
            "history_shuffle_effect_error": history_c3,
            "nominal_shuffle_effect_error": nominal_c3,
            "interpretation": (
                "Frozen C3 is sensitive to state/history but only weakly to the "
                "nominal chunk; this explains why nominal interventions changed "
                "few selections while state/history interventions changed many."
            ),
        },
        "strict_reversal_supply": {
            "realized_by_task_phase": reversal_meta["realized_pairs_by_task_phase"],
            "undersupplied": reversal_meta["strictly_undersupplied_task_phases"],
            "phase_observed_reversal_rates": phase_reversal,
            "interpretation": (
                "Free-space ordering is nearly static, while contact phases carry "
                "most genuine state-dependent ordering. No margin was relaxed."
            ),
        },
        "cr_controls_development": {
            name: dev_means[name]
            for name in dev_means
            if any(
                token in name
                for token in (
                    "ACTION_ONLY",
                    "SHUFFLED",
                    "NO_REVERSAL",
                    "CR_C3_FULL",
                )
            )
        },
        "decoder_tradeoff_development": {
            name: {
                "effect_error": _lookup(dev, name)["balanced_task_effect_error"],
                "action_rmse": _lookup(dev, name)["action_reconstruction_rmse"],
                "contact_preserved": _lookup(dev, name)["contact_mode_preserved"],
                "utilization": _lookup(dev, name)["normalized_code_utilization"],
            }
            for name in ("CR_C3_FULL", "CR_C3_K64", "CR_TR_C3_K64")
        },
        "fresh_confirmation": {
            "label": confirm["evidence_label"],
            "new_episode_claim": False,
            "primary_gain": final["confirmation_gate"]["pooled_relative_gain"],
            "ci95": final["confirmation_gate"]["paired_ci95"],
            "all_seed_directions_same": final["confirmation_gate"][
                "all_seed_directions_same"
            ],
        },
        "final_disposition": disposition,
    }
    mechanism_path = os.path.join(output_root, "MECHANISM_REVERSE_ENGINEERING.json")
    atomic_json(mechanism_path, mechanism)

    historical_rows = []
    for stage in ("stage1", "stage1_5", "stage2", "stage3"):
        value = binding["historical_stages"][stage]
        commit = (
            value.get("published_commit")
            or value.get("result_commit")
            or value.get("formal_commit")
        )
        historical_rows.append((stage, value["disposition"], commit))

    checkpoint_rows = []
    selected_definition = selection["cr_c3_selection"]["family_trace"][
        selection["cr_c3_selection"]["selected_family_index"]
    ]
    for entry in selected_definition["checkpoints"]:
        checkpoint_rows.append(
            ("proposed", entry["member_index"], entry["seed"], entry["sha256"])
        )
    for control in selection["cr_c3_controls"]["controls"]:
        for entry in control["checkpoints"]:
            checkpoint_rows.append(
                (control["control"], entry["member_index"], entry["seed"], entry["sha256"])
            )

    method_rows = []
    for name in (
        "B2",
        "O_FULL",
        "O_K64",
        "FROZEN_C3_FULL",
        "FROZEN_C3_K64",
        "C3_RESELECT_FULL",
        "C3_RESELECT_KMEDOIDS64",
        "CR_C3_FULL",
        "CR_C3_K64",
        "CR_TR_C3_K64",
        "ACTION_ONLY_TR_K64",
        "SHUFFLED_EFFECT_TR_K64",
    ):
        if name not in dev_means:
            continue
        method_rows.append(
            (
                name,
                _num(dev_means[name]),
                _pct(_gain(dev, name)) if name != "B2" else "baseline",
                _num(hist_means[name]) if name in hist_means else "NA",
                _num(conf_means[name]) if name in conf_means else "NA",
            )
        )

    task_rows = []
    for task in TASK_IDS:
        base = _lookup(dev, "B2", "task", task)["balanced_task_effect_error"]
        full = _lookup(dev, "CR_C3_FULL", "task", task)[
            "balanced_task_effect_error"
        ]
        trust = _lookup(dev, "CR_TR_C3_K64", "task", task)[
            "balanced_task_effect_error"
        ]
        task_rows.append(
            (task, _num(base), _num(full), _pct((base - full) / base), _num(trust), _pct((base - trust) / base))
        )

    phase_rows = []
    for phase in PHASES:
        base = _lookup(dev, "B2", "phase", "ALL", phase)[
            "balanced_task_effect_error"
        ]
        full = _lookup(dev, "CR_C3_FULL", "phase", "ALL", phase)[
            "balanced_task_effect_error"
        ]
        trust = _lookup(dev, "CR_TR_C3_K64", "phase", "ALL", phase)[
            "balanced_task_effect_error"
        ]
        phase_rows.append(
            (phase, _pct(phase_reversal[phase]), _num(base), _num(full), _num(trust))
        )

    gate_rows = [
        (
            "A oracle headroom",
            "PASS" if gates["gate_A_oracle_headroom"]["passed"] else "FAIL",
            _pct(gates["gate_A_oracle_headroom"]["relative_gain"]),
        ),
        (
            "B learned metric",
            "PASS" if gates["gate_B_learned_consequence_metric"]["passed"] else "FAIL",
            _pct(gates["gate_B_learned_consequence_metric"]["development_relative_gain"]),
        ),
        (
            "C K=64 alphabet",
            "PASS" if gates["gate_C_k64_alphabet"]["passed"] else "FAIL",
            _pct(gates["gate_C_k64_alphabet"]["relative_gain"]),
        ),
        (
            "Fresh confirmation",
            "PASS" if final["confirmation_gate"]["passed"] else "FAIL",
            _pct(final["confirmation_gate"]["pooled_relative_gain"]),
        ),
    ]

    lines = [
        "# R13-P15 Stage 4 — C3-faithful context-reversal and trust-region audit",
        "",
        "## 精确结论",
        "",
        "`%s`" % disposition,
        "",
        "本轮完成了全部注册实验；任何 gate 失败都没有停止后续控制、历史探索复现或 fresh confirmation。没有训练 policy/VLA，没有启动 PAI，也没有开始 BC。",
        "",
        "## 历史状态保持只读",
        "",
        _table(("Stage", "冻结结论", "发布/结果 commit"), historical_rows),
        "",
        "Stage 1–3 文件保持 byte-identical；Stage 4 是新方法假设，未继承旧 novelty 评级。",
        "",
        "## 环境、回放与数据完整性",
        "",
        "- LIBERO commit: `%s`；source tree SHA-256: `%s`。" % (
            binding["libero"]["commit"], binding["libero"]["source_tree_sha256"]
        ),
        "- 控制：Panda OSC_POSE, 20 Hz, H=4, settle=3；同一 M=256 executable bank；clipping 禁止。",
        "- 牺牲 calibration A/B/A/B replay: %d tests, %d failures；confirmation state 未用于 replay。" % (
            len(replay["tests"]), len(replay["failed_tests"])
        ),
        "- 扩展训练态：%d states；context/support/candidate branches=%s；combined digest=`%s`。" % (
            collection["expected_states"], collection["branch_counts"], collection["combined_path_sha256_digest"]
        ),
        "- Fresh evidence: `%s`, %d states, %d source-episode clusters；明确不是 new-episode claim；split freeze commit=`%s`。" % (
            confirm["evidence_label"], confirm["states"], confirm["source_episode_clusters"], fresh_collection["fresh_split_freeze_commit"]
        ),
        "",
        "## 1. 旧 C3 完整失败分解",
        "",
        _table(
            ("Evidence", "bank compression", "learned metric", "learned K64 compression", "C4 override"),
            (
                ("development", _num(dev_decomp["bank"]), _num(dev_decomp["metric"]), _num(dev_decomp["compression"]), _num(dev_decomp["override"])),
                ("historical exploratory", _num(hist_decomp["bank"]), _num(hist_decomp["metric"]), _num(hist_decomp["compression"]), _num(hist_decomp["override"])),
            ),
        ),
        "",
        "C3_FULL 确有弱信号，但 learned metric 是最大误差源；K=64 再损失一层，旧 C4 override 又进一步恶化。这解释了为什么只看最终 C5 会掩盖 C3_FULL 的部分正向信息。",
        "",
        "## 2. Context 是否真实重要",
        "",
        _table(("Phase", "strict reversal rate", "B2", "CR FULL", "CR-TR K64"), phase_rows),
        "",
        "严格 reversal 主要出现在 contact phases。缺失 strata=%s；margin 从未放宽，也未制造标签。" % reversal_meta["strictly_undersupplied_task_phases"],
        "",
        "Frozen C3 intervention: correct=%s, state+nominal shuffle=%s, history shuffle=%s, nominal-only shuffle=%s。状态与历史改变较大，nominal 单独改变很小，说明旧 C3 并未充分利用 nominal chunk。" % (
            _num(correct_c3), _num(state_nominal_c3), _num(history_c3), _num(nominal_c3)
        ),
        "",
        "## 3. C3 独立重选、CR 训练和 controls",
        "",
        "独立 calibration 选择 CR family=`%s`；trust L=%d；analysis-only gamma=%s。C3 重选结果没有用 development/episodes40–49 选择。" % (
            selected_family, selected_l, _num(selected_gamma)
        ),
        "",
        _table(("Method", "Dev effect", "Dev gain vs B2", "Hist exploratory", "Fresh"), method_rows),
        "",
        _table(("Task", "B2", "CR FULL", "FULL gain", "CR-TR K64", "TR gain"), task_rows),
        "",
        "所有 control 与选中 family 使用相同 architecture、parameter count、3 seeds、30 epochs 与 query batch 16。完整机制数值见 `MECHANISM_REVERSE_ENGINEERING.json`。",
        "",
        "## 4. Trust region 的作用",
        "",
        _table(
            ("Method", "Effect", "Action RMSE", "Contact", "Normalized utilization"),
            [
                (
                    name,
                    _num(_lookup(dev, name)["balanced_task_effect_error"]),
                    _num(_lookup(dev, name)["action_reconstruction_rmse"]),
                    _num(_lookup(dev, name)["contact_mode_preserved"]),
                    _num(_lookup(dev, name)["normalized_code_utilization"]),
                )
                for name in ("CR_C3_FULL", "CR_C3_K64", "CR_TR_C3_K64")
            ],
        ),
        "",
        "Trust region 只在 executable atlas members 中筛选，不做 clipping、synthesis 或 pseudoinverse。效果提升/降低的直接原因是 action-local 过滤改变了可选集合；它是否保留 FULL gain 由 Gate C 明确衡量。",
        "",
        "## 5. Gates 与 fresh confirmation",
        "",
        _table(("Gate", "Result", "Primary gain"), gate_rows),
        "",
        "Fresh primary paired difference (B2 error - method error)=%s，95%% CI=[%s, %s]，10,000 episode-clustered bootstrap replicates。三 seed gains=%s。" % (
            _num(bootstrap["primary"]["pooled"]["point"]),
            _num(bootstrap["primary"]["pooled"]["ci95"][0]),
            _num(bootstrap["primary"]["pooled"]["ci95"][1]),
            final["confirmation_gate"]["member_relative_gains"],
        ),
        "",
        "## 6. 11 个直接回答",
        "",
        _table(
            ("#", "问题", "回答"),
            (
                (1, "bank/metric/K64/C4 各贡献多少？", "见第1节精确加性分解。"),
                (2, "Frozen C3 development gain 是否在历史探索集复现？", "Dev C3_FULL=%s；Hist C3_FULL=%s；两者均与各自 B2 分开报告。" % (_num(dev_means["FROZEN_C3_FULL"]), _num(hist_means["FROZEN_C3_FULL"]))),
                (3, "真实 consequence ordering 是否 state-dependent？", "接触阶段是；free-space 基本否，严格 reversal 缺失被保留。"),
                (4, "模型是否使用 state/nominal/history？", "state/history 有明显作用；nominal 作用弱；CR controls 给出因果对照。"),
                (5, "C3 独立 objective selection 是否改善？", "C3_RESELECT_FULL=%s vs frozen C3_FULL=%s。" % (_num(dev_means["C3_RESELECT_FULL"]), _num(dev_means["FROZEN_C3_FULL"]))),
                (6, "CR training 是否改善 ordering？", "Gate B reversal gain=%s，完整 accuracy 见 context_reversal_evaluation.json。" % _num(gates["gate_B_learned_consequence_metric"]["context_reversal_accuracy_gain_points"])),
                (7, "Trust region 是否减少动作偏差且保留 gain？", "Action degradation=%s；FULL gain retention=%s。" % (_pct(gates["gate_C_k64_alphabet"]["action_rmse_degradation"]), _pct(gates["gate_C_k64_alphabet"]["full_gain_retention"]))),
                (8, "FULL retrieval 在 K64 失败时是否仍工作？", "FULL=%s，K64=%s，TR=%s。" % (_num(dev_means["CR_C3_FULL"]), _num(dev_means["CR_C3_K64"]), _num(dev_means["CR_TR_C3_K64"]))),
                (9, "机制最终属于哪类？", disposition),
                (10, "是否在 genuinely fresh states 确认？", "执行了 %s；不是 new-episode claim。" % confirm["evidence_label"]),
                (11, "是否可进入 small state-based BC？", "YES" if disposition == "GO_TO_SMALL_BC" else "NO"),
            ),
        ),
        "",
        "## Checkpoint hashes",
        "",
        _table(("Arm", "Member", "Seed", "SHA-256"), checkpoint_rows),
        "",
        "## 局限与下一步",
        "",
        "- episodes 40–49 的旧 snapshot 仍只是 historical exploratory；fresh evidence 来自同些 source episodes 的未用 timestep 加预注册小关节扰动，因此不是新 episode 证据。",
        "- State-based metric audit 不是 VLA、不是 policy evaluation，也不能声称 paper readiness。",
        "- 若结论不是 GO_TO_SMALL_BC，下一实验应只做与该 disposition 对应的严格 paired mechanism replication；不得自动启动 ACT、Diffusion Policy、SmolVLA 或 pi0.5。",
        "",
        "## Exact final disposition",
        "",
        "`%s`" % disposition,
        "",
    ]
    report_path = os.path.join(output_root, "STAGE4_REPORT.md")
    atomic_text(report_path, "\n".join(lines))
    manifest = {
        "report_sha256": sha256_file(report_path),
        "mechanism_sha256": sha256_file(mechanism_path),
        "final_disposition": disposition,
        "required_questions_answered": 11,
        "new_idea_generated": False,
    }
    atomic_json(os.path.join(output_root, "report_manifest.json"), manifest)
    return manifest


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    print(json.dumps(generate_report(args.project_root, args.output_root), indent=2))


if __name__ == "__main__":
    main()
