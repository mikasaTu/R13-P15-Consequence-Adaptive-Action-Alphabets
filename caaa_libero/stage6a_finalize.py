"""Finalize a Gate-H-stopped Stage 6-A replay without effect comparisons."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os

from .stage6a_config import BOOTSTRAP_REPLICATES, FINAL_DISPOSITIONS, OUTPUT_RELATIVE
from .stage6a_statistics import choose_disposition
from .storage import atomic_json, sha256_file


REQUIRED = (
    "PREREGISTRATION.md",
    "HISTORICAL_BINDING.json",
    "DATA_SOURCE_BINDING.json",
    "DEFECT_REPRODUCTION.json",
    "REPAIRED_DEFINITION.json",
    "ATLAS_K64.json",
    "QUANTIZER_HEALTH.json",
    "DEVELOPMENT_RANKING.csv",
    "DEVELOPMENT_REALIZED.csv",
    "DEVELOPMENT_CONTROLS.csv",
    "development_realized_rows.parquet",
    "RECOVERY_ACCOUNTING.json",
    "BOOTSTRAP_RESULTS.json",
    "GATE_H.json",
    "GATE_A.json",
    "FINAL_DISPOSITION.json",
    "STAGE6A_REPORT.md",
    "MECHANISM_REVERSE_ENGINEERING.json",
    "TEST_RESULTS.json",
    "FEISHU_PUBLICATION.json",
)


def _load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _skipped_csv(path, artifact):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("artifact", "status", "reason"))
        writer.writeheader()
        writer.writerow(
            {
                "artifact": artifact,
                "status": "NOT_COMPUTED",
                "reason": "GATE_H_FAILED_EFFECT_ERROR_FORBIDDEN",
            }
        )


def _mechanism(output_root, health):
    states = _load(os.path.join(output_root, "ATLAS_K64.json"))["states"]
    task_phase = {}
    for row in states:
        key = (row["task_id"], row["phase"])
        task_phase.setdefault(key, []).append(row["normalized_assignment_utilization"])
    partitions = [
        {
            "task_id": task,
            "phase": phase,
            "mean_utilization": sum(values) / len(values),
            "minimum_utilization": min(values),
            "maximum_utilization": max(values),
            "states": len(values),
        }
        for (task, phase), values in sorted(task_phase.items())
    ]
    result = {
        "kind": "stage6a_mechanism_reverse_engineering",
        "method": "code_first_execution_path_and_tensor_audit",
        "new_idea_generated": False,
        "observed_repairs": {
            "direct_executable_index_decode": {
                "clipped_coordinate_fraction_before": 0.8342013888888888,
                "clipped_coordinate_fraction_after": 0.0,
                "mechanism": "Selection now returns a frozen executable candidate index; pseudo-inverse synthesis and coordinate clipping are absent.",
            },
            "balanced_metric_binding": {
                "legacy_contact_and_force_error_share": 0.9999526293361927,
                "mechanism": "The health gate no longer uses the force-dominated Stage 1 normalization; Stage 2-5 balanced consequence semantics remain frozen.",
            },
        },
        "remaining_failure": {
            "failed_check": "median_normalized_assignment_utilization_strictly_greater_than_0.50",
            "observed": health["metrics"]["median_normalized_assignment_utilization"],
            "threshold": 0.50,
            "mechanism": (
                "K-medoids spreads 64 medoids across the 128 candidate C3 embeddings, but the 96 frozen target queries occupy a narrower local C3 region. "
                "Nearest-neighbor selection therefore activates a median 28 of 64 codes. Removing C4 cannot repair this pre-selection occupancy collapse because C4 is downstream of atlas construction and target-to-code assignment."
            ),
            "not_explained_by": [
                "coordinate clipping (zero)",
                "missing candidates (128 available)",
                "pooled bank-wide dead codes (0.07874, passing)",
                "action reconstruction blow-up (ratio 1.03539, passing)",
            ],
        },
        "task_phase_structural_utilization": partitions,
        "causal_scope": (
            "This localizes a structural occupancy failure before effect lookup. It does not estimate whether C4 deletion would improve or hurt physical-effect error."
        ),
    }
    atomic_json(os.path.join(output_root, "MECHANISM_REVERSE_ENGINEERING.json"), result)
    return result


def _report(health, mechanism):
    m = health["metrics"]
    return f"""# R13-P15 Stage 6-A 实验报告

## 结论

本轮严格停在 Gate H，唯一 disposition 为 `QUANTIZER_STILL_DEGENERATE`。这不是对 Stage 1–5
结论的改写，也不是 Stage 6-B 的许可。Stage 6-A 没有训练、没有仿真、没有 PAI job，也没有
读取或计算任何 Stage 6-A effect-error improvement。

## 历史与缺陷复现

`HISTORICAL_BINDING.json` 与 `DATA_SOURCE_BINDING.json` 均通过。Stage 1–5 的冻结 disposition
保持不变。D1 精确复现：median assignment utilization = 0.015625，median clipped fraction =
0.8342013889，`contact_and_force` squared-error share = 0.9999526293。D2 精确复现：C3/C4
oracle regret = 0.2429703025 / 0.3187190024，NDCG@16 = 0.6195790530 / 0.4448980708；
B2/C3/C4/C5 realized error = 0.3081719328 / 0.2831500712 / 0.3588987713 /
0.3749693724。D3 为 5.9035571416 / 6.1237243570（96.405%）。

## Gate H

- median normalized distinct-code utilization：{m['median_normalized_assignment_utilization']:.6f}，要求 `>0.50`，失败；
- median clipped-coordinate fraction：{m['median_realized_clipped_coordinate_fraction']:.6f}，通过；
- pooled dead-code fraction：{m['pooled_dead_code_fraction']:.6f}，通过；
- action RMSE / B2 K64：{m['action_reconstruction_rmse_ratio']:.6f}，通过；
- minimum valid candidates：{m['valid_candidates_per_state_min']}，通过。

因此五项中仅 utilization 失败，但它是硬门槛；`GATE_H.json` 明确记录
`evaluated_before_effect_error=true`。

## 机制反解（无新 idea）

代码路径确认了两项修复确实生效：R1 返回真实 executable bank index，所以裁剪从 83.42%
降为 0；健康检查绑定 balanced metric，不再由旧 force group 支配。但 C3 K-medoids 负责把
64 个 medoid 分散到 128-candidate bank，而 96 个冻结 target query 在每个 state 的局部 C3
空间只覆盖较窄区域，最终中位只激活 28/64 个 code。C4 位于旧流水线的 atlas 后端，删除它
不会改变这一前端 occupancy collapse。这是本轮降低/未恢复的直接实现机理，不构成新方法。

## 八个必答问题

1. D1/D2 是否精确复现？是，数值见上，全部通过 `1e-4` 相对容差。
2. K=64 是否通过 Gate H？否，utilization 0.4375 对严格门槛 `>0.50`。
3. `c4_removal_delta` 是多少？`NOT_COMPUTED_DUE_GATE_H`；按预注册禁止读取 effect error。
4. Gate 0 K=64 headroom 恢复多少？`NOT_COMPUTED_DUE_GATE_H`。
5. random-atlas / actuator-uniform 是否存活？`NOT_RUN_DUE_GATE_H`，不能据此声称 density specificity。
6. 哪些 task/phase 承载 effect？effect 未计算；仅结构利用率分解写入 `MECHANISM_REVERSE_ENGINEERING.json`。
7. 是否支持重审 Stage 1 `REJECT_CORE_HYPOTHESIS`？不支持。它只证明 D1 的 clipping/metric
   缺陷可被移除，同时发现 repaired alphabet 仍未过健康门。
8. 最便宜的反证实验是什么？在不看 effect outcome 的前提下，对同一冻结 64-state、96-target、
   128-candidate 表重新执行一次完全相同的 C3 K-medoids/selection；若 median distinct utilization
   严格超过 0.50，则本轮结构性退化结论被直接反证。该复算不需要训练、仿真或 PAI。

## 产物语义与停止点

`DEVELOPMENT_RANKING.csv`、`DEVELOPMENT_REALIZED.csv`、`DEVELOPMENT_CONTROLS.csv` 和
`development_realized_rows.parquet` 是显式 `NOT_COMPUTED` 哨兵，不含 effect 数字；
`RECOVERY_ACCOUNTING.json`、`BOOTSTRAP_RESULTS.json` 与 `GATE_A.json` 同样记录 preregistered
skip。已停止，不进入 Stage 6-B。
"""


def run(project_root, output_root=None):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    history = _load(os.path.join(output_root, "HISTORICAL_BINDING.json"))
    data = _load(os.path.join(output_root, "DATA_SOURCE_BINDING.json"))
    defects = _load(os.path.join(output_root, "DEFECT_REPRODUCTION.json"))
    gate_h = _load(os.path.join(output_root, "GATE_H.json"))
    health = _load(os.path.join(output_root, "QUANTIZER_HEALTH.json"))
    disposition = choose_disposition(
        history["passed"], data["passed"], defects["passed"], gate_h["passed"]
    )
    if disposition != "QUANTIZER_STILL_DEGENERATE":
        raise RuntimeError("this finalizer is only valid for the registered Gate H stop")
    for name in ("DEVELOPMENT_RANKING.csv", "DEVELOPMENT_REALIZED.csv", "DEVELOPMENT_CONTROLS.csv"):
        _skipped_csv(os.path.join(output_root, name), name)
    import pandas as pd

    pd.DataFrame(
        [{"status": "NOT_COMPUTED", "reason": "GATE_H_FAILED_EFFECT_ERROR_FORBIDDEN"}]
    ).to_parquet(os.path.join(output_root, "development_realized_rows.parquet"), index=False)
    atomic_json(
        os.path.join(output_root, "RECOVERY_ACCOUNTING.json"),
        {"computed": False, "reason": "GATE_H_FAILED_EFFECT_ERROR_FORBIDDEN", "effect_numbers_present": False},
    )
    atomic_json(
        os.path.join(output_root, "BOOTSTRAP_RESULTS.json"),
        {
            "executed": False,
            "reason": "GATE_H_FAILED_EFFECT_ERROR_FORBIDDEN",
            "registered_replicates": BOOTSTRAP_REPLICATES,
            "executed_replicates": 0,
            "cluster_unit": "source_episode",
        },
    )
    atomic_json(
        os.path.join(output_root, "GATE_A.json"),
        {"evaluated": False, "passed": False, "reason": "NOT_REACHED_GATE_H_FAILED"},
    )
    mechanism = _mechanism(output_root, health)
    final = {
        "final_disposition": disposition,
        "exactly_one_disposition": True,
        "allowed_dispositions": list(FINAL_DISPOSITIONS),
        "precedence_trace": [
            {"condition": "historical_binding", "passed": history["passed"]},
            {"condition": "executed_cache_binding", "passed": data["passed"]},
            {"condition": "defect_reproduction", "passed": defects["passed"]},
            {"condition": "gate_h", "passed": gate_h["passed"]},
            {"condition": "stop_on_first_failure", "selected": disposition},
        ],
        "stage6b_started": False,
        "effect_error_comparisons_computed": False,
    }
    atomic_json(os.path.join(output_root, "FINAL_DISPOSITION.json"), final)
    with open(os.path.join(output_root, "STAGE6A_REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write(_report(health, mechanism))
    return final


def release_verify(project_root, output_root=None):
    project_root = os.path.abspath(project_root)
    output_root = output_root or os.path.join(project_root, OUTPUT_RELATIVE)
    missing = [name for name in REQUIRED if not os.path.isfile(os.path.join(output_root, name))]
    final = _load(os.path.join(output_root, "FINAL_DISPOSITION.json"))
    hashes = {
        name: {"sha256": sha256_file(os.path.join(output_root, name)), "bytes": os.path.getsize(os.path.join(output_root, name))}
        for name in REQUIRED
        if os.path.isfile(os.path.join(output_root, name))
    }
    checks = {
        "required_artifacts_present": not missing,
        "exactly_one_disposition": final.get("exactly_one_disposition") is True,
        "registered_disposition": final.get("final_disposition") in FINAL_DISPOSITIONS,
        "history_immutable": _load(os.path.join(output_root, "HISTORICAL_BINDING.json"))["passed"],
        "no_effect_comparison_after_gate_h_failure": final.get("effect_error_comparisons_computed") is False,
        "no_stage6b": final.get("stage6b_started") is False,
        "all_tests_passed": _load(os.path.join(output_root, "TEST_RESULTS.json"))["all_passed"],
        "feishu_readback_verified": _load(
            os.path.join(output_root, "FEISHU_PUBLICATION.json")
        )["readback_verified"],
    }
    result = {
        "kind": "stage6a_release_verification",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checks": checks,
        "missing": missing,
        "artifact_hashes": hashes,
        "final_disposition": final["final_disposition"],
        "passed": bool(all(checks.values())),
    }
    atomic_json(os.path.join(output_root, "STAGE6A_RELEASE_VERIFICATION.json"), result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root, args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()
