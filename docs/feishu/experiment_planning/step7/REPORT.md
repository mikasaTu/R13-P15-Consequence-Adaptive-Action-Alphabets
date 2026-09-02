# R13-P15 Stage 6-A 实验报告

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

- median normalized distinct-code utilization：0.437500，要求 `>0.50`，失败；
- median clipped-coordinate fraction：0.000000，通过；
- pooled dead-code fraction：0.078740，通过；
- action RMSE / B2 K64：1.035388，通过；
- minimum valid candidates：128，通过。

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
