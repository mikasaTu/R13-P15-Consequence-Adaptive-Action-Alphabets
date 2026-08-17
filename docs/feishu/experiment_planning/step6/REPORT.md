# R13-P15 Stage 5 — Context-Identifiable Consequence Retrieval and Dynamic Local Alphabet

## 精确结论

`STATIC_CONSEQUENCE_METRIC_ONLY`

该结论由预注册优先级锁定：Gate 0 通过；P1 context-identifiable Gate 1 失败；B2 static consequence gate 通过。后续所有可执行的负向实验仍继续运行，但不能升级较早失败的 disposition。

## 历史结论保持只读

| 阶段 | 冻结结论 |
|-|-|
| stage1 | REJECT_CORE_HYPOTHESIS |
| stage1_5 | REJECT_P15_FAMILY |
| stage2 | ORACLE_ONLY_NO_DEPLOYABLE_MODEL |
| stage3 | ORACLE_ONLY_NO_LEARNABLE_RANKER |
| stage4 | STATIC_EFFECT_METRIC_ONLY |

历史目录的 Git tree 均与发布对象一致；本轮没有覆盖或重解释旧结果。

## 环境、控制与审计边界

- LIBERO commit: `8f1084e3132a39270c3a13ebe37270a43ece2a01`；source tree SHA256: `e9197ca08fe4d7325f561fc40d7425167830253e0f0fceb1af2663b23292f71f`。
- repository pre-result commit `eba489ec8f866f712b582083c088e93b0aaccf11`，tree `0137158cfd5a3f4e1162acf4f47bdc073839baf9`；Stage 4 published commit `ac861eb60f83c72ac4785d8d901356434eded2ec`。
- 环境锁 SHA256: `f4421974cf948bfa765098e24819d445b209589611cbc3fe11e04c30fb0f0d3e`；simulation Python 3.8.13，model Python 3.11.11，PyTorch 2.4.1+cu121（CPU execution）。
- Panda `OSC_POSE`，20 Hz，H=4，settle=3；M=128，primary K=64。
- local bank SHA256: `21ba12712891533237796992529d1c460705c21465bff6c47d7943122d41c255`；fresh target bank 与 local/historical overlap 均为 0。
- 模型训练/推理和仿真均在本地完成；GPU 使用数 0，PAI job 数 0。计划要求只有本地技术上不可行才启用 PAI，因此没有提交 PAI。
- nominal generator 是 state-only H4 BC，仅用于 fresh trajectory；10,000 steps，checkpoint `b48a4fee2c32c1107c3da2120df0068f974288e7424718dadb8978bdc9120f88`。
- 牺牲轨迹确定性回放 16/16 通过，confirmation state 未被用于 replay。

## Oracle adaptivity

Gate 0: **PASS**。O_STATE_FULL=0.054725，最强 static/contact=O_STATIC_FULL，state-specific gain=87.630%；contact-onset/post-contact gain=89.073%；strict reversal pairs=6954。  
这说明状态自适应真值确实存在；后续失败不能归因于没有 oracle headroom。

## Static consequence geometry 与 context geometry

B2 FULL 相对 B1 FULL 的 development realized error 改善 18.778%，episode-clustered 95% CI（绝对误差差）为 [0.048107, 0.073403]，四个任务全部改善。P1 FULL 相对 B2 反而变化 -0.415%，CI 为 [-0.004845, 0.002209]，三颗 seed 均为负向。

| 任务 | B2 FULL | P1 FULL | P1 gain vs B2 |
|-|-|-|-|
| bowl_on_plate | 0.473670 | 0.480814 | -1.508% |
| plate_push | 0.216896 | 0.219331 | -1.122% |
| stove_turn_on | 0.181094 | 0.174596 | 3.588% |
| wine_rack | 0.174657 | 0.175923 | -0.725% |

| 阶段 | B2 FULL | P1 FULL | P1 gain vs B2 |
|-|-|-|-|
| free_space | 0.018603 | 0.018970 | -1.973% |
| pre_contact | 0.387563 | 0.393598 | -1.557% |
| contact_onset | 0.355466 | 0.357269 | -0.507% |
| post_contact | 0.284685 | 0.280826 | 1.356% |

P1 joint reversal accuracy=0.041825（阈值 0.35），B2=0.029785；P1 仅高 0.012041，而要求至少 15 percentage points。

## FULL、K=64 与全部 matched controls

| 方法 | development effect error | historical effect error | development oracle regret |
|-|-|-|-|
| B0_CURRENT_CONTACT_KMEANS\_\_FULL | 0.457777 | 0.403149 | 0.403053 |
| B0_CURRENT_CONTACT_KMEANS\_\_K64 | 0.454191 | 0.426061 | 0.399467 |
| B1_ACTION_ONLY\_\_FULL | 0.322054 | 0.290008 | 0.267329 |
| B1_ACTION_ONLY\_\_K64 | 0.341745 | 0.320506 | 0.287020 |
| B2_STATIC_CONSEQUENCE\_\_FULL | 0.261579 | 0.226224 | 0.206854 |
| B2_STATIC_CONSEQUENCE\_\_K64 | 0.282722 | 0.250615 | 0.227997 |
| CONTROL_ACTION_ONLY\_\_FULL | 0.322054 | 0.290008 | 0.267329 |
| CONTROL_ACTION_ONLY\_\_K64 | 0.341745 | 0.320506 | 0.287020 |
| CONTROL_CONSEQUENCE_LABEL_SHUFFLED\_\_FULL | 0.263412 | 0.225933 | 0.208687 |
| CONTROL_CONSEQUENCE_LABEL_SHUFFLED\_\_K64 | 0.281937 | 0.251910 | 0.227212 |
| CONTROL_CONTEXT_SHUFFLED\_\_FULL | 0.260294 | 0.225666 | 0.205569 |
| CONTROL_CONTEXT_SHUFFLED\_\_K64 | 0.285913 | 0.247639 | 0.231188 |
| CONTROL_CURRENT_CONTACT_ONLY\_\_FULL | 0.258914 | 0.224759 | 0.204189 |
| CONTROL_CURRENT_CONTACT_ONLY\_\_K64 | 0.275580 | 0.250874 | 0.220856 |
| CONTROL_JOINT_STATE_NOMINAL_SHUFFLED\_\_FULL | 0.262550 | 0.228616 | 0.207825 |
| CONTROL_JOINT_STATE_NOMINAL_SHUFFLED\_\_K64 | 0.283533 | 0.248728 | 0.228809 |
| CONTROL_NOMINAL_SHUFFLED\_\_FULL | 0.260808 | 0.225438 | 0.206083 |
| CONTROL_NOMINAL_SHUFFLED\_\_K64 | 0.289585 | 0.250887 | 0.234860 |
| CONTROL_NO_REVERSAL_LOSS\_\_FULL | 0.258866 | 0.227131 | 0.204141 |
| CONTROL_NO_REVERSAL_LOSS\_\_K64 | 0.280065 | 0.252525 | 0.225340 |
| CONTROL_PHASE_ONLY\_\_FULL | 0.257697 | 0.223227 | 0.202972 |
| CONTROL_PHASE_ONLY\_\_K64 | 0.270807 | 0.251408 | 0.216082 |
| P1_CONTEXT_GATED_PSD\_\_FULL | 0.262666 | 0.225730 | 0.207941 |
| P1_CONTEXT_GATED_PSD\_\_K64 | 0.280280 | 0.254253 | 0.225556 |

Gate 2: **FAIL**。P1 K=64 相对最强 deployable K=64 baseline 仅改善 0.864%（要求 8%）；FULL gain 为负，因此 75% retention 未定义。action RMSE degradation=2.328%，contact preservation drop=0.602%，utilization=0.302768，clipping=0，valid bank=128。  
历史探索中 P1 FULL 比 B2 FULL 仅改善 0.218%，但 phase-only、contact-only、nominal-shuffled、context-shuffled 均优于 P1；P1 K=64 还弱于 B2 K=64。这不是可识别的 context mechanism。

## Fresh policy-trajectory confirmation

每任务预注册 200 个升序 rollout seed；成功任务的 acceptance 在第 12 个升序成功处冻结，不足任务扫完全部 200 个。冻结 acceptance attempts=`{"bowl_on_plate": 85, "plate_push": 200, "stove_turn_on": 12, "wine_rack": 200}`，successes=`{"bowl_on_plate": 12, "plate_push": 3, "stove_turn_on": 12, "wine_rack": 0}`；并发 shard 在 cutoff 后已完成的少量高 seed 仅保留在 attempts audit，不进入 acceptance。  
供给不足，`FRESH_CONFIRMATION_SPLIT.json` 冻结为 incomplete。按照预注册 firewall，未执行任何 confirmation target/candidate branch；这不是因为 Gate 1 失败而早停，也没有换 generator 或放宽 acceptance。  
Confirmation gate status: `NOT_EXECUTED_FIREWALL`。

## 代码级机理反解（不生成新 idea）

B2 的提升来自对真实 physical-consequence distance 的监督：同一 action encoder/训练预算下，它在 development 比 action-only B1 降低 18.778%，并在四任务和 historical exploratory 均保持方向。  
P1 的 context 只能对冻结 B2 latent 的 24 个轴做 `base_weight * exp(1.25*tanh(...))` 对角重权，不能旋转或改变表示。平均 modulation norm=5.903557 / 理论上限 6.123724（96.405%），condition number 从 B2 的 1.049279 升到 12.592835；reversal loss 仅从 1.468333 降到 1.447313，joint reversal 仍只有 0.041825。  
因此代码确实让 context 改变了距离与 code，但变化主要是接近边界的高条件数轴重权；phase-only/current-contact controls 相当或更好，说明提升/下降来自粗粒度分区和不稳定 medoid 重排，而非可泛化的 state+history+nominal consequence ranking。

## 执行异常（均未改变科学输入）

- 初版 fixed `[-40,40]` zero-mean root bracket 未包住 6 个 pre-tanh logit root；在读取 development 前改为 observed min/max ±20。只重算 buffer offset，未改 trained parameters 或 optimizer steps；修复后最大 train mean error <7.45e-7。
- development evaluator 两次因 consequence group 的旧别名触发 KeyError；均发生在结果 gate 生成前，随后改为从冻结 `PRIMARY_GROUPS` 动态取名并从头重跑。
- replay CLI 曾把 global argument 放在 subcommand 后，argparse 在 simulator 启动前拒绝；正确命令随后完成 16/16 回放。
- rollout 后处理仅采用互斥 CPU seed shards 加速；每个 seed 的 checkpoint、动作、simulator 和升序 acceptance 不变。

## 11 个明确回答

1. Oracle 中 state-specific 部分很强：相对最强 static/contact oracle 为 87.630%。
2. 是，true consequence supervision 的 B2 明显优于 action-only B1。
3. 否，observable context 的 P1 未优于 B2，且 controls 不支持因果归因。
4. 否，joint reversal 0.041825 < 0.35。
5. 否；因为原始 context increment 为负，shuffle retention 不可定义，且多个 shuffled/coarse controls 更好。
6. 只有 static B2 FULL 成立；P1 FULL 不成立。
7. 否；FULL context gain 为负，retention 无法定义。
8. 是，K=64 action deviation、contact、utilization、clipping 和 valid-bank safety 子项通过。
9. 无法检验：预注册 generator 未产生每任务 12 条成功轨迹，firewall 正确阻断 branch。
10. 存活的是 static consequence metric，不是 adaptive alphabet。
11. 否；不建议进入 fixed-policy test-time reranking。

## Bootstrap 与统计单位

所有已执行主比较均用 source episode 做 paired cluster，10,000 replicates；完整 pooled/per-task CI 位于 `BOOTSTRAP_RESULTS.json`。未执行的 fresh gate 明确记为 0 replicates，未伪造 CI。

## 最终 disposition 与下一步

`STATIC_CONSEQUENCE_METRIC_ONLY`

建议停止本方法族的 context-gated dynamic alphabet 路线；若继续该研究程序，只应把 B2 作为静态 consequence retrieval baseline 做独立、预注册的 fixed-policy safety evaluation，而不是把本轮结果称为 adaptive alphabet、VLA 改进或 task-success evidence。Stage 5 到此停止。
