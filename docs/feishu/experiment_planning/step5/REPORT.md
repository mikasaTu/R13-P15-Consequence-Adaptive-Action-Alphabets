# R13-P15 Stage 4 — C3-faithful context-reversal and trust-region audit

## 精确结论

`STATIC_EFFECT_METRIC_ONLY`

本轮完成了全部注册实验；任何 gate 失败都没有停止后续控制、历史探索复现或 fresh confirmation。没有训练 policy/VLA，没有启动 PAI，也没有开始 BC。

## 历史状态保持只读

| Stage | 冻结结论 | 发布/结果 commit |
|-|-|-|
| stage1 | REJECT_CORE_HYPOTHESIS | 434427af0f8adc844851c27cfc050b2c9c6752dc |
| stage1_5 | REJECT_P15_FAMILY | 76433b6e58196ceeedc4ad005a1110ea8e343ae2 |
| stage2 | ORACLE_ONLY_NO_DEPLOYABLE_MODEL | 74c98979910a3831d0abeb8d13111a7c9294b067 |
| stage3 | ORACLE_ONLY_NO_LEARNABLE_RANKER | beb63576e91307260b64687e58ea99e6da93c478 |

Stage 1–3 文件保持 byte-identical；Stage 4 是新方法假设，未继承旧 novelty 评级。

## 环境、回放与数据完整性

- LIBERO commit: `8f1084e3132a39270c3a13ebe37270a43ece2a01`；source tree SHA-256: `e9197ca08fe4d7325f561fc40d7425167830253e0f0fceb1af2663b23292f71f`。
- 控制：Panda OSC_POSE, 20 Hz, H=4, settle=3；同一 M=256 executable bank；clipping 禁止。
- 牺牲 calibration A/B/A/B replay: 4 tests, 0 failures；confirmation state 未用于 replay。
- 扩展训练态：768 states；context/support/candidate branches={'candidate': 197376, 'context': 768, 'support': 74496}；combined digest=`4c18c94bf4fb71a771042af13e9de7dde8c3b815cdfea4077dcdbd3938e49021`。
- Fresh evidence: `FRESH_PERTURBED_STATE_CONFIRMATION`, 160 states, 40 source-episode clusters；明确不是 new-episode claim；split freeze commit=`88ff3f55debb2237ad9e0ec1dd165f971fa1d37b`。

## 1. 旧 C3 完整失败分解

| Evidence | bank compression | learned metric | learned K64 compression | C4 override |
|-|-|-|-|-|
| development | 0.00445268 | 0.0447971 | 0.0110299 | 0.0087987 |
| historical exploratory | 0.00332936 | 0.0557246 | 0.010921 | 0.0163488 |

C3_FULL 确有弱信号，但 learned metric 是最大误差源；K=64 再损失一层，旧 C4 override 又进一步恶化。这解释了为什么只看最终 C5 会掩盖 C3_FULL 的部分正向信息。

## 2. Context 是否真实重要

| Phase | strict reversal rate | B2 | CR FULL | CR-TR K64 |
|-|-|-|-|-|
| free_space | 0.01% | 0.0336724 | 0.0152697 | 0.019791 |
| pre_contact | 33.95% | 0.437274 | 0.364724 | 0.399912 |
| contact_onset | 32.64% | 0.400395 | 0.344096 | 0.359406 |
| post_contact | 27.78% | 0.361346 | 0.28628 | 0.315413 |

严格 reversal 主要出现在 contact phases。缺失 strata=['plate_push/free_space', 'stove_turn_on/free_space', 'wine_rack/free_space']；margin 从未放宽，也未制造标签。

Frozen C3 intervention: correct=0.229493, state+nominal shuffle=0.250881, history shuffle=0.245873, nominal-only shuffle=0.231674。状态与历史改变较大，nominal 单独改变很小，说明旧 C3 并未充分利用 nominal chunk。

## 3. C3 独立重选、CR 训练和 controls

独立 calibration 选择 CR family=`CR_C3_SHARED`；trust L=64；analysis-only gamma=0.2。C3 重选结果没有用 development/episodes40–49 选择。

| Method | Dev effect | Dev gain vs B2 | Hist exploratory | Fresh |
|-|-|-|-|-|
| B2 | 0.308172 | baseline | 0.271995 | 0.194957 |
| O_FULL | 0.0401798 | 86.96% | 0.0298603 | 0.0256242 |
| O_K64 | 0.137691 | 55.32% | 0.120583 | 0.0860105 |
| FROZEN_C3_FULL | 0.229512 | 25.52% | 0.204289 | 0.132187 |
| FROZEN_C3_K64 | 0.28315 | 8.12% | 0.268336 | 0.175505 |
| C3_RESELECT_FULL | 0.233777 | 24.14% | 0.204202 | 0.134519 |
| C3_RESELECT_KMEDOIDS64 | 0.265268 | 13.92% | 0.239777 | 0.161129 |
| CR_C3_FULL | 0.252592 | 18.04% | 0.222428 | 0.142797 |
| CR_C3_K64 | 0.273631 | 11.21% | 0.257837 | 0.170829 |
| CR_TR_C3_K64 | 0.273631 | 11.21% | 0.257837 | 0.170829 |
| ACTION_ONLY_TR_K64 | 0.276005 | 10.44% | 0.255082 | 0.168382 |
| SHUFFLED_EFFECT_TR_K64 | 0.284049 | 7.83% | 0.260265 | 0.172731 |

| Task | B2 | CR FULL | FULL gain | CR-TR K64 | TR gain |
|-|-|-|-|-|-|
| bowl_on_plate | 0.510531 | 0.447924 | 12.26% | 0.473782 | 7.20% |
| plate_push | 0.262825 | 0.213401 | 18.80% | 0.227925 | 13.28% |
| stove_turn_on | 0.258333 | 0.185655 | 28.13% | 0.207238 | 19.78% |
| wine_rack | 0.200999 | 0.163389 | 18.71% | 0.185578 | 7.67% |

所有 control 与选中 family 使用相同 architecture、parameter count、3 seeds、30 epochs 与 query batch 16。完整机制数值见 `MECHANISM_REVERSE_ENGINEERING.json`。

代码级反解：target 和 candidate 共用 `embed(context, residual)` 后做差；当网络采用最容易的加性解 `f(context)+g(action)` 时，`f(context)` 在距离中精确抵消。PROPOSED 的三 seed 平均 reversal loss 从 1.46607 到 1.46617（变化 0.000101431），ACTION_ONLY 最终为 1.46646；development joint reversal accuracy 仅 0.00421257，反而低于 frozen C3 的 0.0599482。与此同时 context-shuffled/no-reversal/shuffled-effect controls 没有系统性恶化。因此 B2-relative 提升来自静态 action/effect 排序，不是 context-reversal 机制。

## 4. Trust region 的作用

| Method | Effect | Action RMSE | Contact | Normalized utilization |
|-|-|-|-|-|
| CR_C3_FULL | 0.252592 | 0.018162 | 0.961751 | 0.658816 |
| CR_C3_K64 | 0.273631 | 0.0172738 | 0.950684 | 0.24388 |
| CR_TR_C3_K64 | 0.273631 | 0.0172738 | 0.950684 | 0.24388 |

Trust region 只在 executable atlas members 中筛选，不做 clipping、synthesis 或 pseudoinverse。Calibration 从 L=8 到 64 时 effect error 由 0.315333 降到 0.30327，而 action RMSE 由 0.0174483 升到 0.0196907；注册的 effect-first 规则因此选择 L=64（明确的 no-trust control）。也就是说，更强 action locality 的确缩短动作距离，但通过删除 consequence-nearest candidates 擦除了效果收益；最终方法并未获得真正的 trust-region 修复。

## 5. Gates 与 fresh confirmation

| Gate | Result | Primary gain |
|-|-|-|
| A oracle headroom | PASS | 55.32% |
| B learned metric | FAIL | 18.04% |
| C K=64 alphabet | FAIL | 11.21% |
| Fresh confirmation | FAIL | 12.38% |

Fresh primary paired difference (B2 error - method error)=0.0240326，95% CI=[0.0177267, 0.0306403]，10,000 episode-clustered bootstrap replicates。三 seed gains={'member_0': 0.10778895557120605, 'member_1': 0.09252554721022845, 'member_2': 0.09738504627404956}。

Fresh gate 仍失败：action RMSE 恶化 30.08%（阈值 20%），context-shuffled 保留 70.21% 的主收益（阈值 25%）；utilization=0.297076、contact drop=0.00195312、clipping=0。正 CI 只证明静态选择相关性可复现，不能挽救 context-specific 机制。

## 6. 11 个直接回答

| # | 问题 | 回答 |
|-|-|-|
| 1 | bank/metric/K64/C4 各贡献多少？ | 见第1节精确加性分解。 |
| 2 | Frozen C3 development gain 是否在历史探索集复现？ | Dev C3_FULL=0.229512；Hist C3_FULL=0.204289；两者均与各自 B2 分开报告。 |
| 3 | 真实 consequence ordering 是否 state-dependent？ | 接触阶段是；free-space 基本否，严格 reversal 缺失被保留。 |
| 4 | 模型是否使用 state/nominal/history？ | Frozen C3 对 state/history 有一定敏感性、对 nominal 很弱；新 CR 模型的 shuffle/action-only controls 表明其收益并不因果依赖这些 context。 |
| 5 | C3 独立 objective selection 是否改善？ | C3_RESELECT_FULL=0.233777 vs frozen C3_FULL=0.229512。 |
| 6 | CR training 是否改善 ordering？ | Gate B reversal gain=-0.0557356，完整 accuracy 见 context_reversal_evaluation.json。 |
| 7 | Trust region 是否减少动作偏差且保留 gain？ | Action degradation=22.52%；FULL gain retention=62.15%。 |
| 8 | FULL retrieval 在 K64 失败时是否仍工作？ | FULL=0.252592，K64=0.273631，TR=0.273631。 |
| 9 | 机制最终属于哪类？ | STATIC_EFFECT_METRIC_ONLY |
| 10 | 是否在 genuinely fresh states 确认？ | 执行了 FRESH_PERTURBED_STATE_CONFIRMATION；不是 new-episode claim。 |
| 11 | 是否可进入 small state-based BC？ | NO |

## Checkpoint hashes

| Arm | Member | Seed | SHA-256 |
|-|-|-|-|
| proposed | 0 | 13150417 | 33d328b16cf19f0552bbad9b23fe77ce4bd58ac7472e180258c66135262739ae |
| proposed | 1 | 13150429 | 21d980187c1d06c24a863ab0ac8ea9819b272d84a36af7b4dfaf3d2200dff38e |
| proposed | 2 | 13150443 | 9cb51eca8a95df77d809a8e788d61b9b38dbd9c6f3a9ea23403f6c455716e4b4 |
| ACTION_ONLY | 0 | 13150417 | 6e8d33463b10f55c2097fb865f2298e48aef08da7d99560b38146397ee3e3001 |
| ACTION_ONLY | 1 | 13150429 | 205c49662de158aba1d4ff9b63c601cb43bbcefb5853471df880c48e3e9a6803 |
| ACTION_ONLY | 2 | 13150443 | e55965a38f1507e6695f37cccb6e328f34545426292afb5ebf43f3c5b7ec9736 |
| CONTEXT_SHUFFLED | 0 | 13150417 | 2c6ab74ef464981d575830e926f14f53e77c2bec90c4a9327f773a11afb7796e |
| CONTEXT_SHUFFLED | 1 | 13150429 | 0cd974762cf1af0e2514f8b8438ebde60185d70e9481088d4b58946df371f666 |
| CONTEXT_SHUFFLED | 2 | 13150443 | aff7cd8e5fb028f3033d2c34b6f718bfbc8407ee09351a87a4000627ddc7515b |
| NOMINAL_SHUFFLED | 0 | 13150417 | c9631cc8ba0faf07fc836653ca134cbc4d87a8e33d40c0d391332695f5e9e72f |
| NOMINAL_SHUFFLED | 1 | 13150429 | 9e6cf4eb116d2afa4e7581423e4d032bd3ba5f2892f06e101db9934aeb2f1cc9 |
| NOMINAL_SHUFFLED | 2 | 13150443 | 363644d28fa0db619d99155ca25c7b88e75b442879a273fef6c87fc00811f118 |
| CONSEQUENCE_LABEL_SHUFFLED | 0 | 13150417 | ca5abb63477694b411ade914e9e6936aeb472bb406c0f9a8ef4c22dd995a9da0 |
| CONSEQUENCE_LABEL_SHUFFLED | 1 | 13150429 | 088f093e79ec4c2517794679de81b0fe39d9bcae40e9498e581c394e06317379 |
| CONSEQUENCE_LABEL_SHUFFLED | 2 | 13150443 | a404786729dcd369fd9211f22cc910b8978a7406560fca24b3c7eacf905f8466 |
| REVERSAL_LABEL_SHUFFLED | 0 | 13150417 | 5fa319b0cde114a15b9d9422734285ac6421a0e3363d1af9cd197784a703e283 |
| REVERSAL_LABEL_SHUFFLED | 1 | 13150429 | 2e99f9d19ba3a50ab835be1dcbdd01c75d761d8adbc958d44d11471c05c10242 |
| REVERSAL_LABEL_SHUFFLED | 2 | 13150443 | 8edd04c3de24869b4a0749b9973d8298421d1a4b1f6dfedebd755527a6229d3e |
| NO_REVERSAL_LOSS | 0 | 13150417 | 603fe9e8536c6ad7a99f32cdcf7831bc344fa9c2cef539de520661aa31e97575 |
| NO_REVERSAL_LOSS | 1 | 13150429 | c14822130ed72ced749dd93af2cdd49d8361b87cf05e2e4f964ad0dd8b2e6999 |
| NO_REVERSAL_LOSS | 2 | 13150443 | e2a87923c1211edbd53333cf5fed4eabbb7aede19c045f96efeee143b5dc57a4 |

## 局限与下一步

- episodes 40–49 的旧 snapshot 仍只是 historical exploratory；fresh evidence 来自同些 source episodes 的未用 timestep 加预注册小关节扰动，因此不是新 episode 证据。
- State-based metric audit 不是 VLA、不是 policy evaluation，也不能声称 paper readiness。
- 下一推荐实验仅是严格 paired 的静态度量复现：优先取得 episode ID >=50 的成功轨迹，在完全相同的 K64 bank 上直接比较 frozen C3、CR、action-only、context-shuffled 与 consequence-label-shuffled；不改变架构、不训练 policy。
- 本轮在 Stage 4 停止；不得自动启动 ACT、Diffusion Policy、SmolVLA 或 pi0.5。

## Exact final disposition

`STATIC_EFFECT_METRIC_ONLY`

## 发布信息

- GitHub main commit: [`ac861eb60f83c72ac4785d8d901356434eded2ec`](https://github.com/mikasaTu/R13-P15-Consequence-Adaptive-Action-Alphabets/commit/ac861eb60f83c72ac4785d8d901356434eded2ec)
- 实验目录: [`experiments/r13_p15_cr_trca/stage4/`](https://github.com/mikasaTu/R13-P15-Consequence-Adaptive-Action-Alphabets/tree/main/experiments/r13_p15_cr_trca/stage4)
- 冻结分支: [`r13-p15-stage4-c3-context-trust-region`](https://github.com/mikasaTu/R13-P15-Consequence-Adaptive-Action-Alphabets/tree/r13-p15-stage4-c3-context-trust-region)
- 本地测试: `38 passed`
- `STAGE4_REPORT.md` SHA-256: `d55b996ba985ed49085720b9995781ec5c9880ea2971fcfbd79e197cd1cbc80b`
