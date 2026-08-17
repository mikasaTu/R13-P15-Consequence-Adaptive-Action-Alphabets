<title>实验报告</title>

# 实验报告

## 中文结论摘要

最终处置：**`ORACLE_ONLY_NO_LEARNABLE_RANKER`**。

- Stage 3 的严格支持 oracle 仍然有价值：Gate A 提升 55.320%，4/4 任务、3/3 接触敏感任务改善。
- 学习排序没有通过：C4 相对 `C3_NC_BIENCODER` 的 oracle regret 变化为 -31.176%，NDCG@16 变化 -0.17468，Recall@8=0.13542，所以 Gate B 失败。
- 主方法 C5 没有保住 C3 的收益：相对 `B2_current_contact_kmeans` 的真实物理效应误差变化为 -21.675%，K=64 利用率 0.11399，action RMSE 退化 67.971%；Gate C 失败。
- episodes 40–49 已按要求全部执行，但证据标签是 `FORCED_EXPLORATORY_HOLDOUT`，不是 untouched confirmation，不能解锁 BC。
- 没有训练 ACT、Diffusion Policy、SmolVLA、pi0.5 或任何策略；PAI 作业数为 0；预测器训练仅使用一张本地 A800。

## 机理反解

代码与控制实验共同指向同一条下降链：C3 bi-encoder 本身能学到较强的效果几何，并把 development 真实误差从 B2 的 0.30817 降到 0.28315。但是 C5 只用 C3 做 K=64 FPS 覆盖，最终选择完全由表现较差的 C4 pair ranker 决定；C4 把 C3 的排序收益逆转，K=64 压缩再带来较小的二次损失。nominal/state/history/label shuffle 没有摧毁正收益，因为 C4/C5 相对 C3 本来就是负收益。详细逐代码路径、support-family 分解和限制见下方完整报告及 `MECHANISM_REVERSE_AUDIT.md`。

## 代码与制品

GitHub 主分支：[https://github.com/mikasaTu/R13-P15-Consequence-Adaptive-Action-Alphabets/tree/main](https://github.com/mikasaTu/R13-P15-Consequence-Adaptive-Action-Alphabets/tree/main)

本页以下内容是仓库内冻结的 `STAGE3_REPORT.md` 完整正文。

---

# R13-P15 Stage 3 — NCER-AA experiment report

## Exact disposition

`ORACLE_ONLY_NO_LEARNABLE_RANKER`

`GO_TO_SMALL_BC` is unavailable. Stage 1 remains `REJECT_CORE_HYPOTHESIS`, Stage 1.5 remains `REJECT_P15_FAMILY`, and Stage 2 remains `ORACLE_ONLY_NO_DEPLOYABLE_MODEL`. No policy or VLA was trained.

## Historical evidence (read-only)

| Stage | Frozen disposition | Published commit |
|-|-|-|
| Stage 1 | REJECT_CORE_HYPOTHESIS | 434427af0f8adc844851c27cfc050b2c9c6752dc |
| Stage 1.5 | REJECT_P15_FAMILY | 76433b6e58196ceeedc4ad005a1110ea8e343ae2 |
| Stage 2 | ORACLE_ONLY_NO_DEPLOYABLE_MODEL | 74c98979910a3831d0abeb8d13111a7c9294b067 |

These artifacts are inputs only. Stage 3 neither rewrites nor relabels them.

## Stage 3 evidence integrity and scope

- LIBERO tasks: bowl_on_plate, plate_push, stove_turn_on, wine_rack; Panda OSC_POSE at 20 Hz; H=4; three settle steps.
- Episodes: historical 0–15, train 16–31, calibration 32–35, development 36–39, holdout 40–49. All demonstrations were successful; 544 four-phase snapshots were frozen.
- Fresh support split overlap is zero; worst cross-split absolute cosine is 0.89737 (limit 0.90); target-bank exact matches are zero; clipping validity passed.
- The user-required holdout was fully executed after settings were frozen even if gates failed. It is labeled `FORCED_EXPLORATORY_HOLDOUT`, not untouched confirmation.
- Pre-result incident `stage3-pre-result-confirmation-replay-001` executed one fixed confirmation support direction twice per snapshot for replay checking before gates. No method/result was computed then, but the literal untouched rule was broken; the incident is preserved in `PRE_RESULT_PROTOCOL_INCIDENT.json`.
- Local execution only: 1 visible training GPU (NVIDIA A800-SXM4-80GB); PAI jobs=0; policy training=false.

## Exact bindings

| Component | Commit/hash |
|-|-|
| Stage 3 input repository | 74c98979910a3831d0abeb8d13111a7c9294b067 |
| Published Stage 2 | 74c98979910a3831d0abeb8d13111a7c9294b067 |
| LIBERO upstream | 8f1084e3132a39270c3a13ebe37270a43ece2a01 |
| LIBERO source tree SHA-256 | e9197ca08fe4d7325f561fc40d7425167830253e0f0fceb1af2663b23292f71f |
| Stage 2 M=256 bank SHA-256 | d41f0dc748866cae3ef151d9f16e39789485d6e633a0a88f62fe4c570661600b |
| Stage 3 code used for report | 04e95630aee87f356940b2522f2025faa8c7c209 |

The simulator and analysis package versions are frozen in `execution_environment.json`; all 256 reused training states and all 544 final branch states are hash-checked in the validation JSON files.

## Reused training and calibration evidence

- Reused Stage 2 support: episodes 16–31 after exact simulator/hash verification. Missing candidate-bank outcomes for 16–23 were collected; 24–31 were reused.
- Training states: 256; calibration states: 64; training pairs: 1010096 rows.
- Calibration alone selected B5 neighbors=9, bandwidth=2.0; strongest deployable comparator=`B2_current_contact_kmeans`; Gate-B comparator=`C3_NC_BIENCODER`; selected learned ranker=`C4_NC_PAIR_RANKER`; ranking objective index=1.
- Candidate-order invariance passed on calibration: True.

## Development gates (episodes 36–39)

| Gate | Result | Failure disposition if first failure |
|-|-|-|
| A | PASS | REJECT_CONSEQUENCE_EQUIVALENCE_ON_STRICT_SUPPORT |
| B | FAIL | ORACLE_ONLY_NO_LEARNABLE_RANKER |
| C | FAIL | LEARNABLE_RETRIEVAL_BUT_ALPHABET_COMPRESSION_FAILED |

- Gate A: oracle gain=55.320%, tasks improved=4/4, contact tasks=3/3.
- Gate B: regret gain=-31.176%, NDCG@16 gain=-0.17468, Recall@8=0.13542, tasks=0/4, contact tasks=0/3, exact permutation=True.
- Gate C: realized gain=-21.675%, oracle gap closed=-39.182%, utilization=0.11399, clipping=0, action-RMSE degradation=67.971%, contact drop=0.0068359.

### Oracle-only result

Gate A is a simulator-outcome upper bound, not a deployable method. The K=64 true-effect atlas improves all four tasks and all three contact-sensitive tasks, but it consumes candidate consequences unavailable at deployment.

## K=64 alphabet and deployable results

| Method | Effect error | Action RMSE | Contact preserved | Normalized utilization | Clipping |
|-|-|-|-|-|-|
| B2_current_contact_kmeans | 0.30817 | 0.014098 | 0.95752 | 0.38855 | 0 |
| C3_NC_BIENCODER | 0.28315 | 0.022903 | 0.95166 | 0.24443 | 0 |
| C4_NC_PAIR_RANKER | 0.3589 | 0.022103 | 0.95378 | 0.19548 | 0 |
| C5_NCER_AA | 0.37497 | 0.023681 | 0.95068 | 0.11399 | 0 |
| C6_SOFT_MIXTURE_NCER_AA | 0.37223 | 0.020298 | 0.95133 | 0.25735 | 0 |

### Realized K=64 effect by task

| Task | B2_current_contact_kmeans | C5 | C5 improvement | True-effect K64 oracle |
|-|-|-|-|-|
| bowl_on_plate | 0.51053 | 0.622 | -21.834% | 0.23462 |
| plate_push | 0.26282 | 0.3239 | -23.238% | 0.14234 |
| stove_turn_on | 0.25833 | 0.31418 | -21.618% | 0.088736 |
| wine_rack | 0.201 | 0.2398 | -19.303% | 0.085066 |

### Realized K=64 effect by phase

| Phase | B2_current_contact_kmeans | C5 | Oracle K64 | C5 contact preservation |
|-|-|-|-|-|
| free_space | 0.033672 | 0.055894 | 0.022907 | 1 |
| pre_contact | 0.43727 | 0.50086 | 0.20716 | 0.81706 |
| contact_onset | 0.4004 | 0.4938 | 0.17689 | 0.98568 |
| post_contact | 0.36135 | 0.44932 | 0.14381 | 1 |

Pooled development realized error is 0.30817 for `B2_current_contact_kmeans` and 0.37497 for C5. C5 action RMSE=0.023681, contact preservation=0.95068, normalized utilization=0.11399, code perplexity=7.5058, clipping=0.

## Privileged diagnostic upper bounds

| Diagnostic | Effect error | Action RMSE | Contact preserved | Normalized utilization |
|-|-|-|-|-|
| B2_PRIV_hard_phase_kmeans | 0.30275 | 0.014034 | 0.95882 | 0.38644 |
| C0_stage2_ncea_reproduction | 0.33498 | 0.022663 | 0.95182 | 0.21597 |

`B2_PRIV` and C0 consume the frozen demonstration hard-phase construction and are diagnostics only. Neither is the deployable NCER-AA result.

## Learned prediction and retrieval

| Method | Oracle regret | Spearman | Kendall | NDCG@16 | Recall@8 | ms/target |
|-|-|-|-|-|-|-|
| C3_NC_BIENCODER | 0.24297 | 0.66487 | 0.53015 | 0.61958 | 0.37815 | 4.3625 |
| C1_NC_VECTOR | 0.3341 | 0.29998 | 0.20639 | 0.46597 | 0.15163 | 0.077703 |
| C2_NC_TEMPORAL_VECTOR | 0.32655 | 0.39213 | 0.27767 | 0.47129 | 0.15928 | 0.077703 |
| C3_NC_BIENCODER | 0.24297 | 0.66487 | 0.53015 | 0.61958 | 0.37815 | 4.3625 |
| C4_NC_PAIR_RANKER | 0.31872 | 0.32664 | 0.23675 | 0.4449 | 0.13542 | 4.3625 |
| C5_NCER_AA | 0.33479 | 0.32664 | 0.23675 | 0.4449 | 0.13542 | 4.3625 |
| C6_SOFT_MIXTURE_NCER_AA | 0.33205 | 0.37159 | 0.26229 | 0.49267 | 0.1613 | 4.3625 |

`predictor_metrics.csv` additionally reports balanced vector error/contact accuracy per task, phase and support family. `retrieval_metrics.csv` reports pairwise accuracy, Spearman, Kendall tau, NDCG@16, Recall@1/8, regret and latency for every learned method and direction family.

### Vector predictor breakdown

| Method | Level | Slice | Normalized RMSE | Balanced prediction error | Contact accuracy |
|-|-|-|-|-|-|
| C0_stage2_ncea_reproduction | pooled | ALL | 1.0435 | 0.37802 | 0.94759 |
| C0_stage2_ncea_reproduction | task | bowl_on_plate | 1.6415 | 0.57023 | 0.89779 |
| C0_stage2_ncea_reproduction | task | plate_push | 0.8179 | 0.28753 | 0.98503 |
| C0_stage2_ncea_reproduction | task | stove_turn_on | 0.67028 | 0.29284 | 0.97852 |
| C0_stage2_ncea_reproduction | task | wine_rack | 1.0445 | 0.3615 | 0.92904 |
| C0_stage2_ncea_reproduction | phase | free_space | 0.25391 | 0.033657 | 1 |
| C0_stage2_ncea_reproduction | phase | pre_contact | 1.302 | 0.48498 | 0.80469 |
| C0_stage2_ncea_reproduction | phase | contact_onset | 1.1703 | 0.44869 | 0.98568 |
| C0_stage2_ncea_reproduction | phase | post_contact | 1.448 | 0.54477 | 1 |
| C0_stage2_ncea_reproduction | direction_family | 0 | 0.94502 | 0.34903 | 0.94531 |
| C0_stage2_ncea_reproduction | direction_family | 1 | 1.274 | 0.45419 | 0.94092 |
| C0_stage2_ncea_reproduction | direction_family | 2 | 0.91163 | 0.33085 | 0.95654 |
| C1_NC_VECTOR | pooled | ALL | 1.8199 | 0.45387 | 0.93213 |
| C1_NC_VECTOR | task | bowl_on_plate | 2.1666 | 0.63091 | 0.89844 |
| C1_NC_VECTOR | task | plate_push | 1.8049 | 0.29731 | 0.98503 |
| C1_NC_VECTOR | task | stove_turn_on | 1.3116 | 0.31005 | 0.97852 |
| C1_NC_VECTOR | task | wine_rack | 1.9964 | 0.5772 | 0.86654 |
| C1_NC_VECTOR | phase | free_space | 0.56274 | 0.054698 | 1 |
| C1_NC_VECTOR | phase | pre_contact | 1.64 | 0.49714 | 0.80534 |
| C1_NC_VECTOR | phase | contact_onset | 2.1397 | 0.47336 | 0.92318 |
| C1_NC_VECTOR | phase | post_contact | 2.9371 | 0.79027 | 1 |
| C1_NC_VECTOR | direction_family | 0 | 1.7499 | 0.42243 | 0.9292 |
| C1_NC_VECTOR | direction_family | 1 | 1.9795 | 0.53642 | 0.92676 |
| C1_NC_VECTOR | direction_family | 2 | 1.7303 | 0.40275 | 0.94043 |
| C2_NC_TEMPORAL_VECTOR | pooled | ALL | 1.0518 | 0.3585 | 0.94775 |
| C2_NC_TEMPORAL_VECTOR | task | bowl_on_plate | 1.6611 | 0.56164 | 0.89844 |
| C2_NC_TEMPORAL_VECTOR | task | plate_push | 0.84653 | 0.2863 | 0.98503 |
| C2_NC_TEMPORAL_VECTOR | task | stove_turn_on | 0.71276 | 0.29626 | 0.97852 |
| C2_NC_TEMPORAL_VECTOR | task | wine_rack | 0.98677 | 0.28982 | 0.92904 |
| C2_NC_TEMPORAL_VECTOR | phase | free_space | 0.32391 | 0.039897 | 1 |
| C2_NC_TEMPORAL_VECTOR | phase | pre_contact | 1.3353 | 0.48833 | 0.80534 |
| C2_NC_TEMPORAL_VECTOR | phase | contact_onset | 1.1685 | 0.4298 | 0.98568 |
| C2_NC_TEMPORAL_VECTOR | phase | post_contact | 1.3794 | 0.47599 | 1 |
| C2_NC_TEMPORAL_VECTOR | direction_family | 0 | 0.95383 | 0.32917 | 0.94482 |
| C2_NC_TEMPORAL_VECTOR | direction_family | 1 | 1.2789 | 0.43572 | 0.94238 |
| C2_NC_TEMPORAL_VECTOR | direction_family | 2 | 0.92258 | 0.31062 | 0.95605 |

### Learned retrieval breakdown

| Method | Level | Slice | Oracle regret | NDCG@16 | Recall@8 |
|-|-|-|-|-|-|
| C3_NC_BIENCODER | pooled | ALL | 0.24297 | 0.61958 | 0.37815 |
| C3_NC_BIENCODER | task | bowl_on_plate | 0.43526 | 0.46549 | 0.31486 |
| C3_NC_BIENCODER | task | plate_push | 0.1972 | 0.59452 | 0.35848 |
| C3_NC_BIENCODER | task | stove_turn_on | 0.16553 | 0.70469 | 0.42065 |
| C3_NC_BIENCODER | task | wine_rack | 0.17389 | 0.71361 | 0.41862 |
| C3_NC_BIENCODER | phase | free_space | 0.014666 | 0.98927 | 0.76611 |
| C3_NC_BIENCODER | phase | pre_contact | 0.37644 | 0.45192 | 0.21647 |
| C3_NC_BIENCODER | phase | contact_onset | 0.33779 | 0.46111 | 0.22298 |
| C3_NC_BIENCODER | phase | post_contact | 0.24298 | 0.57602 | 0.30705 |
| C3_NC_BIENCODER | direction_family | 0 | 0.25096 | 0.61225 | 0.35718 |
| C3_NC_BIENCODER | direction_family | 1 | 0.23985 | 0.6244 | 0.41162 |
| C3_NC_BIENCODER | direction_family | 2 | 0.2381 | 0.62208 | 0.36566 |
| C4_NC_PAIR_RANKER | pooled | ALL | 0.31872 | 0.4449 | 0.13542 |
| C4_NC_PAIR_RANKER | task | bowl_on_plate | 0.52264 | 0.36224 | 0.1403 |
| C4_NC_PAIR_RANKER | task | plate_push | 0.29588 | 0.38261 | 0.11133 |
| C4_NC_PAIR_RANKER | task | stove_turn_on | 0.26819 | 0.46719 | 0.14111 |
| C4_NC_PAIR_RANKER | task | wine_rack | 0.18816 | 0.56756 | 0.14893 |
| C4_NC_PAIR_RANKER | phase | free_space | 0.032493 | 0.86617 | 0.26864 |
| C4_NC_PAIR_RANKER | phase | pre_contact | 0.39349 | 0.38749 | 0.15169 |
| C4_NC_PAIR_RANKER | phase | contact_onset | 0.43997 | 0.27878 | 0.077881 |
| C4_NC_PAIR_RANKER | phase | post_contact | 0.40893 | 0.24715 | 0.043457 |
| C4_NC_PAIR_RANKER | direction_family | 0 | 0.29881 | 0.45152 | 0.12634 |
| C4_NC_PAIR_RANKER | direction_family | 1 | 0.36052 | 0.43181 | 0.16211 |
| C4_NC_PAIR_RANKER | direction_family | 2 | 0.29682 | 0.45136 | 0.1178 |
| C5_NCER_AA | pooled | ALL | 0.33479 | 0.4449 | 0.13542 |
| C5_NCER_AA | task | bowl_on_plate | 0.54219 | 0.36224 | 0.1403 |
| C5_NCER_AA | task | plate_push | 0.29588 | 0.38261 | 0.11133 |
| C5_NCER_AA | task | stove_turn_on | 0.28962 | 0.46719 | 0.14111 |
| C5_NCER_AA | task | wine_rack | 0.21147 | 0.56756 | 0.14893 |
| C5_NCER_AA | phase | free_space | 0.046438 | 0.86617 | 0.26864 |
| C5_NCER_AA | phase | pre_contact | 0.43731 | 0.38749 | 0.15169 |
| C5_NCER_AA | phase | contact_onset | 0.44648 | 0.27878 | 0.077881 |
| C5_NCER_AA | phase | post_contact | 0.40893 | 0.24715 | 0.043457 |
| C5_NCER_AA | direction_family | 0 | 0.3192 | 0.45152 | 0.12634 |
| C5_NCER_AA | direction_family | 1 | 0.37499 | 0.43181 | 0.16211 |
| C5_NCER_AA | direction_family | 2 | 0.31018 | 0.45136 | 0.1178 |

## Mechanism controls

| Control | Realized effect | Oracle regret | NDCG@16 | Order invariant | Index mismatches |
|-|-|-|-|-|-|
| action_only_pair_ranker | 0.37632 | 0.33614 | 0.48142 |  |  |
| candidate_order_permutation | 0.37497 | NA | NA | 1 | 0 |
| consequence_labels_shuffled | 0.37631 | 0.33613 | 0.42382 |  |  |
| history_shuffled | 0.36925 | 0.32908 | 0.47964 |  |  |
| joint_state_nominal_shuffled_within_task | 0.3802 | 0.34002 | 0.47644 |  |  |
| no_nominal_action | 0.37747 | 0.33729 | 0.44724 |  |  |
| nominal_action_shuffled_within_task | 0.37434 | 0.33416 | 0.44495 |  |  |
| soft_routing_labels_shuffled | 0.37805 | 0.33787 | 0.48562 |  |  |
| state_shuffled_within_task | 0.36332 | 0.32314 | 0.50615 |  |  |

Symmetry error and self-distance error are exactly zero by architecture and unit test. Candidate permutation carries immutable bank IDs through FPS and reranking. The full code-to-result explanation, including improvements, degradations and confounds, is in `MECHANISM_REVERSE_AUDIT.md`.

## Holdout episodes 40–49 (not untouched confirmation)

Pooled realized error: `B2_current_contact_kmeans`=0.27199; C5=0.33429; relative gain=-22.904%. The paired difference is baseline minus C5.

| Cluster | Paired difference | 95% bootstrap CI |
|-|-|-|
| pooled | -0.062299 | [-0.075881, -0.049453] |
| bowl_on_plate | -0.088423 | [-0.12657, -0.055737] |
| plate_push | -0.060527 | [-0.07183, -0.048782] |
| stove_turn_on | -0.043293 | [-0.067049, -0.017774] |
| wine_rack | -0.056953 | [-0.087932, -0.027033] |

Replicates=10000. Counterfactual statistical/mechanism GO criteria passed=False; confirmation integrity criterion passed=false; BC remains locked.

## K sensitivity

K=32 and K=128 were evaluated only after `ORACLE_ONLY_NO_LEARNABLE_RANKER` was frozen. Results are in `k_sensitivity.csv`; they cannot change the primary K=64 disposition.

| K | Method | Effect error | Action RMSE | Contact preserved | Normalized utilization |
|-|-|-|-|-|-|
| 32 | C5_NCER_AA_K32 | 0.35238 | 0.023707 | 0.94675 | 0.1408 |
| 32 | O_true_effect_K32 | 0.15937 | 0.023924 | 0.96949 | 0.32351 |
| 128 | C5_NCER_AA_K128 | 0.33957 | 0.023059 | 0.94731 | 0.050689 |
| 128 | O_true_effect_K128 | 0.10062 | 0.02295 | 0.99298 | 0.24059 |

## Direct answers

| # | Question | Answer | Basis |
|-|-|-|-|
| 1 | Nominal a0 materially improves prediction? | NO | Development nominal-shuffle gain retention=1e+09 |
| 2 | Pair/listwise ranking beats vector regression? | NO | C4 regret is slightly lower, but NDCG/Recall are worse and C3 is decisively better. |
| 3 | Short observable history necessary? | NO | History bundle includes values and availability masks. |
| 4 | Soft mixture beats one global model? | MARGINALLY YES, NOT SUFFICIENT | C6 is slightly better than C5 but remains worse than B2 and C3. |
| 5 | Learned retrieval recovers meaningful oracle fraction? | NO | Frozen Gate B. |
| 6 | Gain survives K=64 compression? | NO | Frozen Gate C. |
| 7 | State/nominal/label shuffles destroy gain? | NO | {"consequence_labels_shuffled": 1000000000.0, "joint_state_nominal_shuffled_within_task": 1000000000.0, "nominal_action_shuffled_within_task": 1000000000.0, "state_shuffled_within_task": 1000000000.0} |
| 8 | Confirmed on episodes 40–49? | NO | Executed as FORCED_EXPLORATORY_HOLDOUT; strict untouched confirmation is false. |
| 9 | Ready for small state-based BC? | NO | go_to_small_bc_available=false; final disposition=ORACLE_ONLY_NO_LEARNABLE_RANKER |

## Negative runs and limitations

- All failed gates and all negative control runs remain reported; no gate stopped later experiments.
- One initial no-result local training process was interrupted before development inspection to make state/history shuffle controls permute their masks and contact indicator with the semantic bundle. It was rerun from scratch; no model-selection result was read from that interrupted process.
- Constant-score predictors make rank correlation undefined; the frozen metric implementation records Spearman/Kendall as zero. A diagnostic warning in the first development run led only to an explicit constant-range check with identical numeric semantics.
- Controls use one member versus three-member primary ensembles, so small ablation differences are not cleanly attributable to one input.
- C0 is a privileged hard-phase reproduction and only a conservative diagnostic/comparator; C5/C6 never consume the phase label.
- Raw branch arrays are too large for ordinary Git and remain at the bound scratch paths; repository JSON records their byte sizes and SHA-256 markers, while all row-level decoded-action results are committed as CSV.

## Recommended next experiment

Do not start BC from this audit. The bounded next validation is a preregistered C3-only decoder diagnostic on genuinely untouched episodes: retain the already implemented bi-encoder distance through final action selection, compare it directly with B2 and the true-effect oracle, and separately audit the existing C4 hard-negative/objective path on suffix-localized support. This tests the localized C3-to-C4 reversal without introducing a new policy or claiming a new idea. Do not launch ACT, Diffusion Policy, SmolVLA or pi0.5 automatically.
