<title>实验报告</title>

# R13-P15-v2 Stage 1：LIBERO CAAA-v2 初步验证

[源 Idea：Consequence-Adaptive Action Alphabets](https://icnbwz7kd1ui.feishu.cn/wiki/BftiwVGSbijquakZvx4cQQEsnve)

## 结论

**FINAL_DISPOSITION: REJECT_CORE_HYPOTHESIS**

**科学问题的答案：否。**在这组冻结的 LIBERO-Goal Stage 1 实验中，局部 physical-consequence pullback metric 没有比 Euclidean、covariance Mahalanobis、global/phase-conditioned k-means、PCA+k-means 或旧 diagonal sensitivity 更好地组织 H=4 action chunks；它在真实解码动作回放中的 settled physical-effect error 显著更高。

- 校准集冻结的最强 baseline：**covariance_mahalanobis**。
- K=64 pooled test：baseline error 73,693.26；CAAA-v2 error 102,892.55。
- 相对 improvement 为 **-39.62%**，episode-clustered 95% CI 为 **[-168.41%, -5.50%]**；整个区间位于 0 以下。
- 四个任务的点估计全部为负；CAAA-v2 没有在任何任务上优于冻结 baseline。
- CAAA-v2 action reconstruction error 0.7747，baseline 为 0.0949，约高 8.16 倍；K=64 codebook utilization 仅 7.42%，dead-code ratio 92.58%。
- random-SPD 的 error 104,058.81，与 CAAA-v2 的 102,892.55 接近，说明任意 state-dependent 拉伸能复现其伤害；permuted-J 更差，为 283,447.02。
- 因此不进入 small BC，不启动 ACT、Diffusion Policy、SmolVLA、π0.5、DINO-WM 或任何 policy training。

## 1. 范围与 LIBERO 适配

原计划写的是 ManiSkill3，但本次按用户明确指示改用标准 LIBERO；这是对原 hard scope 的授权覆盖。仅做机制级 oracle Stage 1，不做策略训练。开发机只做 CPU/单 GPU smoke；正式分支采集、真实量化回放与汇总在 PAI 上完成。

### 冻结任务

- **put_the_bowl_on_the_plate**（bowl_on_plate）：低约束 pick-place control。
- **push_the_plate_to_the_front_of_the_stove**（plate_push）：持续滑动接触。
- **turn_on_the_stove**（stove_turn_on）：小尺度 articulated contact。
- **put_the_wine_bottle_on_the_rack**（wine_rack）：精密定向放置。

### 冻结实验设计

- 标准 libero_goal；Panda OSC_POSE；20 Hz；统一 7D normalized action semantics；gripper command 原样复制。
- 每任务 16 条官方成功 demo；episode 0–7 train、8–11 calibration、12–15 test。
- 每 episode 四个 phase snapshot：free_space、pre_contact、contact_onset、post_contact；共 256 个 snapshot。
- H=4；连续 pose chunk 维度 24；24 个 deterministic orthonormal directions；antithetic signs；normalized radii 0.05 与 0.10。
- 每 snapshot 1 个原动作分支加 96 个扰动分支，共 97；正式 branch 总数 24,832。
- 立即后果在 H=4 后测量；settled 后果再执行 3 个 zero-delta-pose step，并保持最后 gripper command。
- 46 维 task-generic consequence schema；不可用 task dimensions 以 mask 处理。state_dict 只用于 oracle labels / task state，不用于生成动作。
- 局部 metric：$G=J^{\top} W J + \lambda I$；J 由 train data 的 local ridge 拟合，所有 ridge/SVD/regularization/PCA 选择只使用 calibration。
- 比较方法：Euclidean farthest、covariance Mahalanobis、global k-means、phase-conditioned k-means、PCA+k-means、old diagonal sensitivity、random SPD、permuted-J、full CAAA-v2。
- K=64 为唯一 primary；K=32 与 128 仅作冻结 sensitivity analysis。

## 2. 环境、提交与数据哈希

- 项目 Git commit：**34995e8e7c3069b22785ad04536f0d429e75c0fc**
- 项目 Git tree：**ad6fa59b782f63624ee3ccef8e880a2398669ce8**
- 项目 source-tree SHA-256：**df66a9429fe2a36cbca2947b0bbdf7e1dfee80f514a0160ce22be986ea0ff3da**
- Stage 1 launcher SHA-256：**569d031517a581a080c3a68a46a67df76432318d5c9e35edb1983e7f3c617e6a**
- PAI registry wrapper SHA-256：**fb6e747b96693d8082a2ae9c87909299a45aee882d0085ee794ba243240ff804**
- LIBERO upstream commit：**8f1084e3132a39270c3a13ebe37270a43ece2a01**
- LIBERO source-tree SHA-256：**e9197ca08fe4d7325f561fc40d7425167830253e0f0fceb1af2663b23292f71f**
- Python 3.8.13；MuJoCo 2.3.7；robosuite 1.4.0；PyTorch 1.11.0+cu113；CUDA build 11.3。

### 官方 demo SHA-256

- bowl_on_plate：e69528b0cf10dfc59b20698e12ec2affc03f3887309034d3eb74cac3ec929406
- plate_push：36b4e1bced49d2f4ff6b2fce6b1596a63978e14199e2513cd0df71e127bf47a6
- stove_turn_on：387fc10747696b80dea6ed8d7f2beaa162bf92ae11750b241073cbd33aac73d5
- wine_rack：f9092aa70734fc4083e97fc58c3ba25f87c614d18326182ddc7a455f0ab4da2e

### 正式 PAI run

- 成功 JobId：**dlc1wxel8qjf7ck8**；run_id：r13p15-caaa-v2-stage1-20260812-f。
- 终态：Succeeded；创建 2026-08-12 15:51:22 UTC；运行 15:53:22–16:07:11 UTC；Duration 949 s。
- 专属 1 worker、8×A800、92 CPU、1600 GiB memory/shared memory；AIMaster 关闭，自动 fault tolerance 关闭，平台/应用重启 0。
- 为了遵守“至多一个本地 GPU”的执行语义，正式 launcher 只让 GPU0 对四个 CPU MuJoCo 并行进程可见；没有多卡训练。
- 没有 PAI probe、没有浏览器提交、没有 W&B、没有 HTML。
- first-work 证据在全部 256 个 branch shard 和四任务 manifest 绑定当前 run 后才写入；两个 predecessor 均已终态，active cleanup targets 明确记录为 0。

## 3. Snapshot restore 与 deterministic branching

**正式 gate：PASS。**256/256 个 replay tests 通过，0 个失败；tolerance 为 1e-12。A/A、A/B/A、B/A/B 共 4,608 个数值比较的最大差异为 **0.0**。

### 所有 replay 失败与修复

- 首次本地 smoke 在 bowl_on_plate / episode 0 / free_space 失败：final-state max |Δ|=0.0122775，immediate consequence max |Δ|=0.0119254，settled max |Δ|=0.00229436。
- 根因：Panda gripper.current_action 会跨 MuJoCo substeps 积分，但 flattened simulator state 没有包含它。
- 修复：snapshot/restore 增加 gripper action history、MuJoCo ctrl、qacc_warmstart 及 solver auxiliary arrays。修复后本地与正式 A/A、A/B/A、B/A/B 全部精确为零。
- 正式 replay failed_tests 数组为 []。

### PAI 运行事件

- dlc1ngywaybfd122：在任何分支采样前失败；干净 HOME 缺少 LIBERO_CONFIG_PATH，LIBERO 非交互导入触发 EOF。修复为冻结的显式 config。
- dlcnyj30quijfvx9：完成 24,832 个 branch 和 69,120 行量化回放后，在 finalize 因漏定义 BASELINE_METHODS 触发 NameError；没有生成伪结论。
- dlc1wxel8qjf7ck8：在修复提交上重新完成 256 replay tests，重建分析与 plans；128/128 量化 shards 均通过 payload hash 和当前 plan arrays 逐数组精确一致性检查，随后正式 finalize 成功。

## 4. Calibration-only 选择与局部线性

- train perturbation samples：6,144；calibration samples：3,072。
- ridge：0.001；singular cutoff：0.0001；metric regularization：1e-8；covariance regularization：1e-8；PCA rank：12。
- 校准集 K=64 settled errors：covariance 48,716.51；PCA+k-means 56,397.67；phase-conditioned k-means 74,594.38；Euclidean 76,684.95；global k-means 80,381.11；old diagonal sensitivity 138,561.79。由此冻结 covariance Mahalanobis 为 test baseline。

### 每任务、每 phase 的局部诊断中位数

每项依次为 R² / normalized RMSE / effective rank / condition number / metric-to-consequence Spearman。

### bowl_on_plate

- free_space：0.93868 / 0.16899 / 3.5704 / 4,233.5 / 0.99696
- pre_contact：0.56888 / 0.60686 / 1.9374 / 7,633.0 / 0.53541
- contact_onset：0.53805 / 0.68639 / 1.5793 / 6,621.1 / 0.53454
- post_contact：0.02393 / 1.3270 / 1.0000 / 6,431.9 / 0.04693

### plate_push

- free_space：0.89339 / 0.17814 / 3.5351 / 4,151.8 / 0.99555
- pre_contact：0.74389 / 0.35112 / 1.6225 / 6,669.1 / 0.86270
- contact_onset：0.88463 / 0.21972 / 1.4945 / 8,058.5 / 0.96730
- post_contact：0.67106 / 0.36866 / 3.0148 / 7,665.7 / 0.88964

### stove_turn_on

- free_space：0.94045 / 0.16545 / 2.0903 / 11.715 / 0.99718
- pre_contact：0.57970 / 0.86752 / 1.6343 / 4,526.7 / 0.47274
- contact_onset：0.61869 / 0.68906 / 1.2622 / 1,873.4 / 0.56919
- post_contact：0.76405 / 0.48518 / 1.0853 / 2,513.7 / 0.79557

### wine_rack

- free_space：0.97180 / 0.16713 / 2.3774 / 10.739 / 0.99815
- pre_contact：0.11951 / 0.81056 / 1.3090 / 7,219.3 / 0.37410
- contact_onset：0.46319 / 0.63220 / 1.1909 / 6,718.1 / 0.67576
- post_contact：0.40933 / 0.97400 / 1.2571 / 7,734.1 / 0.34217

机制观察：free_space 的相关性通常接近 1，但接触相关 phase 的 effective rank 大多只有 1–3，condition number 达到 10³–10⁴，且 bowl post_contact 与 wine pre/post_contact 的局部线性明显变差。局部预测可相关，并不意味着 state-dependent metric 能产生可共享、可解码的全局 alphabet。

## 5. K=64 真实物理效应量化结果

settled/immediate error 是 46 维 consequence 按 train-only robust scale 归一化后的 L2，不是米或弧度。所有 decoded actions 都从相同 restored snapshot 真实执行。

### Pooled test：九种方法

每项依次为 settled error / action reconstruction / contact preservation / progress preservation / codebook utilization。

- caaa_v2：102,892.55 / 0.77472 / 0.72917 / 0.82910 / 0.07422
- covariance_mahalanobis：73,693.26 / 0.09490 / 0.80859 / 0.94303 / 0.21094
- euclidean_farthest：90,563.52 / 0.08043 / 0.88314 / 0.97884 / 0.23047
- global_kmeans：76,893.17 / 0.07180 / 0.85710 / 0.97526 / 0.20703
- phase_conditioned_kmeans：74,943.95 / 0.07362 / 0.87760 / 0.96191 / 0.09570
- pca_kmeans：77,658.44 / 0.07023 / 0.85352 / 0.97786 / 0.20703
- old_diagonal_sensitivity：65,831.13 / 0.51590 / 0.70508 / 0.91602 / 0.03516
- random_spd：104,058.81 / 0.58108 / 0.68945 / 0.82422 / 0.11719
- permuted_j：283,447.02 / 1.02188 / 0.62793 / 0.74870 / 0.10547

### 每任务：CAAA-v2 对冻结 baseline

- bowl_on_plate：CAAA 152,841.14，baseline 92,841.81；contact 0.8177 vs 0.8438；progress 0.7943 vs 0.9701；action 0.7990 vs 0.0996；utilization 0.1406 vs 0.2188。
- plate_push：CAAA 1,143.21，baseline 429.85；contact 0.6888 vs 0.7109；progress 1.0000 vs 1.0000；action 1.0665 vs 0.1416；utilization 0.0156 vs 0.2500。
- stove_turn_on：CAAA 2,790.87，baseline 522.28；contact 0.6354 vs 0.8411；progress 0.6862 vs 0.8464；action 0.6828 vs 0.0779；utilization 0.0156 vs 0.2344。
- wine_rack：CAAA 254,794.97，baseline 200,979.10；contact 0.7747 vs 0.8385；progress 0.8359 vs 0.9557；action 0.5506 vs 0.0605；utilization 0.1250 vs 0.1406。

### 每 phase 的 pooled settled error

- free_space：CAAA 267.31；baseline 49.99；old diagonal 465.65。
- pre_contact：CAAA 3,720.05；baseline 454.29；old diagonal 867.16。
- contact_onset：CAAA 5,292.86；baseline 446.70；old diagonal 1,889.63。
- post_contact：CAAA 402,289.97；baseline 293,822.06；old diagonal 260,102.10。

## 6. Episode-clustered bootstrap

10,000 次 deterministic paired episode-cluster resamples；在每个 task 内以 episode 为 cluster 重采样。baseline 只由 calibration 选择，test 不参与选择。

- pooled：-39.62%，95% CI [-168.41%, -5.50%]。
- bowl_on_plate：-64.63%，95% CI [-192.08%, -22.63%]。
- plate_push：-165.95%，95% CI [-450.38%, -69.93%]。
- stove_turn_on：-434.36%，95% CI [-1,028.22%, -44.77%]。
- wine_rack：-26.78%，95% CI [-294.44%, 7.48%]。

## 7. Mechanism controls

- pooled CAAA absolute gain 相对 baseline 为 -29,199.29，即负收益。
- random-SPD absolute gain 为 -30,365.55；相对 CAAA gain retention 为 1.0399。它几乎复现了 CAAA 的伤害，而不是提供特异性几何收益。
- permuted-J absolute gain 为 -209,753.76；retention 为 7.1835，说明打乱 J 会进一步恶化，但真实 J 仍未胜过 baseline。
- old diagonal sensitivity 在 K=64 test 的 pooled error 为 65,831.13，低于 CAAA 和冻结 baseline；但它在 calibration 上为 138,561.79，因此没有被选为 baseline。这一 calibration/test 反转本身提示几何/解码器存在明显分布不稳定。
- global k-means、phase-conditioned k-means、PCA+k-means 在 test 均低于 CAAA；因此 reject gate 中“k-means stronger”也被触发。

## 8. K sensitivity

- K=32：CAAA 94,961.72；covariance 70,256.00；PCA+k-means 69,546.13；old diagonal 113,201.75。
- K=64：CAAA 102,892.55；covariance 73,693.26；old diagonal 65,831.13。
- K=128：CAAA 89,946.96；covariance 78,841.02；phase-conditioned k-means 73,921.51；old diagonal 59,768.64。

CAAA-v2 在 K=32、64、128 都没有取得最优 realized effect error，因此结论不是单一 K 的偶然结果。K=32/128 只作敏感性分析，没有参与 primary disposition。

## 9. Disposition gate 复核

- 要求 pooled reduction ≥10% 且 CI 支持正收益；实测为 -39.62%，CI 全负：失败。
- 要求至少两个 contact-sensitive tasks 改善且 bowl degradation 不超过 5%；四任务点估计全部为负：失败。
- 要求 CAAA action reconstruction 不超过 baseline 的 1.10 倍；实测约 8.16 倍：失败。
- 要求 dead-code ratio 低于 20%；实测 92.58%：失败。
- random-SPD 不应复现主效应；实测它复现了几乎相同的负效应：失败。
- k-means 不应更强；实测多个 k-means baseline 更强：触发 reject。

**唯一正式 disposition：REJECT_CORE_HYPOTHESIS**

## 10. 下一推荐实验

**不开始 policy training。**下一步应使用相同冻结 replay snapshots 做一个更窄的诊断，用来区分两种失败来源：

1. local linear model failure：检查 train/calibration/test 的 J 预测残差、consequence scale floor、SVD 截断与 contact phase 的低秩/病态性。
2. state-dependent codebook alignment failure：固定同一动作库，分别测试 local metric 只用于 assignment、只用于 decoder、以及将 local frames transport 到一个 global consequence basis；同时报告 native-null residual 与 clipping。
3. 只有当该诊断在 held-out snapshots 上恢复正的 realized effect gain、改善 utilization 且 random/permuted controls 不复现后，才预注册新的 Stage 1；仍不直接进入 BC。

## 11. Required artifacts 与完整性

正式输出目录：/mnt/cpfs/zbl-cpfs-new/dataset/leon/experiments/r13_p15_caaa_v2/stage1

- PREREGISTRATION.md — 07e0ac3123dea4c51b18d41c5a8f989e8a18fb7b61a028e08e977a261f274856
- environment_lock.json — f4421974cf948bfa765098e24819d445b209589611cbc3fe11e04c30fb0f0d3e
- task_and_seed_split.json — 7b11ac5dc44877d0b5011c355d178dda44558b35d33e4af7f705fba1bfe1cc22
- branch_replay_validation.json — 7ae428605a824ba7f36786ccffa7ef32e7a7bf795fa8f866b29d099baacb5cf9
- consequence_schema.json — 0d6545ef9917a2cd25f0016547a67157c674628d9cae79bca0b8a6fde66fced1
- branch_rollouts.zarr — 083893fd04a7e8282fc1ae0ba8ad2d362a87070c92f0bc780b672d8f818df59e
- jacobian_metrics.parquet — 901f33cccf0378184c2d19303a4e1a433d7dfbab4e1abf9370371b3902bcc1ff
- alphabet_codebooks/ — 56cbc8d020c8dcad14713acbd416ded890ac63783429907198bdd31e233187d3
- results_by_task.csv — 7b4a190e03d4134463d10e64870d157217f62a9210587c88330d0c747ee4953b
- results_by_phase.csv — 78fc2d836ff83223e5f32341cedf307ab5e60f55f439c105b8a99a27d1ff9ad9
- bootstrap_results.json — 919fc0dcfd3fba71cc64fa7c8ba07d8ef013b14742322b137feff320899fb9fb
- mechanism_controls.csv — e505cd93d56881d97be5bd535c3fa8e702dc91bbead20dea7b65ef232adbbba6
- STAGE1_REPORT.md — 5924ee1a77bc8c9339c5f450a8be87b9f8ad1e1a91b683fbf848f5f3f2047dd5

独立审计：256/256 branch shard hash 有效；128/128 quantized shard hash 有效且与当前 plans 精确一致；task CSV 108 rows；phase CSV 432 rows；Jacobian Parquet 2,304 rows；27 个 codebooks；Zarr 为 4 tasks × 16 episodes × 4 phases，每 snapshot 97 branches；报告 SHA 与 formal completion sentinel 完全一致。

**Stage 1 到此停止。**
