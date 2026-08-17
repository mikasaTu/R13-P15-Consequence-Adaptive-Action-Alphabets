<title>step1</title>

# 四、第一步：详细执行计划

## Stage 1：ManiSkill3 Oracle Consequence-Geometry Audit

这一阶段：

• 不训练 π0.5；

• 不训练 SmolVLA；

• 不训练 ACT；

• 不训练 world model；

• 只回答 P15 最底层假设是否成立。

## 1. 冻结环境

固定并记录：

• ManiSkill3 exact commit 或 release；

• SAPIEN、PyTorch、CUDA；

• 四个 task 版本；

• robot UID；

• observation mode；

• control mode；

• simulator timestep；

• demo revision；

• action normalization；

• episode seed 列表。

推荐：

obs_mode = state_dict  
control_mode = pd_ee_delta_pose  
tasks =  
  PickCube-v1  
  PushT-v1  
  PegInsertionSide-v1  
  PlugCharger-v1

state_dict

只用于 oracle consequence label，不代表后续 policy 可以使用 privileged state。

## 2. 先验证仿真分支可重复性

使用

num_envs=1

：

save state  
→ execute action chunk  
→ record next state/contact/reward  
→ restore state  
→ execute same action chunk again

要求：

• object pose 一致；

• robot qpos/qvel 一致；

• contact state 一致；

• reward/info 一致；

• success 一致。

再验证：

A branch → B branch

和：

B branch → A branch

在每次都恢复 snapshot 时结果一致。

未通过则返回：

BLOCKED_NONDETERMINISTIC_BRANCHING

禁止继续构建 Jacobian。

## 3. 构造 branch dataset

每个任务：

• 16 个成功 episode；

• 每个 episode 抽取 4 个状态：

• free-space；

• pre-contact；

• contact onset；

• post-contact；

• 共 64 个 snapshot/task；

• episode-disjoint split：

• train：8 episodes；

• calibration：4 episodes；

• test：4 episodes。

每个 snapshot：

1. 从 demo 得到基础 action chunk

a_0

；

2. 使用

H=4

个实际控制步；

3. action dimension 从环境读取，不硬编码；

4. 构造 24 个固定 perturbation directions；

5. 每个 direction 使用正负 antithetic pair；

6. 半径使用 normalized action space 中的：

• 0.05；

• 0.10；

7. 每次执行前恢复完全相同的 snapshot；

8. 保存即时结果和 3 个 zero-action settle steps 后的结果。

总 branch 数约为：

4 \times 16 \times 4 \times 24 \times 2 \times 2 = 24,576

这个规模适合 ManiSkill3 并行执行，不需要大模型训练。

## 4. Consequence vector

统一 schema：

y=[ \Delta p\_{\text{object}}, \Delta R\_{\text{object}}, \Delta p\_{\text{tcp-object}}, \Delta R\_{\text{tcp-object}}, c\_{\text{onset}}, c\_{\text{persist}}, c\_{\text{release}}, \text{grasp relation}, \Delta \text{task progress}, \text{constraint violation} ]

要求：

• rotation 使用 log-map 或稳定 6D representation；

• 连续维度只用 train split 估计 robust scale；

• 不存在的 task dimension 用 mask，不填伪常数；

• success 作为外部评测，不直接作为 Jacobian 连续维。

## 5. 估计 Jacobian

对每个 snapshot 使用局部 ridge regression：

\hat J_s = \arg\min_J \sum_i \\|\Delta y_i-J\Delta a_i\\|\_W^2 + \lambda\\|J\\|\_F^2

只在 calibration split 选择：

• ridge

\lambda

；

• eigenvalue cutoff；

• rank

r

；

• metric regularization。

报告：

• local linearity

R^2

；

• effective rank；

• condition number；

• phase-wise anisotropy；

• contact vs free-space 差异。

## 6. 构造 alphabet

主设置：

K = 64

敏感性：

K = 32, 128

在 train split 的 consequence-sensitive coordinates

z_s

上学习共享 codebook。

解码时：

• token 解码 consequence-sensitive component；

• null-space component 作为连续 residual；

• residual 的 metric norm 受限：

r^\top G_sr\le \rho^2

## 7. 必须比较的 baseline

所有方法使用相同：

• 数据；

• K；

• residual 维数；

• residual budget；

• branch rollout；

• train/calibration/test split。

比较：

1. Uniform per-dimension bins

2. Global action-space k-means

3. Phase-conditioned k-means

4. PCA + k-means

5. Action-covariance Mahalanobis

6. Old diagonal/scalar sensitivity

7. Random SPD metric

8. Permuted-

J

9. Full CAAA-v2

10. Continuous unquantized upper bound

其中

permuted-J

是最重要的机制对照：

保持 Jacobian 数值分布，但把 state 与 Jacobian 配对打乱。

如果它和 proposed 一样有效，就说明增益不是来自局部 consequence geometry。

## 8. Primary metrics

### Metric A：距离是否对应物理后果

\rho= \operatorname{Spearman} \left( d\_{\text{metric}}(a_i,a_j), d\_{\text{consequence}}(y_i,y_j) \right)

### Metric B：真实 effect quantization error

把原动作量化并重新在 simulator 中执行：

E\_{\text{effect}} = \\| y(a\_{\text{quantized}}) - y(a\_{\text{original}}) \\|\_W

不能只比较 Jacobian 线性预测，必须重新执行物理 branch。

### Metric C：Contact-mode preservation

量化前后：

• no-contact；

• onset；

• persistent；

• release；

是否保持一致。

### Metric D：Task-progress preservation

量化动作是否保留原动作带来的 task progress。

### Secondary

• raw action reconstruction error；

• codebook perplexity；

• dead-code ratio；

• null-space residual norm；

• per-phase error；

• inference cost。

## 9. 预注册 Gate

### GO_TO_SMALL_BC

必须同时满足：

1. pooled test effect error 相比最强非-consequence baseline 至少降低 10%；

2. episode-clustered bootstrap 95% CI 支持该提升；

3. PegInsertionSide

、

PlugCharger

、

PushT

中至少 2 个任务提升；

4. PickCube

性能不退化超过 5%；

5. permuted-

J

保留的增益不超过 proposed 增益的 25%；

6. random-SPD 不应复制 proposed 的主要收益；

7. raw action reconstruction error不恶化超过 10%；

8. dead-code ratio低于 20%。

### REVISE_ALPHABET

满足任一情况：

• 物理距离相关性提高，但 effect quantization 只提高 5%–10%；

• 只有插入/contact 状态有效；

• full

G

只比 diagonal 略好；

• null-space residual 过大；

• metric condition number 不稳定。

此时优先收窄为：

Contact-Phase Consequence Alphabet

而不是直接训练 VLA。

### REJECT_CORE_HYPOTHESIS

满足任一：

• 比最佳 baseline 提升不足 5%；

• k-means 更强；

• permuted-

J

保留超过 50% 的增益；

• 随机 metric 与 proposed 相当；

• oracle

G

在 test episode 上不稳定；

• 提升只发生在 task selection 时见过的状态。

# 五、可直接交给 Agent 的第一步计划

Execute Stage 1 for R13-P15-v2:  
Consequence-Riemannian Action Alphabets (CAAA-v2).  
  
Scientific question:  
Does a local physical-consequence pullback metric organize action chunks  
better than Euclidean distance, covariance, k-means, PCA, and the previous  
scalar-sensitivity implementation?  
  
Hard scope:  
  
\- Use ManiSkill3 only.  
\- Do not use π0.5, SmolVLA, DINO-WM, or any policy training.  
\- Do not launch PAI.  
\- Use at most one local GPU for parallel simulation.  
\- Do not generate HTML.  
\- Do not build formal activation, custom publication systems, cryptographic  
  audit infrastructure, or large mutation frameworks.  
\- Use normal Git, JSON/JSONL, CSV, NPZ/Zarr and Markdown.  
  
Tasks:  
  
\- PickCube-v1  
\- PushT-v1  
\- PegInsertionSide-v1  
\- PlugCharger-v1  
  
Use state_dict only for oracle labels.  
Use the same control mode and action semantics for all compared methods.  
  
Required stages:  
  
1. Freeze ManiSkill3/SAPIEN/PyTorch/CUDA commits and environment.  
2. Validate snapshot restore and deterministic branching.  
3. Download or regenerate official successful demonstrations.  
4. Freeze 16 episodes per task and four phase snapshots per episode.  
5. Split episodes 8/4/4 into train/calibration/test.  
6. Generate 24 deterministic perturbation directions, antithetic signs and  
   normalized radii 0.05 and 0.10 around H=4 action chunks.  
7. Execute every branch from the identical restored simulator state.  
8. Save immediate and three-step-settled physical consequences.  
9. Define and freeze a task-generic consequence schema.  
10. Fit local ridge Jacobians using train data and calibration-only  
    hyperparameter selection.  
11. Build:  
    \- Euclidean  
    \- covariance Mahalanobis  
    \- global k-means  
    \- phase-conditioned k-means  
    \- PCA+k-means  
    \- old diagonal sensitivity  
    \- random SPD  
    \- permuted-J  
    \- full CAAA-v2  
12. Use K=64 as primary and K={32,128} only as sensitivity analysis.  
13. Quantize held-out actions, execute the decoded actions in simulation,  
    and measure realized physical-effect error.  
14. Compute episode-clustered bootstrap confidence intervals.  
15. Return exactly one:  
    GO_TO_SMALL_BC  
    REVISE_ALPHABET  
    REJECT_CORE_HYPOTHESIS  
  
Required artifacts:  
  
experiments/r13_p15_caaa_v2/stage1/  
├── PREREGISTRATION.md  
├── environment_lock.json  
├── task_and_seed_split.json  
├── branch_replay_validation.json  
├── consequence_schema.json  
├── branch_rollouts.zarr  
├── jacobian_metrics.parquet  
├── alphabet_codebooks/  
├── results_by_task.csv  
├── results_by_phase.csv  
├── bootstrap_results.json  
├── mechanism_controls.csv  
└── STAGE1_REPORT.md  
  
The report must include:  
  
\- exact commits and hashes;  
\- all failed replay tests;  
\- per-task and per-phase local linearity;  
\- effective rank and condition number;  
\- metric-to-consequence Spearman correlation;  
\- realized effect quantization error;  
\- contact-mode and task-progress preservation;  
\- action reconstruction error;  
\- codebook utilization;  
\- permuted-J and random-SPD controls;  
\- pooled and per-task confidence intervals;  
\- the exact final disposition;  
\- the next recommended experiment.  
  
Stop after Stage 1.  
Do not automatically start ACT, Diffusion Policy, SmolVLA or π0.5.
