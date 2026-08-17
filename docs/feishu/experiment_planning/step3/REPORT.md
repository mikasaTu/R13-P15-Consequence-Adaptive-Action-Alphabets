# 实验报告

## 结论先行

Stage 2 的 development 顺序门槛在 Gate B 停止。Gate A 明确通过：在全新的 episode 与扰动方向上，真实物理后果 oracle（O1）相对最强动作空间基线 B2 将 `BALANCED_TASK_EFFECT` 从 0.332598 降至 0.133410，下降 59.89%，4/4 个任务均改善。这证明“真实后果等价”在 fresh support 上有很强的组织价值。

但可部署预测器没有实现这一价值。最佳全局 NCEA 的 development 预测 MSE 为 1.944852，高于 linear-J/O2 的 1.691440，即相对改善为 −14.98%；三个 contact-sensitive 任务中改善 0/3；O1–O2 gap 闭合率为 −0.72%；并且没有击败全部 shuffled controls。因此 Gate B 失败。Gate C 未运行，confirmation 未解锁，行为克隆未启动。

唯一最终 disposition：

`ORACLE_ONLY_NO_DEPLOYABLE_MODEL`

## 1. 科学状态与不可追溯修改

Stage 1 和 Stage 1.5 的拒绝结论保持不变，本实验没有重新解释或覆盖旧结果。

| 证据 | Commit / tree | 原结论 | 本次校验 |
|-|-|-|-|
| Stage 1 formal | `34995e8e7c3069b22785ad04536f0d429e75c0fc` | `REJECT_CORE_HYPOTHESIS` | 全树 SHA-256 `047aae35193339a460cd1dbac0e4495d7f9cff4a1cb2799c58b738e86e0e4c5c`，与冻结值一致 |
| Stage 1 published | `434427af0f8adc844851c27cfc050b2c9c6752dc` | 同上 | 必需工件逐项哈希写入 `INPUT_BINDING.json` |
| Stage 1.5 prereg/method/result | `9a3ac1a4c774103fe618bd283909c2793ed581ec` / `aa82d46c5e0828956aef15918c2aa7656844472f` / `76433b6e58196ceeedc4ad005a1110ea8e343ae2` | `REJECT_P15_FAMILY` | 全树 SHA-256 `b5f1c40c32711c5e34ec0d5d2fce175c8642d11f60b5c45af1c6dddc46897b24` |

Stage 2 输入 commit 是 `154d4a89e071d94208f5302955c55c13e3cff7f3`，tree 是 `4199a6280cfb8f5e43b04547291fa792d132b725`。预注册及 fresh-support 冻结 commit 是 `5061f2e`；方法实现 commit 是 `46d7484`；可执行 k-means 与机制诊断冻结 commit 是 `ce26246`；development 结果 commit 是 `7e8b6bc00177a21e205442c0113ca2584cce084c`，tree 是 `ab80f794b9e514fc89dbed8bc65bbdf2b550bd4d`。

LIBERO upstream commit 为 `8f1084e3132a39270c3a13ebe37270a43ece2a01`，源码全树 SHA-256 为 `e9197ca08fe4d7325f561fc40d7425167830253e0f0fceb1af2663b23292f71f`。冻结环境锁 SHA-256 为 `f4421974cf948bfa765098e24819d445b209589611cbc3fe11e04c30fb0f0d3e`。迁移后的 LIBERO 子模块 Git 元数据不可用，因此没有伪称 `git clean`；本次用完整源码树哈希绑定其内容。

## 2. 硬范围执行情况

- 任务保持为 `bowl_on_plate`、`plate_push`、`stove_turn_on`、`wine_rack`。
- Panda、`OSC_POSE`、20 Hz、`H=4`、3 settle steps；连续 6-D 坐标被扰动，gripper 命令逐步复制 nominal demonstration。
- 仿真全部使用 CPU。小型后果预测器仅使用一张 NVIDIA A800-SXM4-80GB：物理 GPU 1 映射为进程内 `cuda:0`。
- PAI 作业数为 0；没有使用多卡，没有 PAI orchestration machinery。
- 没有训练 ACT、Diffusion Policy、SmolVLA、π0.5、DINO-WM 或任何 policy。
- 没有生成 HTML，也没有运行 `K=32/128` sensitivity。

## 3. Fresh episode、snapshot 与确定性

每个任务的官方 HDF5 均有 50 条成功演示；冻结的 16–39 共 24 条/任务全部成功，未发生替换。0–15 只作为历史证据存在，不进入 Stage 2 fit/calibration/development/confirmation。

| Task | HDF5 bytes | HDF5 SHA-256 | Fresh IDs |
|-|-|-|-|
| bowl_on_plate | 468,246,288 | `e69528b0cf10dfc59b20698e12ec2affc03f3887309034d3eb74cac3ec929406` | 16–39 |
| plate_push | 762,855,139 | `36b4e1bced49d2f4ff6b2fce6b1596a63978e14199e2513cd0df71e127bf47a6` | 16–39 |
| stove_turn_on | 447,509,922 | `387fc10747696b80dea6ed8d7f2beaa162bf92ae11750b241073cbd33aac73d5` | 16–39 |
| wine_rack | 878,958,730 | `f9092aa70734fc4083e97fc58c3ba25f87c614d18326182ddc7a455f0ab4da2e` | 16–39 |

共冻结 384 个 phase snapshot（4 tasks × 24 episodes × 4 phases）。每个 snapshot 做 same-action-twice、A→B→A 和 B→A→B 顺序检查，容差 `1e-12`。384/384 个 snapshot 测试通过，即 1,152 个成对比较全部为逐元素数值差 0。

失败的 replay tests：无。

## 4. Unseen support 与公共 action bank

每个 snapshot 使用 12 个 smooth-DCT、6 个 suffix-contact、6 个 low-rank 方向；每方向两个 `[0.04,0.12]` 内确定性半径和正负反向分支，因此每状态有 96 个 target 分支与一个 nominal 分支。四个 split 使用独立派生种子。

- 任意 split 对之间精确 direction overlap：0。
- 任意 split 对之间精确 residual hash overlap：0。
- 最大跨 split 绝对 cosine similarity：0.997272（train/calibration）；这不是精确重复，已完整报告所有六个 split 对。
- Development/confirmation target residual 与 action-bank member 的精确匹配数：0。
- Train-only 公共 bank：M=256；所有 384 个冻结状态均有 256 个有效候选，最低值 256，高于 128 门槛。
- `K=64`；所有 decoded actions 来自公共 bank，clipping 为 0，未做 Jacobian pseudoinversion。

Development Zarr 包含 256 个 support states、128 个 candidate-bank states 和 57,728 个分支：24,832 个 nominal/target support 分支加 32,896 个 nominal/candidate 分支。其全树 SHA-256 为 `ee5420ba32c1a478059f49322cd58185d8ea8bd93697588627c2f7c510646bdd`。

## 5. 指标与 calibration-only 选择

主指标严格使用五组等权 `BALANCED_TASK_EFFECT`：object pose、TCP–object relative pose、contact mode/penetration、gripper/articulation、task progress/constraint。尺度只由 train data 的 MAD/IQR 与预注册物理 floor 决定；使用 capped Huber；raw force 不进入主指标。

Calibration 只使用 episode 24–27，并在 development 被分析前写入选择文件：

- 最强 deployable baseline：B2 phase-conditioned residual k-means，error 0.332290；B1 0.372199、B3 0.369596、B4 0.400484。
- Linear-J：ridge 0.1、9 个 train-state neighbors。
- NCEA：5-member `(128,128)` MLP ensemble。
- MC-NCEA：5-member shared-trunk/four-head `(256,256)` ensemble。
- Contact confidence threshold：0.5。
- UG risk 的 calibration Spearman：0.1972；50/70/90% thresholds 分别为 0.111495、0.194912、0.984944。

128 个 train local-J 的数值秩中位数为 24，但 effective rank 中位数仅 2.519，condition number 中位数为 `5.81e9`。这说明 nominal 局部作用高度低维且病态；本实验只做 bank 上的前向预测，没有使用逆映射。

## 6. Development Gate A：真实后果 oracle

最强 B1/B2/B3 基线是 B2。O1 的 pooled gain 为 59.89%，超过 10% 门槛，且 4/4 任务改善。

| Task | B2 error | O1 error | Relative gain |
|-|-|-|-|
| bowl_on_plate | 0.357182 | 0.126080 | 64.70% |
| plate_push | 0.287326 | 0.125391 | 56.36% |
| stove_turn_on | 0.370617 | 0.129354 | 65.10% |
| wine_rack | 0.315267 | 0.152814 | 51.53% |
| pooled | 0.332598 | 0.133410 | 59.89% |

Oracle gain 也覆盖所有 phase，而不是被 free-space 或某一个 contact phase 驱动：

| Phase | B2 | O1 | Relative gain |
|-|-|-|-|
| free_space | 0.051775 | 0.024190 | 53.28% |
| pre_contact | 0.451000 | 0.201388 | 55.35% |
| contact_onset | 0.393303 | 0.154803 | 60.64% |
| post_contact | 0.434316 | 0.153259 | 64.71% |

五组后果的 O1 gain 分别为 contact/penetration 59.77%、gripper/articulation 63.48%、object pose 55.25%、task progress/constraint 86.15%、TCP-relative pose 50.12%。因此 Gate A 的改善不是 Stage 1 force-dominated metric 的重演；它在预注册的平衡指标内跨组成立。

Gate A：PASS。

## 7. Predictor 结果与 Gate B

下表是 development episode 28–31 的 pooled predictor 指标。MSE 是 34 个 train-scaled 连续后果维度的均值；balanced prediction error 另含等组 Huber 与预测 contact mode。Linear-J 没有 categorical head，因此其 balanced/contact 项记为 N/A，而不是伪造为 0。

| Predictor | MSE | Balanced pred. error | Contact acc. | Effect-norm Spearman | Pred./true norm |
|-|-|-|-|-|-|
| Linear-J / O2 | 1.691440 | N/A | N/A | 0.6193 | 0.2479 |
| NCEA | 1.944852 | 0.381346 | 87.19% | 0.3678 | 0.2147 |
| MC-NCEA | 6.745261 | 0.386701 | 92.01% | 0.1467 | 0.8284 |
| P3 mode-shuffled | 2.263067 | 0.409018 | 87.26% | 0.2931 | 0.3295 |
| P4 state-shuffled | 1.919752 | 0.384539 | 87.14% | 0.3125 | 0.1882 |
| P5 effect-shuffled | 1.884161 | 0.359270 | 87.21% | 0.2860 | 0.1300 |
| P6 random latent | 1,134.855622 | 1.257101 | 73.54% | 0.3223 | 12.8651 |

NCEA 相对 Linear-J 的 prediction gain 为 −14.98%，而要求是至少 +20%。按 contact-sensitive task 分解，NCEA 对 plate_push 为 2.372163 vs linear 2.085306，对 stove_turn_on 为 1.047748 vs 0.717087，对 wine_rack 为 2.475147 vs 2.186891，改善 0/3。

O1 error 为 0.133410，O2 realized error 为 0.353479，NCEA realized error 为 0.355074，因而 O1–O2 gap closed 为 −0.72%，要求为至少 50%。P4 和 P5 的 prediction MSE 都低于 NCEA，所以“击败所有 shuffled/random controls”也失败。

Gate B：FAIL。

## 8. Deployable 与 oracle-only realized results

这些 realized results 是 Gate B 判断所需的 development 证据；Gate C 因 Gate B 失败而未作为门槛执行。

| Method | Balanced effect error | Gain vs B2 | Action RMSE | Contact preserved | Progress abs. err. | Per-state utilization |
|-|-|-|-|-|-|-|
| B2 phase residual | 0.332598 | 0.00% | 0.015816 | 93.99% | 0.002198 | 0.1976 |
| B4 action-only VQ | 0.410683 | −23.48% | 0.023290 | 93.13% | 0.002915 | 0.2601 |
| O1 true oracle | 0.133410 | +59.89% | 0.023624 | 96.97% | 0.000660 | 0.3089 |
| O2 linear atlas | 0.353479 | −6.28% | 0.023151 | 93.12% | 0.001969 | 0.2551 |
| NCEA | 0.355074 | −6.76% | 0.022701 | 94.51% | 0.001994 | 0.2606 |
| MC-NCEA | 0.331092 | +0.45% | 0.022700 | 93.46% | 0.001806 | 0.2556 |
| P3 mode-shuffled | 0.337379 | −1.44% | 0.022570 | 93.39% | 0.001850 | 0.2417 |
| P4 state-shuffled | 0.345809 | −3.97% | 0.022632 | 93.54% | 0.002024 | 0.2346 |
| P5 effect-shuffled | 0.417953 | −25.66% | 0.022653 | 93.00% | 0.002609 | 0.3450 |
| P6 random latent | 0.365318 | −9.84% | 0.023664 | 93.86% | 0.002128 | 0.3232 |

MC-NCEA 的 +0.45% 不具有门槛意义：只改善 2/4 个任务，其中 contact-sensitive tasks 仅 stove_turn_on 改善；相对 B2 的 action reconstruction degradation 为约 43.5%，也远超后续 Gate C 的 10% 条件。NCEA 和 MC-NCEA clipping 均为 0，per-state utilization 略高于 0.25，但这些必要条件不能补偿 Gate B 的失败。

## 9. Uncertainty-gated 结果

Calibration-frozen UG thresholds 在 development 上分别实现 48.81%、69.68%、88.98% 的实际 consequence-quantization coverage。相对 B2 的 realized gains 分别为 −0.58%、+0.02%、+0.34%。更关键的是，development risk/error Spearman 只有 0.0162、0.0088、0.0060，几乎没有排序信息；calibration 上的 0.1972 也很弱。因此 H3 没有得到支持，coverage 不能把一个未通过 Gate B 的 predictor 变成可靠部署方法。

## 10. 机理反解：哪些代码机制带来提升或降低

以下解释只使用预注册的 control 与分解，不生成新 idea。

### 10.1 O1 为什么大幅提升

O1 唯一改变的是 atlas 的组织坐标：候选与 target 都按真实 settled physical consequence 选择，没有生成新动作。它在三类未见方向、四个 phase、四个任务和五个平衡后果组全部改善；同时 action RMSE 反而高于 B2（0.023624 vs 0.015816）。所以提升不是“动作更接近”，而是同一 executable bank 中动作距离较远但物理效果更等价。该机制被 O1/B2 对照直接支持。

### 10.2 NCEA 为什么没有复制 O1

Development support 的 small-radius→large-radius linear extrapolation NRMSE 在 free_space 仅 0.056，但 pre_contact、contact_onset、post_contact 分别为 0.667、0.514、0.452；antithetic asymmetry 分别为 0.914、0.690、0.722。这确认 contact 附近确有局部非线性，O1 的 headroom 不是虚构的。

然而 NCEA 对真实 effect norm 的秩相关只有 0.368，明显低于 Linear-J 的 0.619；预测幅度只有真实均值的 21.5%。更直接地，删除正确 state–effect 配对的 P4 state-shuffled MSE 1.920 反而略优于 NCEA 1.945。这表明当前网络没有从 state features 中提取可泛化的局部后果几何。它在 task-progress group 比 B2 好 14.21%，但 contact/penetration 和 object-pose 分别差 14.82% 和 6.03%，最终 realized error 恶化 6.76%。

### 10.3 P5 为什么“预测 MSE 好、真实量化差”

P5 打乱 effect labels 后把预测幅度压到真实值的 13.0%。在大量小/稀疏物理变化下，这种向条件均值收缩可以让 pointwise MSE（1.884）看起来优于 NCEA（1.945），但它破坏了局部邻域顺序：effect-norm Spearman 仅 0.286，真实 atlas error 为 0.417953，比 B2 差 25.66%。因此 P5 的“提升”是损失函数收缩效应，不是 consequence-equivalence 机制；realized simulation 解除了这个假象。

### 10.4 MC-NCEA 为什么分类提升但连续后果降低

Mode conditioning 将 contact accuracy 从 NCEA 的 87.19% 提高到 92.01%，但连续 MSE 从 1.945 恶化到 6.745。恶化集中在 post_contact（21.462 vs NCEA 2.859）和 wine_rack（21.011 vs 2.475），说明拆分后的专家在某个精密接触域发生了跨 episode 外推失稳，而不是全局容量不足。

MC-NCEA 的 realized error 仍有微小改善，是因为相对 B2，它在 gripper/articulation、object pose、task progress、TCP-relative pose 四个组分别改善 10.91%、6.56%、16.75%、13.08%；但 contact/penetration 组反而恶化 11.95%，抵消了大部分收益。Mode-shuffled P3 的 prediction MSE 2.263 远好于真正 MC 的 6.745，realized error 0.337379 也接近 MC 的 0.331092。因此现有证据不支持“正确 mode routing 是必要机制”；正确 phase heads 在此实现中反而引入了不稳定性。

### 10.5 Linear-J 为什么仍然赢 predictor gate

Local-J 的 effective rank 很低且 condition number 很高，它并没有准确恢复完整动力学；其预测幅度也只有真实值的 24.8%。但在 fresh development 上，它保持了最高的 effect-norm ordering（Spearman 0.619），比所有学习模型更适合进行 bank 内相对排序。NCEA 的非线性容量没有转化为更好的排序，因此 O2 与 NCEA 的 realized error 几乎相同且 NCEA 略差。这解释了“物理系统非线性明显”与“linear-J 仍赢 Gate B”并不矛盾。

## 11. Negative runs 与未执行项

- B1、B3、B4 都不如 calibration-frozen B2；B4 的 action-only latent 不能解释 oracle gain。
- NCEA 不如 Linear-J prediction，且 realized error 比 B2 差 6.76%。
- MC-NCEA 连续预测出现 wine_rack/post-contact 失稳；0.45% realized gain 不满足任何推进门槛。
- P3/P4/P5/P6 均未形成可部署 gain；P4/P5 还否定了 NCEA 已学到可靠 state-conditioned mapping 的说法。
- UG 的 uncertainty ordering 在 development 基本消失。
- Gate C 状态为 `NOT_RUN_GATE_B_FAILED`。
- Confirmation episodes 32–39 只在结果可见前冻结了 ID/hash/snapshot metadata；没有执行其 target/candidate consequence branches，也没有读取 confirmation result。
- `confirmation_rollouts.zarr` 与 `confirmation_quantization.csv` 是 0-observation、`NOT_RUN_DEVELOPMENT_GATE_FAILED` 工件。
- 预注册的 10,000 confirmation bootstrap replicates 实际执行 0 次；因此没有 pooled/per-task confirmation CI。这里缺少 CI 是顺序停止的预期结果，不能用 development bootstrap 替代或越过 Gate B。
- 未运行 `K=32/128`，未运行 BC 或任何 policy。

## 12. 七个问题的明确回答

1. **真实后果等价能否泛化到未见动作方向？** 能，oracle-only 证据强：pooled gain 59.89%，4/4 任务、4/4 phases、5/5 groups 均改善。
2. **非线性预测是否显著优于 linear Jacobian？** 否。最佳 NCEA MSE 高 14.98%，contact-sensitive tasks 改善 0/3。
3. **Contact-mode conditioning 是否必要？** 当前证据不支持。它提高分类准确率，但使连续预测在 wine_rack/post-contact 失稳；mode-shuffled control 没有表现出预期的机制破坏模式。
4. **后果感知 action selection 是否优于 action-only VQ？** NCEA/MC-NCEA 都优于弱的 B4，但只有 MC-NCEA 勉强优于最强动作基线 B2 0.45%，不足以说明可部署优势。
5. **Shuffled/random controls 是否摧毁 gain？** 对 tiny realized MC gain，controls 没有复制；但在 predictor gate 上 P4/P5 反而优于 NCEA MSE，所以关键的 learned state-consequence 机制未被验证。
6. **任何 gain 是否在平衡 task-effect metric 下保留？** O1 的大幅 oracle gain 保留；deployable nonlinear gain 没有达到门槛。
7. **是否准备好进行 small BC policy experiment？** 否。

## 13. 测试、哈希与停止

冻结仿真 Python 3.8 环境与分析 Python 3.11 环境的最终全仓 pytest 均为 18/18 通过。单 GPU smoke 确认模型参数位于 CUDA、早停完成且 calibration loss 有限。

关键文件 SHA-256：

- `PREREGISTRATION.md`: `614a6655e1a2c4df68b79aecfa7c5a745a491cd9719830f3a56c4111ca8aad59`
- `INPUT_BINDING.json`: `033a29585367ea3a03af17c7b824b6eb0cdf3f25e2c71c50398cb3a6d371a538`
- `consequence_metrics.json`: `efef61178d084717819df11437d2da395d386391c28f73bae1357bd14429688e`
- `perturbation_banks.npz`: `dc4db5d1a7bd6892dcc2283a97fe8b63ee2aafe4be7cf4afbf2b896886bbab40`
- `action_bank.npz`: `d41f0dc748866cae3ef151d9f16e39789485d6e633a0a88f62fe4c570661600b`
- `development_quantization.csv`: `97caa4e5c4a3d5378c7b5e386b662267d2056dbbf83141cdcd4d35d5d1c02018`
- `predictor_metrics.csv`: `1f361c3c064b648f9efc7b268469c64d71a29ac5ac25552bd2f9a317a56ca346`
- `development_controls.csv`: `7b4ae7e758235b8ed8f1c7d4dde428cd09dc1994b1f4eb98fc32365d7abd83a9`
- `bootstrap_results.json`: `06e4cefd1c57a104c7a3d3171901b97ea912301a1b7ac4d8ca6ec308b76a3c70`

下一步建议不是启动新实验，而是接受 Gate B 的停止结论：不启动 small BC，不读取 confirmation，不把 oracle-only 价值表述为 deployable result。若未来继续，必须另做新的预注册；本报告不生成该新 idea。

最终 disposition（exact）：

`ORACLE_ONLY_NO_DEPLOYABLE_MODEL`
