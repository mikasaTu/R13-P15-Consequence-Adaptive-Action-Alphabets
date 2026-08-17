# R13-P15 Stage 5 preregistration

## Identity and stopping point

- Experiment: **Context-Identifiable Consequence Retrieval and Dynamic Local Alphabet (CICR-DLA)**.
- Branch: `r13-p15-stage5-context-identifiable-consequence-metric`.
- Pre-result repository input: commit `eba489ec8f866f712b582083c088e93b0aaccf11`, tree `0137158cfd5a3f4e1162acf4f47bdc073839baf9`.
- Frozen benchmark: the existing four standard `libero_goal` tasks (`bowl_on_plate`, `plate_push`, `stove_turn_on`, and `wine_rack`) with Panda `OSC_POSE`, 20 Hz, H=4, and three settle steps.
- This is a state-based mechanism audit. It stops after Stage 5. It does not authorize fixed-policy reranking, ACT, Diffusion Policy, SmolVLA, pi0.5, DINO-WM, any VLA, or any policy training other than the explicitly bounded nominal trajectory generator below.
- Simulation is local CPU where possible. Small-model training may expose at most one local GPU. A PAI job is permitted only if local execution is technically impossible; none is planned.

The user's explicit execution instruction requires every registered experiment to be completed even when an intermediate gate fails. Gate failures therefore freeze the corresponding scientific disposition but do not censor later negative/control measurements. Later measurements cannot override an earlier failed gate.

## Immutable history

The following conclusions and their original artifacts are read-only:

- Stage 1: `REJECT_CORE_HYPOTHESIS`.
- Stage 1.5: `REJECT_P15_FAMILY`.
- Stage 2: `ORACLE_ONLY_NO_DEPLOYABLE_MODEL`.
- Stage 3: `ORACLE_ONLY_NO_LEARNABLE_RANKER`.
- Stage 4: `STATIC_EFFECT_METRIC_ONLY`.

Stage 5 is a new falsification experiment. It cannot rescue, rename, or overwrite a historical result. `HISTORICAL_BINDING.json` must match the published Git trees, selected result files, all Stage 4 checkpoints, the LIBERO source tree, the action bank, and frozen scalers before any Stage 5 development metric is computed. A mismatch fixes the disposition to `BLOCKED_HISTORICAL_BINDING_MISMATCH`.

## Questions and estimands

Primary question: after removing additive shared-context cancellation, does observable state/history/nominal action learn a context-specific physical-effect metric that beats an equally trained static consequence metric and matched action-only, shuffle, and no-reversal controls?

Secondary question: if the full-bank metric works, does deterministic state-specific K=64 medoid compression retain at least 75% of its incremental gain while returning only existing executable actions and respecting action/contact budgets?

The primary outcome is the exact frozen Stage 2--4 `BALANCED_TASK_EFFECT`: equal-weight object pose, TCP-object relative pose, contact/penetration, gripper/articulation, and task-progress/constraint groups; train-only robust scales; capped Huber; raw force excluded. The consequence scale is not refit. The old force-dominated Stage 1 value is secondary continuity evidence only.

## Data firewall

Episode splits are fixed:

- training and train-only reversal margins: episodes 16--31;
- calibration and temperature/model selection: episodes 32--35;
- development: episodes 36--39;
- historical exploratory only: episodes 40--49;
- confirmation: genuinely new successful generator trajectories, never an official demonstration episode.

The Stage 4 expanded training cache supplies 768 train states. Historical calibration/development/exploratory shards retain their original simulator outcomes. Development and exploratory results cannot select architecture, temperature, bank, loss, thresholds, generator, or confirmation split.

## Frozen M=128 local executable bank

The source is the byte-bound Stage 4 M=256 residual bank. The covariance is fit to the union of the 96 frozen Stage 4 train target residuals and 256 source-bank residuals with eigenvalue regularization `1e-6`. Whitening is used only to compute each residual's zero-origin norm.

Candidates are stratified in the fixed order `source phase (free_space, pre_contact, contact_onset, post_contact) × family (smooth-DCT, suffix-contact, low-rank temporal-action) × sign (-,+)`. Each of 24 strata receives five entries and the first eight lexicographic strata receive one additional entry. Within a stratum, selection is ascending whitened norm and then original source index. Exact target equality and Euclidean near-duplicates within `1e-12` are forbidden. Original M=256 indices are preserved. No residual is clipped, synthesized, or pseudoinverted.

M=128 is primary and K=64 is the only primary compression. M=256 and K other than 64 are locked until after the exact Stage 5 disposition is frozen. Every evaluated state must have at least 96 executable local candidates.

## Oracle adaptivity audit and Gate 0

On development episodes 36--39, evaluate:

- `O_STATE_FULL`: current-state true-effect nearest member over M=128;
- `O_STATE_K64`: current-state true-effect K=64 medoids and decoding;
- `O_STATIC_FULL`: train-state mean true-effect table;
- `O_CONTACT_FULL`: train mean conditional only on current observable contact;
- `O_PHASE_FULL`: train mean conditional on the privileged phase, diagnostic only.

Report pooled, task, phase, target-family, and consequence-group results; state-versus-static/contact/phase gaps; and full-versus-K64 compression loss.

Gate 0 requires `O_STATE_FULL` to beat the stronger of `O_STATIC_FULL` and `O_CONTACT_FULL` by at least 8% pooled, 12% across contact-onset plus post-contact, and in at least two of the three contact-sensitive tasks. The strict reversal benchmark must contain at least 1,000 pairs, with reversal rate at least 15% in at least two contact phases. Failure fixes `STATIC_EFFECT_GEOMETRY_SUFFICIENT`, while the user-required remaining experiments continue as labeled negative evidence.

## Strict context reversals

A tuple uses the same target residual and the same ordered candidate pair at two states within a task. Preference is for the same current-contact category. With a task/phase train-only 25th-percentile robust margin, it is valid only if

`D_s1(t,i) + margin < D_s1(t,j)` and `D_s2(t,j) + margin < D_s2(t,i)`.

Quotas are 256 per task/phase for train and 128 for calibration/development. Target and direction-family sampling cycles deterministically before repetition. Margins are never weakened, labels are never fabricated, and undersupplied strata are reported. Episodes and exact reversal tuples are split-disjoint.

## Model hierarchy

All distances are symmetric and exactly zero for identical inputs.

- `B0_CURRENT_CONTACT_KMEANS`: current-contact residual k-means mapped to executable local-bank medoids.
- `B1_ACTION_ONLY`: the matched action encoder trained without consequence labels, using covariance-whitened action distance and action-neighbor ranking.
- `B2_STATIC_CONSEQUENCE`: a state-independent diagonal PSD metric trained on true consequence distance.
- `P1_CONTEXT_GATED_PSD`: first train and freeze B2's `z=psi(nominal,residual)` and positive diagonal base weights. Then train only a bounded context modulator `m_s=g(observable context, nominal)`, with distance `sum_j softplus(w0_j) exp(m_s_j) (z_tj-z_cj)^2`.

P1 has no raw-action-L2 multiplier and no additive context path. Setting modulation to exact zero recovers B2 bit-for-bit. The modulation bound is 1.25 log units. A train-context offset is solved and frozen so matched-train mean modulation is zero. Modulation vectors, norms, and metric condition numbers are reported. Optional P2 is not enabled, preventing post-hoc architecture search.

P1 and every matched control use identical architecture, parameter count, seeds, optimizer steps, and budget: `ACTION_ONLY`, `CONTEXT_SHUFFLED`, `NOMINAL_SHUFFLED`, `JOINT_STATE_NOMINAL_SHUFFLED`, `CONSEQUENCE_LABEL_SHUFFLED`, `NO_REVERSAL_LOSS`, `PHASE_ONLY`, and `CURRENT_CONTACT_ONLY`. Phase-only is privileged diagnostic evidence and can never be the selected proposal.

Permitted proposed inputs are the current observable state/mask, two prior observable-state deltas/masks, two prior actions/availability masks, current observable contact, nominal H=4 chunk, target/candidate residual chunks, and task identity. Future state/consequence, simulator outcomes, phase, success/future reward, IDs, row index, confirmation result, post-execution contact, and oracle membership are forbidden. Candidate-order permutation must preserve selected original IDs exactly.

## Training contract

The encoder hidden widths are `[128,96]`, context modulator widths `[128,64]`, and embedding dimension 24. Every final seed is one of `[13150517,13150529,13150543]`. AdamW uses learning rate `3e-4`, weight decay `1e-5`, gradient clip 5.0, and exactly 2,500 optimizer steps per final method/control.

The loss is frozen as

`1.0 L_distance + 0.5 L_pairwise + 0.5 L_listwise + 1.0 L_reversal + 0.01 L_gate`.

- Distance: Huber between `log1p(predicted)` and `log1p(true)`.
- Pairwise: top-8 positives against ranks 9--32, contact-changing candidates, action-close/effect-far candidates, and action-far/effect-close candidates. Easy random negatives cannot dominate.
- Listwise: the complete M=128 bank.
- Reversal: only frozen strict cross-state tuples.
- Gate: mean-modulation, norm, and condition-number penalties.

Listwise temperature candidates are `[0.10,0.15,0.20]`. Calibration chooses the lowest mean realized `BALANCED_TASK_EFFECT`, then lowest oracle regret, highest NDCG@16, then smallest temperature. Development cannot refit it.

## Development retrieval and K=64

FULL selects the minimum predicted distance over all 128 existing candidates. Dynamic K=64 computes all candidate-candidate distances at the current state, performs deterministic metric-space K-medoids with original-index tie breaks, and maps each target to its nearest medoid. It never clips, synthesizes, or replaces an invalid action.

Ranking outcomes are joint/side reversal accuracy, Spearman, Kendall tau, NDCG@16, Recall@1/8, oracle regret, context-intervention code changes, modulation norm, task, and phase. Simulator outcomes are pooled/five-group `BALANCED_TASK_EFFECT`, object pose, TCP-object pose, contact preservation, progress error, action RMSE, utilization, perplexity, clipping, valid-bank count, and latency. Confidence intervals use 10,000 paired bootstraps clustered by source episode.

Gate 1 requires P1 FULL versus B2 FULL: at least 5% pooled realized gain with paired 95% CI lower bound above zero, 3/4 tasks, 2/3 contact tasks, 10% oracle-regret reduction, 0.05 NDCG@16 gain, joint reversal accuracy at least 0.35 and 0.15 above B2, and all three seed directions positive. Joint state+nominal shuffle and consequence-label shuffle may retain at most 25% of incremental gain; action-only at most 50%; no-reversal may not reproduce reversal improvement.

Static consequence value is separately established only if B2 beats the stronger of B0/B1 by the same 5% pooled/positive-CI/3-task/2-contact-task realized screen plus 10% regret and 0.05 NDCG gains. If P1 fails but this static gate passes, the disposition is `STATIC_CONSEQUENCE_METRIC_ONLY`; otherwise `REJECT_LEARNED_CONSEQUENCE_METRIC`.

Gate 2 requires P1 K64 versus the strongest deployable B0/B1/B2 K64: 8% realized gain, at least 75% retention of FULL incremental gain, 3/4 tasks, 2/3 contact tasks, action-RMSE degradation at most 20%, contact-preservation drop at most one percentage point, normalized utilization at least 0.25, zero clipping, and at least 96 valid candidates per state. Gate 1 pass plus Gate 2 failure gives `PIVOT_TO_CONSEQUENCE_RETRIEVAL_STEERING`.

## Frozen nominal generator and fresh firewall

No compatible pre-existing frozen state policy covers all four tasks. The rejected checkpoint is recorded in `NOMINAL_GENERATOR_BINDING.json`. Therefore one shared, small state-only H=4 chunk BC is trained strictly from official demonstrations 0--31. Input is the current 46D physical observable, its 46D mask, previous 7D action, and 4D task one-hot. A `[256,256,128]` MLP predicts a tanh-bounded 4×7 chunk; closed-loop rollout executes only the first predicted action. Training uses AdamW for exactly 10,000 steps, seed 13150505, and no image, VLA, or Stage 5 consequence label. The generator is frozen before any development result is inspected and is never treated as a proposed contribution.

Two hundred ascending rollout seeds per task are precommitted. After all development choices and checkpoints are frozen, run them and keep the first 12 trajectories per task whose environment task-success predicate becomes true. No metric result enters acceptance. Fewer than 12 successes for any task fixes `BLOCKED_NO_FRESH_TRAJECTORIES`; later possible diagnostics cannot change it.

Before any confirmation branch, freeze and commit the selected architecture/temperature, all three proposal checkpoints, B0/B1/B2/controls, M=128 bank, K64 algorithm, metrics, thresholds, successful rollout IDs/seeds, and exact four phase indices in `FRESH_CONFIRMATION_SPLIT.json`. Replay validation uses separate sacrificial trajectories, never a confirmation state before the split commit.

Confirmation uses 12 successful new trajectories per task × four phase states × 96 independently seeded fresh targets. The fresh target bank has zero exact overlap with the local candidate bank and all historical target supports. Nominal, target, and all 128 candidates execute from each identical restored state. Evidence is labeled `FRESH_POLICY_TRAJECTORY_CONFIRMATION`.

The confirmation gate requires P1 K64 to beat the strongest deployable baseline by 10% pooled with positive paired-CI lower bound, 3/4 tasks, 2/3 contact tasks, context- and label-shuffle retention at most 25%, action-RMSE degradation at most 20%, contact drop at most one point, utilization at least 0.25, zero clipping, positive direction for all three seeds, and retained oracle adaptive headroom of at least 8%.

## Disposition precedence

Exactly one disposition is emitted in this order:

1. historical mismatch → `BLOCKED_HISTORICAL_BINDING_MISMATCH`;
2. Gate 0 failure → `STATIC_EFFECT_GEOMETRY_SUFFICIENT`;
3. Gate 1 failure with static B2 value → `STATIC_CONSEQUENCE_METRIC_ONLY`;
4. Gate 1 failure without static B2 value → `REJECT_LEARNED_CONSEQUENCE_METRIC`;
5. Gate 1 pass and Gate 2 failure → `PIVOT_TO_CONSEQUENCE_RETRIEVAL_STEERING`;
6. Gates 0--2 pass but generator shortage → `BLOCKED_NO_FRESH_TRAJECTORIES`;
7. confirmation failure → `CONFIRMATION_FAILED`;
8. all gates pass → `GO_TO_FIXED_POLICY_RERANKING`.

Later user-required negative experiments never upgrade an earlier failed gate. No Stage 5 result is evidence of task-success improvement, VLA improvement, paper readiness, or restored novelty.
