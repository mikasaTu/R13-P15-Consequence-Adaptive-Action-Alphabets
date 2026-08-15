# R13-P15 Stage 4 preregistration

Frozen before any Stage 4 method result is computed or inspected. This is a
new method hypothesis, not a relabeling of earlier evidence and not an
inheritance of the original novelty grade.

## Historical status (immutable)

- Stage 1: `REJECT_CORE_HYPOTHESIS`.
- Stage 1.5: `REJECT_P15_FAMILY`.
- Stage 2: `ORACLE_ONLY_NO_DEPLOYABLE_MODEL`.
- Stage 3: `ORACLE_ONLY_NO_LEARNABLE_RANKER`.
- Stage 3 C5/C6 remain rejected. Frozen C3 is only a weak exploratory signal.
- Episodes 40–49 remain historical exploratory evidence and may never be
  described as untouched confirmation.

All source commits, Git tree objects, checkpoints, action banks, scalers and
historical result hashes are asserted in `HISTORICAL_BINDING.json`. Any
mismatch stops execution; no historical file may be edited.

## Scientific questions

1. Does frozen C3 reproduce through its complete decoding path?
2. How much error is due to bank coverage, learned metric, K=64 compression,
   and the C4 decoder override?
3. Does consequence ordering actually depend on observable current state,
   nominal chunk and history?
4. Does context-reversal training improve state-dependent ordering?
5. Can an action-local trust region retain effect gains while controlling
   action deviation?
6. Is any surviving mechanism an alphabet or only full-bank retrieval?

## Hard scope

- Tasks: `bowl_on_plate`, `plate_push`, `stove_turn_on`, `wine_rack`.
- Panda `OSC_POSE`, 20 Hz, H=4, three settle steps; normalized 6-DoF pose
  coordinates and demonstration gripper values are identical for every arm.
- State-based mechanism audit only. No policy, ACT, Diffusion Policy, SmolVLA,
  pi0.5, DINO-WM or behavior cloning is trained.
- CPU simulation where possible; at most one local GPU for small metric models.
- No PAI job is submitted.
- No HTML, activation system, cryptographic publication system or mutation
  farm is produced.
- Complete every registered experiment even after a failed gate, then stop at
  Stage 4.

## Frozen data and support

The exact training manifest contains 768 states: four tasks × train episodes
16–31 × four phases × three unused timesteps. For each Stage 3 phase anchor,
the selector takes the three nearest valid H=4 timesteps in a deterministic
midpoint window, with a stable hash tie-break. Every Stage 3 snapshot is
excluded. Every selected nominal chunk obeys `max(abs(a)) <= 0.875`, and all
M=256 bank residuals plus all support residuals must remain executable without
clipping.

The shared support contains 24 unit directions: eight smooth-DCT, eight
suffix-contact and eight low-rank temporal-action directions. Radii are 0.06
and 0.10 with antithetic signs, yielding 96 target branches per state and exact
1:1:1 family balance. Snapshot indices, states, actions, episode hashes,
support tensors and seeds are frozen in `TRAINING_STATE_MANIFEST.json` and
`training_support_bank.npz` before collection.

Train episodes are used only for fitting. Calibration episodes 32–35 select
registered alternatives. Development episodes 36–39 evaluate gates.
Episodes 40–49 are reported separately as historical exploratory replication.
No development or exploratory result may choose a model or setting.

## Frozen failure decomposition

Without retraining, evaluate `B2`, `O_FULL`, `O_K64`, `C3_FULL`, `C3_K64` and
`C5` on the existing Stage 3 development and exploratory shards. Report:

- bank-compression loss = `O_K64 - O_FULL`;
- learned-metric loss = `C3_FULL - O_FULL`;
- learned-compression loss = `C3_K64 - C3_FULL`;
- C4-override loss = `C5 - C3_K64`.

Report pooled and by task, phase, direction family and five consequence groups.

## Frozen context audit

Using matched targets and bank candidates across states, measure context
reversal rate, true top-8 Jaccard, best-candidate churn, state-conditioned
versus globally averaged oracle, and distance variance explained by
state/task/phase/action. A reversal is valid only when both opposite rankings
clear a robust train-only margin.

On the same frozen three-member C3 ensemble, run exactly these inference-only
interventions: correct context; nominal zeroed; nominal shuffled within task;
state/mask/contact shuffled within task; history/actions/masks shuffled within
task; state and nominal jointly shuffled; all context zeroed while retaining
the action pair. Report distance and selection changes, NDCG@16, Recall@8,
oracle regret and realized effect. These intervention arms are not retrained.

## C3 independent objective selection

Retrain the four Stage 3 objective tuples with the exact C3 architecture. The
screening seed is 56229435. Select on calibration C3 FULL only by: (1) lowest
oracle regret, (2) highest NDCG@16, (3) lowest tuple index. Then train only the
selected tuple's three members with seeds 56229435, 2279153700 and 2652429101.
Evaluate FULL, deterministic predicted-space FPS64 and deterministic
predicted-space k-medoids64. Simulator outcomes are forbidden in atlas
construction.

## Context-reversal C3

Evaluate `CR_C3_SHARED` and `CR_C3_GROUP`. The shared family produces one
32-dimensional context-conditioned effect embedding. The group family uses
five 16-dimensional embeddings and an equal-weight mean distance. Inputs are
only observable current state/mask, two observable history deltas/masks, two
previous actions/masks, current contact, nominal H=4 chunk, task identity, and
target/candidate residual. Future state, simulator outcome at inference,
demonstration phase, episode outcome and row IDs are forbidden.

Training uses full-bank listwise, pairwise and context-reversal losses with
weights 1.0, 0.5 and 0.5, respectively. True/model temperatures are 0.15.
Train seeds are 13150417, 13150429 and 13150443 for every family and matched
control. Maximum training is 30 epochs, query batch 16, AdamW learning rate
3e-4 and weight decay 1e-5. Matched controls have the same architecture,
parameter count, ensemble size and budget: action-only, context-shuffled,
nominal-shuffled, consequence-label-shuffled, reversal-label-shuffled and
no-reversal-loss.

## Trust region and bounded diagnostic

For each K=64 atlas, rank codes by train-covariance-whitened residual-action
distance, keep L in {8,16,32,64}, and choose minimum CR-C3 distance. L=64 is
the no-trust-region control. Calibration selects the lowest realized balanced
effect error, then lowest action RMSE, then smallest L. The decoder must return
an existing executable bank member: clipping, synthesis and pseudoinverse are
forbidden.

The optional post-training correction is
`d_C3 * exp(gamma * tanh(g_symmetric))`, with gamma in {0,0.1,0.2}; gamma=0 is
the exact sham. Calibration only may select gamma. It cannot bypass C3 and is
not required for success.

## Development metrics and gates

Report balanced task effect, five per-group errors, oracle regret, Spearman,
Kendall, NDCG@16, Recall@1/8, action RMSE, contact preservation, task-progress
error, normalized utilization, perplexity, clipping, inference latency,
reversal accuracy and context-shuffle gain retention.

Gate A requires O_K64 to improve at least 20% over B2 on development, with at
least 3/4 tasks and 2/3 contact-sensitive tasks improving. Failure disposition:
`REJECT_CONSEQUENCE_HEADROOM`.

Gate B requires the best CR-C3 FULL to improve over B2 by at least 5% on
episodes 36–39, 5% on historical episodes 40–49, and 8% pooled; improve 3/4
tasks and 2/3 contact tasks; improve at least 5% over frozen C3; add at least 10
percentage points of reversal accuracy; retain at most 50% of gain under joint
state+nominal shuffle; and not be reproduced by action-only or shuffled-label
controls. Failure is `REJECT_LEARNED_CONSEQUENCE_METRIC`; if gains survive but
context does not matter, use `STATIC_EFFECT_METRIC_ONLY`.

Gate C requires CR-TR-C3 K64 to improve at least 8% over B2, retain at least
75% of FULL gain, improve 3/4 tasks and 2/3 contact tasks, limit action-RMSE
degradation to 20%, limit contact-preservation drop to one percentage point,
achieve normalized utilization at least 0.25 and clip zero actions. If FULL
passes but every K=64 method fails, use
`PIVOT_TO_CONSEQUENCE_RETRIEVAL_STEERING`; if only contact-onset/post-contact
passes, use `NARROW_TO_CONTACT_CONSEQUENCE_METRIC`.

## Fresh confirmation firewall

The inventory is audited now, but no confirmation state, perturbation or branch
is frozen or executed at this preregistration step. Official files end at demo
49, so source 1 is unavailable. After all development methods, ensemble,
atlas, K, L, metrics and thresholds are frozen, the next permitted source is a
precommitted `FRESH_PERTURBED_STATE_CONFIRMATION` split made from previously
unused timesteps. It is explicitly not a new-episode claim. Replay validation
must use separate sacrificial calibration states.

The final gate uses 10,000 paired episode/state-clustered bootstrap replicates.
`GO_TO_SMALL_BC` requires pooled gain at least 10%, lower 95% CI above zero,
3/4 tasks and 2/3 contact tasks improving, action-RMSE degradation at most 20%,
contact drop at most one point, utilization at least 0.25, zero clipping,
context-shuffled gain retention at most 25%, and the same improvement direction
for all three training seeds.

## Final disposition

Return exactly one of:

`REJECT_CONSEQUENCE_HEADROOM`, `REJECT_LEARNED_CONSEQUENCE_METRIC`,
`STATIC_EFFECT_METRIC_ONLY`, `PIVOT_TO_CONSEQUENCE_RETRIEVAL_STEERING`,
`NARROW_TO_CONTACT_CONSEQUENCE_METRIC`, `BLOCKED_NO_FRESH_CONFIRMATION`,
`CONFIRMATION_FAILED`, or `GO_TO_SMALL_BC`.
