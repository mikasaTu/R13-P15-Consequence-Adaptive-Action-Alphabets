# R13-P15 Stage 3 Preregistration: NCER-AA

## Status and evidence boundary

This document freezes Stage 3 before any new calibration, development, or
episodes 40–49 method result is collected or inspected.  Stage 1 remains
`REJECT_CORE_HYPOTHESIS`, Stage 1.5 remains `REJECT_P15_FAMILY`, and Stage 2
remains `ORACLE_ONLY_NO_DEPLOYABLE_MODEL`.  Stage 3 is a new method hypothesis
and cannot retroactively relabel those outcomes.

The scientific question is whether an observable, nominal-action-conditioned
model can rank the physical-effect equivalence between a target H=4 action
chunk and an executable M=256 candidate bank well enough to form a dynamic
K=64 action alphabet.

## User-directed execution amendment

The supplied plan contains sequential development gates and normally locks
episodes 40–49 after a failed development gate.  Before results, the user
explicitly required every planned experiment to run and forbade gate-triggered
early stopping.  Therefore:

- every baseline, proposed model, control, development evaluation, and
  episodes 40–49 evaluation will run even after a failed gate;
- gates retain their original thresholds and determine the scientific
  disposition in their original A → B → C order;
- if A, B, or C fails, episodes 40–49 are labeled
  `FORCED_EXPLORATORY_HOLDOUT`, are not called untouched confirmation, cannot
  unlock `GO_TO_SMALL_BC`, and are never used for tuning;
- only if A, B, and C all pass and confirmation integrity remains intact could
  episodes 40–49 be interpreted under the original confirmation and GO criteria.

This amendment changes execution completeness, not evidentiary standards.

## Pre-result protocol incident and frozen consequence

Before any predictor, retrieval, quantization, calibration, or development
metric was computed or inspected, the first replay implementation executed one
fixed confirmation-support perturbation twice at each of the 160 confirmation
snapshots.  The executions were used only for deterministic equality/order
checks; no target-to-candidate distance, method score, selected bank action, or
aggregate physical-effect result was calculated, displayed, or used for a
choice.  Nevertheless, branch execution alone violates the literal rule that
confirmation target branches remain unexecuted until the development gates.

The complete incident is preserved in `PRE_RESULT_PROTOCOL_INCIDENT.json`.
The replay implementation is now frozen to three nominal-action repetitions
only.  Conservatively, episodes 40–49 cannot be described as strictly untouched
in this Stage 3 run and cannot unlock `GO_TO_SMALL_BC`.  If all development
gates pass their evidence label is `PRE_RESULT_REPLAY_EXPOSED_HOLDOUT`; if any
development gate fails it is `FORCED_EXPLORATORY_HOLDOUT`.  All requested
experiments still run, but this incident cannot be erased or reinterpreted.

## Frozen tasks, simulator, and splits

The tasks are `bowl_on_plate`, `plate_push`, `stove_turn_on`, and `wine_rack`
from standard `libero_goal`.  The robot is Panda, controller is `OSC_POSE` at
20 Hz, horizon is H=4, the six continuous controls are perturbed, the nominal
gripper command is copied at every step, and every branch has three settled
zero-delta-pose steps.  Simulation is CPU-only.  Predictor training may use at
most one local GPU.  PAI is allowed only if local completion is impossible.

Per task, episode IDs are frozen as:

- historical: 0–15;
- train: 16–31;
- calibration: 32–35;
- development: 36–39;
- confirmation/forced holdout: 40–49.

Every episode must be successful and is content-hashed in `episode_split.json`.
Episodes 0–15 remain historical only.  Stage 2 support branches for 16–31 can
be reused only after their completion markers, hashes, controller semantics,
and deterministic replay evidence verify.  Existing Stage 2 candidate effects
for 24–31 may be reused; missing M=256 candidate effects for 16–23 must be
collected from the identical restored snapshots.

## Strict fresh supports and common bank

`support_codebooks.npz` contains exactly one fixed 24-direction codebook for
each of calibration, development, and confirmation.  Each has eight smooth
temporal DCT, eight suffix-localized contact, and eight low-rank
temporal-action directions; two deterministic radii lie in [0.04, 0.12], and
both antithetic signs are used.

Before branch collection the freeze must prove:

- zero exact direction and residual overlap between splits;
- maximum cross-split absolute cosine similarity ≤ 0.90;
- no target residual equals a bank residual;
- every target and candidate action is executable without clipping.

Failure returns `BLOCKED_SUPPORT_SEPARATION`; no threshold may be relaxed after
results.  The exact Stage 2 train-only M=256 residual bank is reused only if
its file hash, residual hashes, action semantics, and per-snapshot validity
pass.  K=64 is primary.  K=32/128 stay locked until the Stage 3 disposition is
frozen and are not a substitute for a required K=64 result.

## Observable inputs and forbidden leakage

Permitted inputs are the current observable state and mask, previous two
observable state deltas and masks, previous two executed actions and
availability masks, current observable contact indicator, nominal H=4 chunk,
target/candidate residual chunks, and task identity.

Future state/consequence, candidate or target simulator outcomes at inference,
episode outcome, target/bank IDs, future demonstration phase, confirmation
results, and post-execution oracle contact mode are forbidden.  The primary
models cannot use hard phase.  Hard-phase k-means is diagnostic only.

## Frozen methods

The complete machine-readable definitions are in `model_definitions.json`.
They include B1 covariance residual medoids; B2 observable-current-contact
k-means; privileged B2 hard-phase k-means; B3 action-space FPS; B4
state-conditioned action-only VQ; B5 local kernel consequence prediction; C0
the Stage 2 NCEA input/loss reproduction; C1 nominal-conditioned vector
prediction; C2 temporal vector prediction; C3 a nominal-conditioned
bi-encoder; C4 a symmetric pair ranker; C5 NCER-AA; and C6 an observable soft
mixture.

C3 embeds all M=256 candidates, and deterministic predicted-effect FPS/medoids
select K=64.  C4 reranks only those K.  Its construction is symmetric and has
exact zero self-distance.  Candidate ties are resolved by the lowest frozen
bank index, so candidate-order permutation must return identical bank indices.

## Frozen objectives and calibration selection

Vector predictors use equal balanced-group Huber loss plus the fixed contact
loss, not raw unweighted MSE alone.  C3/C4 use the frozen combination of Huber
distance regression, pairwise ordering, and listwise matching.  Every target
contains oracle top-8 positives, ranks 9–32 hard negatives, deterministic
random negatives, and up to eight contact-changing candidates; training is not
dominated by easy random negatives.

Only episodes 32–35 select among the architecture and objective tuples listed
in `model_definitions.json`, including lambdas and temperatures.  Early
stopping, scaling, and model selection use train/calibration only.  No setting
can change after development becomes visible.  Episodes 40–49 are never used
for selection.

The exact frozen training contract uses three members for selected proposed
families, one member per matched mechanism control, 60 epochs maximum with
eight-epoch patience, Adam at 3e-4 and weight decay 1e-5.  Candidate
architectures are evaluated with one frozen seed; only the calibration-selected
candidate receives two additional members.  Pair models use 192x192 hidden
layers and a 32-D bi-encoder embedding.  C6 has four experts and a 64-unit
router; its optional auxiliary routing label is
`2*current_contact + 1[previous task-progress delta >= 0]`, which is entirely
observable and is not a hard phase.  C0 fixes the previously selected Stage 2
128x128 NCEA architecture, five members, the original 160-epoch/20-patience
contract, batch 512, Adam 1e-3, and the original smooth-L1 plus 0.25 contact
loss.  B4 fixes a 32-D latent and 128x128 action autoencoder.  B5 selects
neighbors {3,5,9} and bandwidth {0.5,1,2} by calibration only.

The strongest deployable comparator is the lowest pooled calibration
`BALANCED_TASK_EFFECT` among B1–B5 and C0.  Gate B's learned/action comparator
is the lowest pooled calibration oracle regret among B3–B5 and C0–C3.  Neither
development nor episodes 40–49 may select a comparator.  Ranking-objective
selection is lexicographic: lowest calibration mean oracle regret, highest
NDCG@16, then frozen candidate order.

## Frozen controls

The audit trains/evaluates no nominal action, within-task shuffled nominal
action, within-task shuffled state, joint state+nominal shuffle, shuffled
history, shuffled consequence labels, shuffled soft-routing labels,
action-only ranker, and candidate-order permutation.  Each shuffle uses a
frozen seed and preserves the indicated task strata.  All negative and failed
runs are retained.

## Metrics

Predictor/retrieval metrics are pairwise accuracy, candidate-distance
Spearman, Kendall tau, NDCG@16, oracle-neighbor Recall@1/8, mean oracle regret,
O1–baseline gap fraction closed, symmetry error, self-distance error, and
latency, pooled and by task, phase, and direction family.

Every selected bank action is evaluated from the exhaustive branch executed
from the identical restored simulator snapshot.  Realized metrics are
`BALANCED_TASK_EFFECT`, all five group errors, object-pose and TCP-object
relative-pose errors, contact-mode preservation, task-progress error, action
RMSE, code utilization/perplexity, clipping, and latency.  Predictor scores
never replace realized outcomes.

## Frozen development gates and disposition order

Gate A requires the strict-support true-effect K=64 oracle to improve pooled
`BALANCED_TASK_EFFECT` by at least 20% versus the strongest deployable
baseline, improve at least 3/4 tasks, and at least 2/3 contact-sensitive tasks.
Failure disposition is `REJECT_CONSEQUENCE_EQUIVALENCE_ON_STRICT_SUPPORT`.

Gate B requires the best learned ranker to reduce mean oracle regret by at
least 25%, improve NDCG@16 by at least 0.10, reach Recall@8 ≥ 0.50, improve at
least 3/4 tasks and 2/3 contact-sensitive tasks, satisfy all shuffle-retention
limits, not be reproduced by label shuffle, and be exactly candidate-order
invariant.  Failure disposition is `ORACLE_ONLY_NO_LEARNABLE_RANKER`.

Gate C requires NCER-AA to improve realized pooled error by at least 10%, close
at least 25% of the oracle gap, improve at least 3/4 tasks and 2/3
contact-sensitive tasks, keep bowl degradation ≤5%, action-RMSE degradation
≤20%, contact-preservation drop ≤1 percentage point, normalized utilization
≥0.25, clipping exactly zero, and use no privileged phase.  If B passes but C
fails, disposition is `LEARNABLE_RETRIEVAL_BUT_ALPHABET_COMPRESSION_FAILED`;
otherwise pass disposition is `DEVELOPMENT_PASSED_CONFIRMATION_REQUIRED`.

## Episodes 40–49 and bootstrap

All episode-40–49 selected actions will execute after development settings are
frozen.  Ten thousand paired episode-cluster bootstrap replicates will be
computed.  Because of the frozen pre-result replay incident, these calculations
are reported as exposed or forced exploratory holdout evidence, never as
strictly untouched confirmation; `GO_TO_SMALL_BC` is unavailable in this run.

## Hard exclusions

No ACT, Diffusion Policy, SmolVLA, π0.5, DINO-WM, or other policy is trained.
No HTML, activation system, cryptographic provenance layer, mutation farm, or
custom publication system is built.  Ordinary Git, Markdown, JSON, CSV,
Parquet, NPZ/Zarr, and pytest are used.  Work stops after Stage 3 and no small
BC starts automatically.
