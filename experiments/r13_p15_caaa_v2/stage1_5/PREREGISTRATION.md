# R13-P15 Stage 1.5 Preregistration

## Status and evidence boundary

This document was committed before any Stage 1.5 revised-method result was
computed or inspected. Stage 1 remains frozen with disposition
`REJECT_CORE_HYPOTHESIS`; Stage 1.5 is a failure-localization and rescue audit,
not a reinterpretation of that result.

- Benchmark: the same four frozen `libero_goal` tasks used in Stage 1.
- Controller: Panda `OSC_POSE`, 20 Hz, normalized 7D actions, H=4.
- Alphabet coordinates: the 24 continuous pose coordinates; the four gripper
  commands are copied unchanged from the demonstration chunk.
- Primary alphabet size: K=64 only.
- Frozen primary outcome: three-step-settled realized consequence error under
  the Stage 1 consequence schema and train-only scaling.
- Development evidence: Stage 1 train/calibration/test episodes 0-15.
- Confirmatory evidence: fresh episodes 16-23 only if the internal screen
  passes. Stage 1 test episodes are never treated as confirmatory evidence for
  a revised method.
- No policy training, HTML, publication system, audit infrastructure, or
  automatic follow-on experiment is permitted.
- CPU is preferred; no more than one GPU may be visible if rendering requires
  it.

The frozen inputs and hashes are recorded in `STAGE1_INPUT_BINDING.json`.
Stage 1 files must remain byte-identical.

## Scientific questions

The audit distinguishes five candidate causes of the Stage 1 failure:

1. inaccurate or nonlinear local Jacobians;
2. applying perturbation Jacobians to uncentered full actions;
3. prototypes outside each state's reachable consequence subspace;
4. inverse amplification followed by clipping;
5. assignment/codebook collapse.

Retrospective associations are descriptive and not causal. A revised-method
claim requires a fresh untouched holdout.

## Frozen retrospective diagnostics

For each of the 256 Stage 1 snapshot states, compute:

- task, episode, split, phase and snapshot index;
- local R2 and normalized RMSE on radius-0.10 branches after a radius-0.05 fit;
- antithetic nonlinearity for matched direction/radius pairs,
  `||dy(+d)+dy(-d)|| / (||dy(+d)||+||dy(-d)||+epsilon)`;
- radius derivative drift between normalized radius 0.05 and 0.10;
- contact-mode switch rate across radius-0.10 branches;
- Jacobian singular values, truncated/effective rank and condition number;
- truncated-pseudoinverse operator norm;
- prototype reachable-subspace residual;
- decoded pre-clip action norm, clipped-coordinate fraction, assignment
  entropy/utilization and realized settled effect error;
- error contributions for object pose, TCP/object relative pose, contact and
  force, gripper/articulation, task progress and constraint violations.

Descriptive standardized least-squares regressions will relate realized error
to local-model error, center infeasibility, inverse amplification, clipping and
codebook collapse. The regressions will include task and phase indicators and
will be labeled diagnostic.

## Matched K=64 method matrix

Every deployable method uses the same frozen train/calibration episodes,
targets, consequence dimensions, action bounds, snapshots, branch rollouts and
gripper treatment.

- **M0 frozen CAAA-v2:** exact Stage 1 K=64 implementation and realized rows.
- **M1 covariance Mahalanobis:** exact Stage 1 calibration winner and realized
  rows.
- **M2 centered covariance residual:** quantize `delta_a=a-a0` in the frozen
  covariance-whitened space; decode and add `a0`.
- **M3 CARA:** quantize `T_s(a-a0)` with the frozen CAAA transform and decode
  `a0 + T_s^dagger c_k`.
- **M4 RECA:** learn K=64 prototypes in normalized realized consequence-delta
  space from train branches. At each state/token solve the bounded, radius
  constrained ridge problem
  `min ||J_s delta_a-u_k||_W^2 + beta||delta_a||_2^2`, subject to
  `a0+delta_a in [-1,1]^24` and `||delta_a||_2<=r`; retain the gripper. Beta,
  radius and a feasibility threshold are selected on calibration episodes
  only. Infeasible tokens are masked and are never pseudoinverted then clipped.
- **M5 phase-conditioned residual k-means:** a matched non-consequence
  state-conditioning control.
- **M6 permuted-J RECA:** cyclically permute Jacobians within task, split and
  physical phase while retaining the matched constrained solver.
- **M7 random-SPD constrained decoder:** deterministic matched-spectrum random
  geometry with the same bounds and solver.

All new fitting uses train episodes only. Choices among the following frozen
grids use calibration episodes only:

- RECA beta: `{1e-6, 1e-4, 1e-2, 1}`;
- residual radius cap: `{0.10, 0.20, 0.40}` in 24D normalized action space;
- feasibility threshold: calibration quantiles `{0.90, 0.95, 0.99}` of the
  selected decoder's prototype residuals.

Ties are broken by lower settled linear-effect error, then lower infeasible
rate, lower action reconstruction error, lower beta and lower radius cap, in
that order. No choice may use Stage 1 test episodes.

## Oracle diagnostics

These methods are diagnostics, never deployable methods.

- **O1 true-effect oracle:** for each radius-0.10 target, select from the 48
  signed radius-0.05 branches at the same state using settled realized
  consequence distance. The target branch itself cannot be in the dictionary.
- **O2 linear-J oracle:** use the same dictionary but select by `J_s delta_a`.

O1 versus M1 tests whether any local effect dictionary has upper-bound value;
O1 versus O2 isolates local-model loss; O2 versus RECA isolates global
prototype/decoder loss; M3 versus M0 isolates centering loss.

## Internal screen and stopping rule

The old Stage 1 test set is used only for the internal development screen. A
revised deployable method advances only if all conditions hold:

1. at least 5% lower settled realized effect error than M1 when pooled;
2. lower error on at least two of four tasks;
3. at least 80% lower clipped-coordinate rate than M0;
4. its improvement is not reproduced by M6 or M7;
5. action reconstruction error degrades by no more than 15% relative to M1.

If no method passes, do not collect fresh data. Materialize the required
fresh-holdout files as explicit `NOT_COLLECTED_INTERNAL_SCREEN_FAILED`
manifests, report `REJECT_P15_FAMILY`, and stop.

## Fresh holdout protocol if and only if the screen passes

Use successful demonstration IDs 16-23 for every task. Before computing any
revised result, verify success, SHA-256 and four phase indices, apply the
smallest-unused-successful-ID fallback if needed, freeze exactly eight episodes
per task, and commit `fresh_holdout_split.json`.

Collect the unchanged Stage 1 protocol: four phases, 24 deterministic
directions, signs +/-1, radii 0.05/0.10, H=4, three settle steps, identical
restore, frozen consequence schema/scaling/ridge/cutoff/K/solver settings and
no retuning.

## Metrics and uncertainty

Primary: settled realized consequence error. Secondary: immediate consequence
error, contact-mode preservation, task-progress preservation, action
reconstruction error, infeasible-token rate, clipped-coordinate rate,
normalized codebook perplexity, dead-code ratio, per-task/per-phase effects and
solver latency.

If a fresh holdout is collected, use 10,000 paired episode-cluster bootstrap
replicates within task. For an internal-screen rejection, bootstrap intervals
are explicitly retrospective and cannot support a revised-method claim.

## Frozen final dispositions

Return exactly one:

- `REJECT_P15_FAMILY` if O1 fails to beat covariance by 10%, no revised method
  passes the internal screen/fresh holdout, or geometry-destroying controls
  reproduce the effect.
- `CENTERING_FIX_ONLY` if CARA fixes clipping/reconstruction but does not beat
  centered covariance or phase-conditioned residual k-means.
- `NARROW_TO_CONTACT_PHASE` only if the pooled gate fails, contact-onset and
  post-contact improve by at least 10% with a positive paired 95% interval,
  free-space degradation is at most 5%, and M6 retains at most 25% of the gain.
- `GO_TO_SMALL_BC` only if every gate in the Step2 plan is satisfied: pooled
  gain at least 10% over the strongest matched baseline with positive paired
  95% lower bound, at least three tasks and two contact-sensitive tasks improve,
  bowl degradation at most 5%, control retention at most 25%, action
  reconstruction degradation at most 10%, infeasible rate below 5%, and
  clipping below 1%.

No K=32/K=128 inspection is allowed until the K=64 disposition is frozen. Stop
after Stage 1.5 and do not begin policy training.
