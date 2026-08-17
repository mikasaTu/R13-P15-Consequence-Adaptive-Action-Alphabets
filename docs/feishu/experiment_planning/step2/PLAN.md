<title>step2</title>

# Agent Prompt：R13-P15 Stage 1.5 Failure Localization and Rescue Audit

Continue work in:

`https://github.com/mikasaTu/R13-P15-Consequence-Adaptive-Action-Alphabets`

The completed Stage 1 result is frozen as:

`REJECT_CORE_HYPOTHESIS`

Do not modify, overwrite, reinterpret, or delete any Stage 1 artifact.

The purpose of this task is not to rescue the result by parameter tuning. The purpose is to determine why CAAA-v2 failed and whether a scientifically distinct, preregistered residual/effect-alphabet construction survives on a new untouched holdout.

## Scientific questions

Test four possible failure sources:

1. The local Jacobian is inaccurate or nonlinear.
2. The method incorrectly applies a locally fitted perturbation Jacobian to uncentered full actions.
3. Global codebook centers are infeasible under the current state's reachable consequence subspace.
4. Unconstrained pseudoinverse decoding amplifies actions and clipping destroys the intended consequence geometry.

## Hard evidence boundary

- Do not train ACT, Diffusion Policy, SmolVLA, π0.5, DINO-WM, or any policy.
- Do not claim that the original CAAA-v2 passed.
- Do not use Stage 1 test results as confirmatory evidence for the revised method.
- Do not change the frozen consequence schema or primary error metric for the main test.
- Do not generate HTML.
- Do not build formal activation, custom publication, cryptographic audit, mutation, or release infrastructure.
- Reuse the existing Git, NPZ/Zarr, CSV, Parquet and Markdown pipeline.
- Use no more than one GPU if rendering requires it; otherwise use CPU simulation.
- Stop after Stage 1.5.

Create a new branch and output root:

```Plain Text
branch:
  r13-p15-stage1_5-failure-localization

experiments/r13_p15_caaa_v2/stage1_5/
├── PREREGISTRATION.md
├── STAGE1_INPUT_BINDING.json
├── retrospective_diagnostics.parquet
├── error_decomposition.csv
├── fresh_holdout_split.json
├── fresh_branch_rollouts.zarr
├── method_definitions.json
├── quantization_results_by_task.csv
├── quantization_results_by_phase.csv
├── mechanism_controls.csv
├── bootstrap_results.json
└── STAGE1_5_REPORT.md
```

Commit `PREREGISTRATION.md` and `STAGE1_INPUT_BINDING.json` before inspecting any revised-method result.

## Part A — Bind and preserve Stage 1

Record and verify:

- current repository commit and tree;
- Stage 1 formal commit;
- LIBERO commit;
- environment lock;
- branch rollout hash;
- Jacobian metrics hash;
- codebook hash;
- quantized-result hashes;
- Stage 1 final disposition.

Stage 1 must remain byte-identical.

## Part B — Retrospective failure localization

Use only the existing Stage 1 frozen snapshots and rollouts.

For every state, compute:

- task, episode and phase;
- local R² and normalized RMSE;
- antithetic nonlinearity score:

[

n_s=

\frac{|\Delta y(+\delta)+\Delta y(-\delta)|}

{|\Delta y(+\delta)|+|\Delta y(-\delta)|+\epsilon}

]

- radius derivative drift between 0.05 and 0.10;
- contact-mode switch rate;
- Jacobian effective rank and condition number;
- transform singular spectrum;
- pseudoinverse operator norm;
- code-center reachable-subspace residual;
- decoded action norm before clipping;
- clipped-coordinate fraction;
- code assignment entropy and utilization;
- realized effect error;
- per-consequence-group error contribution.

Separate consequence groups into at least:

- object pose;
- TCP/object relative pose;
- contact and force;
- gripper/articulation;
- task progress;
- constraint violations.

Fit descriptive, not causal, regressions to determine whether realized error is more strongly associated with:

- local-model error;
- center infeasibility;
- inverse amplification;
- clipping;
- codebook collapse.

This retrospective analysis is diagnostic only.

## Part C — Implement the matched method matrix

Primary alphabet size is `K=64`.

Do not inspect K=32 or K=128 until the K=64 primary disposition is frozen.

All deployable methods must use the same:

- training episodes;
- calibration episodes;
- codebook size;
- target actions;
- consequence dimensions;
- action bounds;
- branch rollouts;
- simulator snapshots;
- gripper treatment.

Implement:

### M0 — Frozen current CAAA-v2

Reproduce the exact Stage 1 implementation without alteration.

### M1 — Covariance Mahalanobis

The frozen Stage 1 calibration winner.

### M2 — Centered covariance residual alphabet

For each state:

[

\delta a=a-a_0

]

Fit and quantize the residual rather than the full action, then decode and add back (a_0).

This is the strongest non-consequence centering control.

### M3 — Centered CAAA residual alphabet, CARA

Use:

[

z_s=T_s(a-a_0)

]

Decode:

[

\hat a=a_0+T_s^\dagger c_k

]

Use exactly the same consequence Jacobian, regularization, K and data as M0.

This isolates absolute-versus-residual centering.

### M4 — Reachability-Constrained Effect Alphabet, RECA

Learn prototypes directly in normalized realized consequence-delta space using train episodes only:

[

u_k \in \Delta y

]

At each state, decode each prototype by solving:

# [  
\delta a\_{s,k}

\arg\min\_{\delta a}

|J_s\delta a-u_k|\_W^2

\+

\beta|\delta a|\_2^2

]

subject to:

[

a_0+\delta a\in[-1,1]^d

]

[

|\delta a|\_2\le r

]

The gripper command remains unchanged.

For every state/token record:

[

f\_{s,k}=|J_s\delta a\_{s,k}-u_k|\_W

]

Select the feasibility threshold and (\beta) using calibration episodes only.

Mask infeasible tokens. Do not decode an infeasible prototype with an unconstrained pseudoinverse and then clip it.

### M5 — Phase-conditioned residual k-means

A non-consequence state-conditioning control using the same four physical phases.

### M6 — Permuted-J RECA

Permute Jacobians only within the same task, split and physical phase.

Keep the same Jacobian spectrum distribution, codebook size and solver.

### M7 — Random-SPD constrained decoder

Use a matched random metric spectrum and the same constrained decoder.

## Part D — Oracle diagnostic upper bounds

These are diagnostics and must never be presented as deployable methods.

For each state:

### O1 — Local true-effect oracle

Use radius-0.05 local branches as a local dictionary.

For every radius-0.10 target action, choose the radius-0.05 action whose realized consequence is closest to the target's realized consequence.

Do not allow the target branch itself into the dictionary.

### O2 — Local linear-J oracle

Use the same dictionary, but choose by (J_s\delta a) rather than true realized consequence.

Interpretation:

- O1 versus covariance measures whether a consequence-local alphabet has any upper-bound value.
- O1 versus O2 measures local-model loss.
- O2 versus RECA measures global prototype and decoder loss.
- CARA versus current CAAA measures centering loss.

## Part E — Internal development screen

Use the original Stage 1 train/calibration episodes for all fitting and hyperparameter selection.

The original Stage 1 test episodes may be used only for retrospective diagnosis and an internal screen. They cannot support the final revised-method claim.

Continue to a new holdout only if at least one revised deployable method:

- improves realized effect error over covariance by at least 5% on the old test set;
- improves at least two tasks;
- reduces clipping by at least 80% relative to current CAAA;
- is not reproduced by permuted-J or random-SPD;
- has no greater action reconstruction degradation than 15%.

If no method passes this internal screen, return:

`REJECT_P15_FAMILY`

and stop without collecting new data.

## Part F — Freeze a fresh untouched holdout

If Part E passes, select new demonstrations that were not used in Stage 1.

Preferred IDs per task:

```Plain Text
16–23
```

Before collecting any revised result:

1. verify that all selected demonstrations are successful;
2. if an ID is invalid, choose the smallest unused successful episode ID by a deterministic rule;
3. freeze exactly eight new episode IDs per task;
4. freeze their SHA256 values and phase snapshot indices;
5. commit `fresh_holdout_split.json`.

Do not reuse episodes 0–15.

For each new episode, collect the same:

- four phases;
- 24 perturbation directions;
- signs ±1;
- radii 0.05 and 0.10;
- H=4 action chunk;
- three settle steps;
- consequence schema;
- deterministic snapshot restore.

Use the already frozen ridge, cutoff, consequence scaling, K and solver choices. Do not retune on the new holdout.

## Part G — Fresh-holdout primary metrics

Execute every decoded action from the identical restored simulator snapshot.

Primary metric:

- settled realized consequence error under the frozen Stage 1 consequence metric.

Secondary metrics:

- immediate consequence error;
- contact-mode preservation;
- task-progress preservation;
- action reconstruction error;
- infeasible-token rate;
- clipping rate;
- normalized codebook perplexity;
- dead-code ratio;
- per-phase and per-task performance;
- solver latency.

Use paired episode-clustered bootstrap with 10,000 replicates.

## Part H — Final gates

Return exactly one final disposition.

### `REJECT_P15_FAMILY`

Use when either:

- O1 local true-effect oracle fails to beat covariance by 10%; or
- no revised deployable method beats covariance on the fresh holdout; or
- permuted-J/random-SPD reproduce the effect.

### `CENTERING_FIX_ONLY`

Use when CARA fixes clipping and reconstruction but does not beat centered covariance or phase-conditioned residual k-means.

This is an implementation diagnosis, not a paper result.

### `NARROW_TO_CONTACT_PHASE`

Use only when:

- pooled full-task gate fails;
- contact-onset/post-contact subsets show at least 10% improvement;
- the paired 95% CI supports the contact-phase gain;
- free-space degradation is at most 5%;
- permuted-J retains at most 25% of the contact-phase gain.

### `GO_TO_SMALL_BC`

Requires all:

1. best consequence method improves pooled settled effect error by at least 10% over the strongest matched baseline;
2. paired episode-clustered 95% CI lower bound is above zero;
3. at least three of four tasks improve;
4. at least two of the three contact-sensitive tasks improve;
5. bowl-on-plate degradation is at most 5%;
6. permuted-J retains at most 25% of the gain;
7. random-SPD does not reproduce the gain;
8. action reconstruction error degrades by at most 10%;
9. infeasible-token rate is below 5%;
10. clipping rate is below 1%.

## Required report

`STAGE1_5_REPORT.md` must state:

- why Stage 1 remained rejected;
- exact Stage 1 inputs and hashes;
- whether the dominant failure was local nonlinearity, uncentered actions, infeasible prototypes, inverse amplification or clipping;
- retrospective versus confirmatory evidence;
- all method definitions;
- old-test internal-screen results;
- fresh holdout IDs;
- pooled, per-task and per-phase results;
- bootstrap confidence intervals;
- O1/O2 oracle gaps;
- CARA/current-CAAA centering gap;
- RECA/permuted-J mechanism gap;
- all failed and negative runs;
- the exact final disposition;
- the next permitted experiment.

Stop after Stage 1.5.

Do not automatically begin policy training.
