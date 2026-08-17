<title>step6</title>

# step6

# R13-P15 Stage 5 Agent Prompt — Context-Identifiable Consequence Retrieval and Dynamic Local Alphabet

## Repository

Continue work in:

`https://github.com/mikasaTu/R13-P15-Consequence-Adaptive-Action-Alphabets`

Create a new branch:

`r13-p15-stage5-context-identifiable-consequence-metric`

Create the output root:

`experiments/r13_p15_cicr_dla/stage5/`

## Historical scientific status

The following conclusions are immutable:

- Stage 1: `REJECT_CORE_HYPOTHESIS`
- Stage 1.5: `REJECT_P15_FAMILY`
- Stage 2: `ORACLE_ONLY_NO_DEPLOYABLE_MODEL`
- Stage 3: `ORACLE_ONLY_NO_LEARNABLE_RANKER`
- Stage 4: `STATIC_EFFECT_METRIC_ONLY`

Do not edit, delete, reinterpret, relabel, or overwrite any historical artifact.

Stage 4 established:

- strong state-specific true-effect oracle headroom;
- weak positive numerical performance relative to B2;
- no demonstrated causal dependence on state, nominal action, history, reversal labels, or true consequence labels;
- action-only, context-shuffled, no-reversal and shuffled-effect controls were competitive;
- the shared target/candidate embedding admitted the shortcut `embed(s,a)=f(s)+g(a)`, for which context cancels in the target-candidate difference;
- K=64 and action deviation remained unresolved.

Stage 5 is a new preregistered falsification experiment. It must not retroactively rescue any previous method.

## Primary scientific question

After eliminating additive context cancellation and explicitly separating static action geometry, static consequence geometry, and context-dependent consequence geometry:

> Can observable state, history and nominal action learn a context-specific physical-effect metric that outperforms an equally trained static consequence metric and its action-only, context-shuffled and consequence-label-shuffled controls?

Secondary question:

> If such a metric exists, can it be compressed into a genuinely executable dynamic K=64 local action alphabet without losing most of the full-bank retrieval gain or moving too far from the target action?

## Hard scope

- Use the existing four LIBERO tasks:

  - `bowl_on_plate`
  - `plate_push`
  - `stove_turn_on`
  - `wine_rack`
- Panda `OSC_POSE`, 20 Hz, H=4, three settle steps.
- State-based mechanism audit only.
- Do not train or modify ACT, Diffusion Policy, SmolVLA, pi0.5, DINO-WM or any VLA.
- A small state-based nominal trajectory generator is permitted only for producing genuinely new confirmation trajectories.
- The nominal generator is not a proposed method and its performance is not a Stage 5 result.
- Use CPU simulation where possible and at most one GPU for small model training.
- Do not submit PAI jobs unless local execution is technically impossible.
- Do not generate HTML.
- Do not construct publication, activation, mutation-farm or cryptographic infrastructure.
- Use ordinary Git, Markdown, JSON, CSV, Parquet, NPZ/Zarr and pytest.
- Stop after Stage 5.
- Do not automatically begin policy reranking, BC or VLA training.

## Required artifacts

Create:

```text
experiments/r13_p15_cicr_dla/stage5/
├── PREREGISTRATION.md
├── HISTORICAL_BINDING.json
├── DATA_PROTOCOL.json
├── LOCAL_BANK.npz
├── LOCAL_BANK_BINDING.json
├── ORACLE_ADAPTIVITY_AUDIT.csv
├── ORACLE_ADAPTIVITY_GATE.json
├── CONTEXT_REVERSAL_PAIRS.parquet
├── CONTEXT_REVERSAL_METADATA.json
├── MODEL_DEFINITIONS.json
├── MODEL_SELECTION.json
├── STATIC_METRIC_CHECKPOINTS/
├── CONTEXT_METRIC_CHECKPOINTS/
├── DEVELOPMENT_RANKING.csv
├── DEVELOPMENT_REALIZED.csv
├── DEVELOPMENT_CONTROLS.csv
├── DEVELOPMENT_GATE.json
├── NOMINAL_GENERATOR_BINDING.json
├── FRESH_TRAJECTORY_SEEDS.json
├── FRESH_CONFIRMATION_SPLIT.json
├── FRESH_BRANCH_MANIFEST.json
├── CONFIRMATION_RANKING.csv
├── CONFIRMATION_REALIZED.csv
├── BOOTSTRAP_RESULTS.json
├── FINAL_DISPOSITION.json
├── STAGE5_RELEASE_VERIFICATION.json
└── STAGE5_REPORT.md
```

Commit the following before computing or inspecting any Stage 5 development metric:

- `PREREGISTRATION.md`
- `HISTORICAL_BINDING.json`
- `DATA_PROTOCOL.json`
- local-bank selection rule and hash
- all model definitions
- all loss weights
- all seeds
- all gates
- all split rules
- all confirmation-generator rules

## Part 1 — Bind historical evidence

Record and verify:

- current repository commit and tree;
- Stage 1, 1.5, 2, 3 and 4 result commits;
- every historical final disposition;
- LIBERO commit and source-tree hash;
- environment lock;
- Stage 4 action-bank hash;
- Stage 4 consequence scaler hash;
- Stage 4 context scaler hash;
- Stage 4 training/development/exploratory/fresh result hashes;
- Stage 4 checkpoint hashes.

Verify that all Stage 1–4 paths are byte-identical to their published values.

Any mismatch must stop with:

`BLOCKED_HISTORICAL_BINDING_MISMATCH`

## Part 2 — Freeze the primary metric

Use the exact frozen `BALANCED_TASK_EFFECT` definition from Stage 2–4:

- object pose;
- TCP-object relative pose;
- contact mode and penetration;
- gripper and articulation;
- task progress and constraint violation;
- equal group weights;
- train-only robust scales;
- capped Huber;
- raw force excluded from the primary metric.

Do not refit or retune consequence scales.

The old Stage 1 force-dominated metric may be reported only as a secondary continuity diagnostic.

## Part 3 — Build a local executable candidate bank

The primary bank size is:

`M=128`

Construct it deterministically from the frozen Stage 4 M=256 residual bank.

Selection must be frozen before metric results:

1. compute train-covariance-whitened residual norm;
2. retain the most local executable residuals;
3. preserve balance across:

   - smooth-DCT;
   - suffix-contact;
   - low-rank temporal-action;
   - signs;
   - temporal support positions;
4. deduplicate near-identical residuals;
5. forbid target residual equality;
6. preserve original candidate indices;
7. require no clipping or action synthesis;
8. require at least 96 valid candidates at every evaluated state.

Save exact residuals, source indices, family IDs, norms and hashes.

Use M=128 as primary.

M=256 may be reported only as a post-disposition sensitivity analysis.

Primary alphabet size:

`K=64`

Do not inspect K=32 or K=96/128 until the final Stage 5 disposition is frozen.

## Part 4 — Oracle adaptivity decomposition

Before training a new model, evaluate on development episodes 36–39:

### O_STATE_FULL

For each state and target, select the candidate with minimum true current-state `BALANCED_TASK_EFFECT`.

### O_STATE_K64

Use true current-state effect distances to select a K=64 state-specific atlas, then decode each target through that atlas.

### O_STATIC_FULL

For every target/candidate residual pair, average true effect distance over training states. Use this state-independent table for retrieval.

### O_CONTACT_FULL

Average true effect distance conditional only on current observable contact.

### O_PHASE_FULL

Average true effect distance conditional on the privileged four-phase label.

`O_PHASE_FULL` is diagnostic only.

Report:

- state-specific versus static oracle gap;
- state-specific versus current-contact oracle gap;
- state-specific versus privileged-phase oracle gap;
- full-bank versus K=64 oracle compression loss;
- per-task;
- per-phase;
- per-direction-family;
- per-consequence-group.

### Oracle adaptivity Gate 0

`O_STATE_FULL` must outperform the strongest of `O_STATIC_FULL` and `O_CONTACT_FULL` by:

- at least 8% pooled;
- at least 12% over contact-onset and post-contact pooled;
- at least two of three contact-sensitive tasks.

The strict reversal dataset must additionally contain:

- at least 1,000 valid pairs;
- at least two contact phases with reversal rate at least 15%.

If this gate fails, return:

`STATIC_EFFECT_GEOMETRY_SUFFICIENT`

and stop before training a context metric.

## Part 5 — Construct a strict context-reversal benchmark

Use the same target residual and candidate pair across different states.

A valid reversal requires:

```text
D_s1(target, candidate_i) + margin
<
D_s1(target, candidate_j)

D_s2(target, candidate_j) + margin
<
D_s2(target, candidate_i)
```

Requirements:

- pair states within the same task;
- prefer the same current-contact category;
- do not use demonstration phase as a proposed-model input;
- balance task, contact phase, direction family and target ID;
- use a train-only robust margin;
- do not relax the margin after observing shortages;
- do not fabricate reversal labels;
- do not place the same episode in multiple splits;
- report undersupplied strata explicitly;
- prevent exact reversal-tuple overlap across train, calibration and development.

Use:

- episodes 16–31 for training reversal pairs;
- episodes 32–35 for calibration;
- episodes 36–39 for development;
- episodes 40–49 only as historical exploratory evidence.

## Part 6 — Model hierarchy

Train and compare the following matched-capacity methods.

### B0 — Current-contact residual k-means

Reproduce the strongest deployable action-space baseline used in Stage 4.

### B1 — Learned action-only metric

Use the same action encoder architecture as the proposed model.

Train without consequence labels.

The objective may use action reconstruction, covariance-whitened action distance and action-neighbor ranking only.

### B2 — Static consequence metric

Train a state-independent consequence metric using true consequence distances.

This is the primary comparator for context adaptivity.

### P1 — Context-Gated PSD Consequence Metric

First train and freeze the static action/consequence representation:

```text
z = psi(nominal_chunk, residual_action)
```

Then train only a context modulator:

```text
m_s = g(observable_context, nominal_chunk)
```

Primary distance:

```text
d_context(s, target, candidate)
=
sum_j [
  softplus(w0_j) * exp(m_s_j)
  * (z_target_j - z_candidate_j)^2
]
```

Requirements:

- exact symmetry;
- exact zero self-distance;
- no multiplication by raw action L2 distance;
- no additive shared-context term that cancels in target-candidate subtraction;
- `m_s=0` must exactly recover B2;
- report the context modulation vector and norm;
- use bounded modulation;
- enforce zero-mean modulation over matched training contexts.

### P2 — Low-Rank Hyper-PSD Metric

Optional second preregistered architecture:

```text
d(s,t,c)
=
||L_s (z_t-z_c)||^2
+ lambda ||z_t-z_c||^2
```

Calibration may choose P1 or P2.

Development and historical exploratory data may not choose the architecture.

### Matched controls

Train with identical architecture, parameter count, seeds, optimizer steps and budget:

- `ACTION_ONLY`
- `CONTEXT_SHUFFLED`
- `NOMINAL_SHUFFLED`
- `JOINT_STATE_NOMINAL_SHUFFLED`
- `CONSEQUENCE_LABEL_SHUFFLED`
- `NO_REVERSAL_LOSS`
- `PHASE_ONLY`
- `CURRENT_CONTACT_ONLY`

Candidate order permutation must leave selected candidate IDs exactly unchanged.

## Part 7 — Permitted and forbidden inputs

Permitted proposed-model inputs:

- current observable state and mask;
- previous two observable state deltas and masks;
- previous two executed actions and availability masks;
- current observable contact;
- nominal H=4 action chunk;
- target residual action chunk;
- candidate residual action chunk;
- task identity.

Forbidden inputs:

- future state;
- future consequence;
- target or candidate simulator outcomes at inference;
- demonstration-derived phase;
- episode success or future reward;
- candidate ID;
- target ID;
- row index;
- confirmation result;
- post-execution contact;
- oracle atlas membership.

## Part 8 — Training objectives

Use the frozen loss:

```text
L =
1.0 * L_distance
+ 0.5 * L_pairwise
+ 0.5 * L_listwise
+ 1.0 * L_reversal
+ 0.01 * L_gate
```

### Distance loss

```text
Huber(
  log1p(predicted_effect_distance),
  log1p(true_effect_distance)
)
```

### Pairwise loss

Include:

- oracle top-8 positives;
- ranks 9–32 hard negatives;
- contact-changing candidates;
- action-close but effect-far pairs;
- action-far but effect-close pairs.

Do not train primarily on easy random negatives.

### Listwise loss

Use the complete M=128 local bank:

```text
p_true(j) ∝ exp(-d_true_j / tau_true)
p_model(j) ∝ exp(-d_model_j / tau_model)
```

Temperatures are selected using calibration episodes only.

### Reversal loss

Use strict cross-state reversal pairs.

### Gate regularization

Penalize:

- nonzero mean context modulation;
- excessive modulation norm;
- unstable metric condition number.

Train three seeds for every primary and matched-control method.

## Part 9 — Development evaluation

Evaluate two proposed paths.

### FULL retrieval

Select the minimum predicted-effect candidate over all M=128 candidates.

### Dynamic K=64 alphabet

For every state:

1. compute candidate-candidate distances under the current predicted metric;
2. select deterministic K=64 medoids;
3. map each target to its nearest predicted-effect medoid;
4. execute the selected existing bank action.

Never:

- pseudoinvert;
- synthesize actions;
- clip;
- replace an invalid action with a clipped action.

## Part 10 — Development metrics

Report:

### Ranking

- joint context-reversal accuracy;
- side reversal accuracy;
- Spearman;
- Kendall tau;
- NDCG@16;
- Recall@1;
- Recall@8;
- mean oracle regret;
- selected-code change under context interventions;
- context modulation norm;
- per-task and per-phase results.

### Realized simulator outcomes

- `BALANCED_TASK_EFFECT`;
- five consequence-group errors;
- object pose error;
- TCP-object relative-pose error;
- contact-mode preservation;
- task-progress error;
- action reconstruction RMSE;
- normalized utilization;
- code perplexity;
- clipping;
- valid-bank size;
- inference latency.

Use paired episode-clustered bootstrap with 10,000 replicates.

## Part 11 — Development gates

### Gate 1 — Context-identifiable learned metric

The selected proposed FULL model must outperform B2 static consequence metric by:

- at least 5% realized pooled effect error;
- paired 95% CI lower bound above zero;
- at least three of four tasks;
- at least two of three contact-sensitive tasks;
- at least 10% mean oracle-regret reduction;
- at least 0.05 NDCG@16 improvement.

It must also achieve:

- joint reversal accuracy at least 0.35;
- joint reversal accuracy at least 15 percentage points above B2;
- all three seeds with the same improvement direction.

Mechanism requirements:

- joint state+nominal shuffle retains at most 25% of the incremental gain;
- consequence-label shuffle retains at most 25%;
- action-only retains at most 50% of the context increment;
- no-reversal-loss does not reproduce the reversal improvement.

If P1/P2 fails but B2 static consequence metric beats B1/action baselines, return:

`STATIC_CONSEQUENCE_METRIC_ONLY`

If neither P1/P2 nor B2 establishes consequence-label value, return:

`REJECT_LEARNED_CONSEQUENCE_METRIC`

### Gate 2 — Dynamic K=64 alphabet

The K=64 proposed method must:

- improve at least 8% over the strongest deployable K=64 baseline;
- retain at least 75% of the FULL incremental gain;
- improve at least three of four tasks;
- improve at least two of three contact-sensitive tasks;
- degrade action RMSE by at most 20%;
- reduce contact preservation by at most one percentage point;
- achieve normalized utilization at least 0.25;
- clip zero actions;
- have at least 96 valid local-bank candidates per state.

If FULL passes Gate 1 but K=64 fails Gate 2, return:

`PIVOT_TO_CONSEQUENCE_RETRIEVAL_STEERING`

Do not reinterpret a retrieval-only result as an action alphabet.

## Part 12 — Freeze a nominal trajectory generator before development results

A genuinely new confirmation set is required.

Preferred order:

1. use an already available frozen successful state-based policy checkpoint;
2. otherwise train one small state-based H=4 chunk BC generator using only official demonstrations 0–31.

The generator:

- is not a proposed R13-P15 method;
- is used only to produce nominal state/action trajectories;
- may not use Stage 5 consequence labels;
- must be frozen before any Stage 5 development result is inspected;
- must not be changed after seeing Stage 5 metrics;
- must use a precommitted rollout seed list;
- must use no image or VLA input;
- must not be evaluated as a paper contribution.

Freeze 200 rollout seeds per task in ascending deterministic order.

After all development choices are frozen, execute the seeds and retain the first 12 successful trajectories per task.

Acceptance uses only environment task success.

It must not use any consequence-metric result.

If fewer than 12 successful trajectories are available for any task, return:

`BLOCKED_NO_FRESH_TRAJECTORIES`

Do not replace the generator or alter the acceptance rule.

## Part 13 — Fresh confirmation firewall

Before executing any confirmation branch:

- freeze the selected architecture;
- freeze all three checkpoints;
- freeze B0/B1/B2 and controls;
- freeze M=128 bank;
- freeze K=64 atlas algorithm;
- freeze metrics;
- freeze thresholds;
- freeze all successful rollout IDs and seeds;
- freeze exact phase-state indices;
- commit `FRESH_CONFIRMATION_SPLIT.json`.

Use separate sacrificial generator trajectories for deterministic replay tests.

Do not execute a confirmation state for replay validation before the split is frozen.

For every task:

- use 12 new successful trajectories;
- select free-space, pre-contact, contact-onset and post-contact states;
- use 96 new target residuals generated from independent seeds;
- require exact target/candidate overlap zero;
- require exact historical target-support overlap zero;
- execute nominal, target and all M=128 candidate branches from identical restored states.

Label the evidence:

`FRESH_POLICY_TRAJECTORY_CONFIRMATION`

## Part 14 — Confirmation gate

The proposed K=64 method must satisfy all:

1. pooled realized effect gain at least 10% over the strongest deployable baseline;
2. paired 95% CI lower bound above zero;
3. at least three of four tasks improve;
4. at least two of three contact-sensitive tasks improve;
5. context-shuffled gain retention at most 25%;
6. consequence-label-shuffled gain retention at most 25%;
7. action RMSE degradation at most 20%;
8. contact-preservation drop at most one percentage point;
9. normalized utilization at least 0.25;
10. clipping equals zero;
11. all three training seeds improve in the same direction;
12. state-specific oracle retains at least 8% adaptive headroom over the strongest static/contact oracle.

If all pass, return:

`GO_TO_FIXED_POLICY_RERANKING`

Otherwise return:

`CONFIRMATION_FAILED`

## Exact final dispositions

Return exactly one:

- `BLOCKED_HISTORICAL_BINDING_MISMATCH`
- `STATIC_EFFECT_GEOMETRY_SUFFICIENT`
- `STATIC_CONSEQUENCE_METRIC_ONLY`
- `REJECT_LEARNED_CONSEQUENCE_METRIC`
- `PIVOT_TO_CONSEQUENCE_RETRIEVAL_STEERING`
- `BLOCKED_NO_FRESH_TRAJECTORIES`
- `CONFIRMATION_FAILED`
- `GO_TO_FIXED_POLICY_RERANKING`

## Required code tests

Add tests for:

- exact distance symmetry;
- exact zero self-distance;
- exact static-model recovery when context modulation is zero;
- context modulation changes distance for a synthetic reversal case;
- candidate-order permutation invariance;
- no candidate or target ID in model inputs;
- no future or phase field in proposed inputs;
- split-disjoint reversal tuples;
- local-bank balance and determinism;
- no clipping;
- deterministic K-medoids;
- bootstrap cluster identity;
- historical path immutability;
- confirmation firewall;
- exact-one-disposition logic.

Run the complete existing test suite plus Stage 5 tests.

Create a Stage 5 release verifier that checks:

- all artifact hashes;
- all checkpoints;
- all split bindings;
- all branch row counts;
- all completion markers;
- all bootstrap replicates;
- all gate values;
- all final-disposition logic;
- no policy/VLA training beyond the frozen nominal generator;
- zero PAI jobs unless explicitly documented as technically necessary.

## Required final report

`STAGE5_REPORT.md` must clearly separate:

- historical rejected evidence;
- oracle adaptivity evidence;
- static action geometry;
- static consequence geometry;
- context-dependent consequence geometry;
- development evidence;
- historical exploratory evidence;
- fresh policy-trajectory confirmation;
- FULL retrieval;
- K=64 alphabet;
- matched controls;
- all negative runs.

It must explicitly answer:

1. How much oracle value is truly state-dependent rather than static or contact-conditioned?
2. Does true consequence supervision outperform action-only learning?
3. Does observable context add value beyond a static consequence metric?
4. Does the proposed architecture pass strict context-reversal tests?
5. Do state, nominal-action and consequence-label shuffles destroy the gain?
6. Does FULL retrieval work?
7. Does K=64 preserve at least 75% of FULL gain?
8. Is action deviation within the registered budget?
9. Does the result reproduce on genuinely new successful trajectories?
10. Is the surviving method an adaptive alphabet, static metric, retrieval method, or rejected hypothesis?
11. Is the mechanism ready for fixed-policy test-time candidate reranking?

Stop after Stage 5.

Do not automatically train or modify ACT, Diffusion Policy, SmolVLA, pi0.5 or any VLA.

Do not claim paper readiness, novelty restoration, VLA improvement or task-success improvement from oracle-only, development-only, historical exploratory or branch-level evidence.
