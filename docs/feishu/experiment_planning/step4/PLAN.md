# step4

Task: Execute R13-P15 Stage 3:  
Nominal-Conditioned Effect-Ranking Action Alphabet Audit (NCER-AA).

Repository:  
[https://github.com/mikasaTu/R13-P15-Consequence-Adaptive-Action-Alphabets](https://github.com/mikasaTu/R13-P15-Consequence-Adaptive-Action-Alphabets)

Create branch:

r13-p15-stage3-nominal-effect-ranking

Create output root:

experiments/r13_p15_ncer_aa/stage3/

Scientific status:

- Stage 1 remains REJECT_CORE_HYPOTHESIS.
- Stage 1.5 remains REJECT_P15_FAMILY.
- Stage 2 remains ORACLE_ONLY_NO_DEPLOYABLE_MODEL.
- Do not modify, delete, reinterpret, or relabel any historical artifact.
- Stage 3 is a new preregistered method hypothesis.
- It cannot retroactively rescue CAAA-v2 or the old NCEA implementation.

Scientific question:

Can an observable, nominal-action-conditioned model learn the local physical  
effect equivalence between a target action chunk and executable candidate  
chunks well enough to construct a K=64 dynamic action alphabet?

Primary new mechanism:

Given observable history h, nominal chunk a0, target residual dt and candidate  
residual dc, learn:

```
d_theta(h, a0, dt, dc)
≈ BALANCED_TASK_EFFECT(
      F(s, a0 + dt),
      F(s, a0 + dc)
  )
```

The primary method is NCER-AA:

1. a nominal-conditioned bi-encoder embeds all M=256 executable residuals;
2. K=64 candidates are selected by deterministic effect-space medoids/FPS;
3. a nominal-conditioned pairwise cross-encoder reranks the K candidates;
4. the selected bank action is executed in the simulator.

Hard scope:

- Use the existing four LIBERO tasks:  
bowl_on_plate  
plate_push  
stove_turn_on  
wine_rack
- Panda, OSC_POSE, 20 Hz, H=4, three settle steps.
- State-based mechanism audit only.
- Do not train ACT, Diffusion Policy, SmolVLA, pi0.5, DINO-WM or any policy.
- Use CPU simulation and at most one GPU for small predictor training.
- Do not use PAI unless local execution is impossible.
- Do not generate HTML.
- Do not build formal activation, cryptographic provenance, mutation farms,  
custom publication infrastructure, or large orchestration systems.
- Use ordinary Git, Markdown, JSON, CSV, Parquet, NPZ/Zarr and pytest.
- Stop after Stage 3.
- Do not automatically start behavior cloning.

Historical evidence boundary:

- Episodes 0–15 remain historical only.
- Existing Stage 2 results may be used only as historical/development evidence.
- No Stage 2 confirmation result was executed.
- Running, synthetic, local smoke or unit-test PASS is not a method result.

Required artifacts:

experiments/r13_p15_ncer_aa/stage3/  
├── PREREGISTRATION.md  
├── INPUT_BINDING.json  
├── episode_split.json  
├── support_codebooks.npz  
├── action_bank_binding.json  
├── model_definitions.json  
├── training_pairs.parquet  
├── predictor_metrics.csv  
├── retrieval_metrics.csv  
├── development_quantization.csv  
├── mechanism_controls.csv  
├── development_gate.json  
├── confirmation_quantization.csv  
├── bootstrap_results.json  
└── STAGE3_REPORT.md

Commit PREREGISTRATION.md, all episode IDs, support-generation seeds, method  
definitions, primary metric, model-selection rules and gates before collecting  
or inspecting any new calibration/development/confirmation result.

1. Freeze episode splits

Per task:

historical:  
IDs 0–15

train:  
IDs 16–31

calibration:  
IDs 32–35

development:  
IDs 36–39

confirmation:  
IDs 40–49

Verify that every episode is successful and hash every episode.

Do not execute confirmation target or candidate branches until all development  
gates pass.

1. Reuse and extend historical training data

The frozen Stage 2 support branches for episodes 16–31 may be reused only after  
their hashes and simulator settings are verified.

Collect missing M=256 candidate-bank consequences for every training state.

The training split may contain all previous Stage 2 train/calibration/development  
results because it is no longer confirmatory.

1. Generate genuinely separated fresh supports

Generate one fixed 24-direction codebook for each of:

- calibration
- development
- confirmation

Each codebook contains:

- 8 smooth temporal DCT directions;
- 8 suffix-localized contact directions;
- 8 low-rank temporal-action directions.

Use two deterministic radii in [0.04, 0.12] and antithetic signs.

Hard checks:

- exact direction overlap between splits = 0;
- exact residual overlap between splits = 0;
- maximum absolute cosine similarity between any two split codebooks <= 0.90;
- target residual may not equal any action-bank residual;
- every action must be executable without clipping.

If the cosine-separation constraint cannot be met, stop with:

BLOCKED_SUPPORT_SEPARATION

Do not relax it after results are visible.

1. Freeze the common executable bank

Use the existing train-only M=256 residual bank if and only if its exact hash,  
action semantics and validity checks pass.

Primary alphabet size:

K=64

Do not evaluate K=32 or K=128 until the final Stage 3 disposition is frozen.

1. Predictor inputs

Permitted inputs:

- current observable state vector and mask;
- previous two observable state deltas;
- previous two executed actions;
- current observable contact indicator;
- nominal H=4 action chunk a0;
- target and/or candidate residual action chunks;
- task identity.

Forbidden inputs:

- future state;
- future consequence;
- target or candidate simulator outcome;
- episode outcome;
- target ID or candidate-bank ID;
- demonstration-derived future phase label;
- confirmation result;
- oracle contact mode after execution.

The primary proposed model may not use hard  
free_space/pre_contact/contact_onset/post_contact labels.

A hard-phase model may be evaluated only as a diagnostic upper bound.

1. Baselines

Implement and freeze:

B1:  
centered covariance residual alphabet.

B2:  
observable-current-contact-conditioned residual k-means.

B2_PRIV:  
demonstration hard-phase residual k-means, diagnostic only.

B3:  
dynamic action-space medoids.

B4:  
state-conditioned action-only VQ.

B5:  
local kNN/kernel consequence model using state, nominal action and train  
branch outcomes.

C0:  
exact reproduction of the Stage 2 NCEA input and loss.

1. Proposed models

C1 — NC_VECTOR:

```
(state, history, a0, residual)
-> full consequence vector and contact logits
```

Use the frozen balanced-group Huber target, not raw unweighted MSE alone.

C2 — NC_TEMPORAL_VECTOR:

Use the same inputs but encode a0 and residual as H x 6 sequences with a small  
temporal encoder rather than flattening them.

C3 — NC_BIENCODER:

```
e_theta(state, history, a0, residual)
```

Learn an effect-equivalence embedding.

C4 — NC_PAIR_RANKER:

```
d_theta(state, history, a0, target_residual, candidate_residual)
```

Predict the true balanced target-candidate effect distance.

Use a symmetric pair construction and require:

```
d(target, candidate) == d(candidate, target)
d(action, action) ≈ 0
```

C5 — NCER_AA:

- C3 selects K=64 candidates from M=256 in predicted effect space;
- C4 reranks the K candidates;
- execute the selected bank action.

C6 — SOFT_MIXTURE_NCER_AA:

Use observable-history-conditioned soft expert weights.  
Do not use the true demonstration phase as the routing label at inference.

1. Training objectives

For C3/C4 use a frozen combination of:

distance regression:

```
Huber(predicted_distance, true_balanced_effect_distance)
```

pairwise ranking:

```
candidate i should outrank candidate j
whenever its true effect is closer to the target
```

listwise matching:

```
p_true(candidate) ∝ exp(-true_effect_distance / tau)
p_model(candidate) ∝ exp(-predicted_distance / tau_model)
```

Select lambda values and temperatures using calibration episodes only.

For every target, include:

- oracle top-8 positives;
- ranks 9–32 hard negatives;
- random negatives;
- candidates that change contact mode.

Do not train primarily on easy random negatives.

1. Mechanism controls

Train and evaluate:

- no_nominal_action;
- nominal_action_shuffled within task;
- state_shuffled within task;
- history_shuffled;
- consequence_labels_shuffled;
- soft-routing labels shuffled;
- action-only pair ranker;
- candidate-order permutation.

Candidate-order permutation must produce identical selected bank indices.

1. Predictor and retrieval metrics

Report:

- pairwise accuracy;
- candidate-distance Spearman;
- Kendall tau;
- NDCG@16;
- oracle-neighbor Recall@1 and Recall@8;
- mean oracle regret;
- O1–baseline gap fraction closed;
- per-task, per-phase and per-direction-family results;
- symmetry error;
- self-distance error;
- inference latency.

Oracle regret is:

```
true_effect_error(selected_candidate)
- minimum true_effect_error over all 256 candidates
```

1. Realized simulator metrics

Execute every selected candidate from the identical restored snapshot.

Report:

- BALANCED_TASK_EFFECT;
- per-group errors;
- object pose error;
- TCP-object relative-pose error;
- contact-mode preservation;
- task-progress error;
- action reconstruction RMSE;
- code utilization and perplexity;
- clipping rate;
- latency.

Never replace realized simulator metrics with predictor scores.

1. Sequential development gates

Gate A — strict-support oracle value

On episodes 36–39, the true-effect K=64 oracle must:

- improve pooled BALANCED_TASK_EFFECT by at least 20% versus the strongest  
deployable baseline;
- improve at least 3/4 tasks;
- improve at least 2/3 contact-sensitive tasks.

Failure disposition:

REJECT_CONSEQUENCE_EQUIVALENCE_ON_STRICT_SUPPORT

Gate B — learnable ranking

The best learned ranker must:

- reduce mean oracle regret by at least 25% versus the strongest learned or  
action-space baseline;
- improve NDCG@16 by at least 0.10;
- achieve Recall@8 >= 0.50;
- improve at least 3/4 tasks;
- improve at least 2/3 contact-sensitive tasks;
- have joint state+nominal shuffle retain at most 25% of the gain;
- have state-only and nominal-only shuffles each retain at most 50%;
- not be reproduced by consequence-label shuffle;
- be exactly invariant to candidate permutation.

Failure disposition:

ORACLE_ONLY_NO_LEARNABLE_RANKER

Gate C — K=64 action alphabet

NCER-AA must:

- improve realized pooled BALANCED_TASK_EFFECT by at least 10% versus the  
strongest deployable baseline;
- close at least 25% of the oracle gap;
- improve at least 3/4 tasks;
- improve at least 2/3 contact-sensitive tasks;
- degrade bowl_on_plate by no more than 5%;
- degrade action RMSE by no more than 20%;
- reduce contact-mode preservation by no more than 1 percentage point;
- have normalized K=64 utilization >= 0.25;
- have clipping rate = 0;
- not use the privileged hard-phase input.

If full-bank retrieval passes Gate B but K=64 fails, return:

LEARNABLE_RETRIEVAL_BUT_ALPHABET_COMPRESSION_FAILED

If Gate C passes, return:

DEVELOPMENT_PASSED_CONFIRMATION_REQUIRED

1. Untouched confirmation

Only after Gate A, B and C pass may episodes 40–49 be executed.

Do not retune:

- models;
- losses;
- action bank;
- K;
- thresholds;
- support codebooks;
- routing;
- consequence metric.

Use 10,000 paired episode-cluster bootstrap replicates.

GO_TO_SMALL_BC requires:

- pooled gain >= 10%;
- paired 95% CI lower bound > 0;
- at least 3/4 tasks improve;
- at least 2/3 contact-sensitive tasks improve;
- shuffle controls retain at most 25% of the gain;
- action RMSE degradation <= 20%;
- contact preservation degradation <= 1 percentage point;
- code utilization >= 0.25;
- clipping = 0.

Otherwise return:

CONFIRMATION_FAILED

1. Required final report

STAGE3_REPORT.md must clearly separate:

- historical evidence;
- reused training evidence;
- calibration evidence;
- development evidence;
- untouched confirmation evidence;
- oracle-only results;
- learned retrieval results;
- K=64 alphabet results;
- privileged diagnostic upper bounds;
- deployable results;
- mechanism controls;
- all negative runs.

It must explicitly answer:

1. Does nominal action a0 materially improve consequence prediction?
2. Does pairwise/listwise ranking outperform full-vector regression?
3. Is short observable history necessary?
4. Does a soft contact mixture outperform one global model?
5. Can learned retrieval recover a meaningful fraction of the true-effect oracle?
6. Does the gain survive K=64 alphabet compression?
7. Do state, nominal-action and label shuffles destroy the gain?
8. Is the result confirmed on episodes 40–49?
9. Is the mechanism ready for a small state-based BC experiment?

Stop after Stage 3.

Do not start ACT, Diffusion Policy, SmolVLA or pi0.5 automatically.  
Do not claim novelty, paper readiness, acceptance or VLA improvement from a  
development-only or oracle-only result.
