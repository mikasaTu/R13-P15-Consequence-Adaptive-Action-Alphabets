# step5

Task: Execute R13-P15 Stage 4:  
C3-Faithful Context-Reversal and Trust-Region Consequence Alphabet Audit.

Repository:  
[https://github.com/mikasaTu/R13-P15-Consequence-Adaptive-Action-Alphabets](https://github.com/mikasaTu/R13-P15-Consequence-Adaptive-Action-Alphabets)

Create branch:

r13-p15-stage4-c3-context-trust-region

Output root:

experiments/r13_p15_cr_trca/stage4/

Scientific status:

- Stage 1 remains REJECT_CORE_HYPOTHESIS.
- Stage 1.5 remains REJECT_P15_FAMILY.
- Stage 2 remains ORACLE_ONLY_NO_DEPLOYABLE_MODEL.
- Stage 3 remains ORACLE_ONLY_NO_LEARNABLE_RANKER.
- Stage 3 C5/C6 are rejected.
- Stage 3 C3 is only a weak exploratory signal:  
development gain was about 8.1%, but the executed non-untouched holdout gain was  
only about 1.35%, with 2/4 tasks improving, about 70% action-RMSE degradation,  
and lower code utilization.
- Do not relabel or modify any historical artifact.
- Stage 4 is a new method hypothesis and cannot inherit the original R13-P15  
novelty grade.

Primary questions:

1. Is the Stage 3 C3 bi-encoder signal reproducible when C3 is used through the  
complete decoding path?
2. Is the remaining error caused by learned metric error, K=64 compression, or  
final decoder replacement?
3. Does the learned metric genuinely depend on current state, nominal chunk and  
observable history?
4. Can context-reversal training make consequence ordering state-dependent?
5. Can an action-local trust region preserve consequence gains without selecting  
actions far from the target?
6. Is the successful mechanism an action alphabet or only full-bank retrieval?

Hard scope:

- Keep the existing four LIBERO tasks:  
bowl_on_plate  
plate_push  
stove_turn_on  
wine_rack
- Panda OSC_POSE, 20 Hz, H=4, three settle steps.
- State-based mechanism audit only.
- Do not train ACT, Diffusion Policy, SmolVLA, pi0.5, DINO-WM or any policy.
- Use at most one GPU for small metric models.
- Use CPU simulation where possible.
- Do not submit PAI jobs.
- Do not generate HTML.
- Do not build activation, cryptographic, mutation-farm or publication  
infrastructure.
- Use ordinary Git, Markdown, JSON, CSV, Parquet and NPZ/Zarr.
- Stop after Stage 4.
- Episodes 40–49 are historical exploratory evidence only and may never be  
called untouched confirmation.
- Never execute a fresh confirmation state for replay testing before all  
development methods and settings are frozen. Use a sacrificial calibration  
state set for determinism tests.

Required artifacts:

experiments/r13_p15_cr_trca/stage4/  
├── PREREGISTRATION.md  
├── HISTORICAL_BINDING.json  
├── C3_FAILURE_DECOMPOSITION.csv  
├── CONTEXT_DEPENDENCE_AUDIT.csv  
├── C3_CONTEXT_INTERVENTIONS.csv  
├── FRESH_STATE_INVENTORY.json  
├── METHOD_DEFINITIONS.json  
├── TRAINING_STATE_MANIFEST.json  
├── CONTEXT_REVERSAL_PAIRS.parquet  
├── MODEL_SELECTION.json  
├── DEVELOPMENT_RETRIEVAL.csv  
├── DEVELOPMENT_REALIZED.csv  
├── DEVELOPMENT_GATE.json  
├── FRESH_CONFIRMATION_SPLIT.json  
├── CONFIRMATION_RETRIEVAL.csv  
├── CONFIRMATION_REALIZED.csv  
├── BOOTSTRAP_RESULTS.json  
└── STAGE4_REPORT.md

Commit PREREGISTRATION.md, historical hashes, all method definitions, data  
selection rules, model seeds, losses and gates before inspecting any Stage 4  
method result.

Part 1 — Bind historical evidence

Record and verify:

- current repository commit/tree;
- Stage 1, 1.5, 2 and 3 commits and dispositions;
- LIBERO commit and environment;
- Stage 3 C3/C4/C5 checkpoint hashes;
- action-bank hash;
- metric/scaler hashes;
- development and exploratory-holdout result hashes.

Historical files must remain byte-identical.

Part 2 — Frozen C3 failure decomposition

Use existing development and Stage 3 exploratory data only.

Evaluate without retraining:

B2:  
current-contact K=64 k-means.

O_FULL:  
true-effect nearest candidate over all M=256.

O_K64:  
true-effect K=64 atlas.

C3_FULL:  
frozen Stage 3 C3 embedding, nearest candidate over all 256.

C3_K64:  
frozen Stage 3 C3 embedding, deterministic K=64 atlas and C3 nearest-code  
decoding.

C5:  
frozen Stage 3 C3 atlas followed by C4 reranking.

Compute:

oracle bank-compression loss:  
O_K64 - O_FULL

learned metric loss:  
C3_FULL - O_FULL

learned compression loss:  
C3_K64 - C3_FULL

C4 override loss:  
C5 - C3_K64

Report these pooled and by task, phase, direction family and consequence group.

Part 3 — Context-dependence audit

For matched target residuals and candidate actions across different states,  
compute:

- context reversal rate;
- true top-8 candidate Jaccard across states;
- best-candidate churn;
- state-conditioned oracle versus global averaged-effect oracle;
- consequence-distance variance explained by state/task/phase/action.

A context reversal is valid only when:

D_s1(t,i) + margin < D_s1(t,j)  
D_s2(t,j) + margin < D_s2(t,i)

Use a train-only robust margin.

On the same frozen 3-member C3 checkpoint, run inference-time interventions:

- correct context;
- nominal action zeroed;
- nominal action shuffled within task;
- current state/mask/contact shuffled within task;
- history/actions/masks shuffled within task;
- state and nominal jointly shuffled;
- all context zeroed, action pair retained.

Do not retrain separate control models for this audit.

Report distance changes, selected-code changes, NDCG@16, Recall@8, oracle regret  
and realized simulator effect.

Part 4 — Reselect C3 independently

Retrain the four frozen Stage 3 ranking-objective tuples with the same C3  
architecture and fixed seeds.

Select the tuple using C3-alone calibration:

1. lowest C3_FULL oracle regret;
2. highest C3_FULL NDCG@16;
3. lowest frozen tuple index.

Do not select using C3-atlas plus C4 ranking.

Train three ensemble members only after selecting the tuple.

Evaluate:

C3_RESELECT_FULL  
C3_RESELECT_FPS64  
C3_RESELECT_KMEDOIDS64

K-medoids must be deterministic and use only predicted embeddings.  
No target simulator outcome may enter atlas construction.

Part 5 — Expand independent training contexts

The previous training set contained about one million pairs but only 256  
independent states.

Create 768–1024 train states by deterministically selecting additional unused  
timesteps from train episodes only.

Requirements:

- balance free-space, pre-contact, contact-onset and post-contact;
- balance smooth-DCT, suffix-contact and low-rank supports at 1:1:1;
- never use confirmation states;
- freeze all snapshot indices and hashes before collection;
- use the same M=256 executable action bank;
- validate snapshot restore on a sacrificial non-confirmation set.

Part 6 — Train Context-Reversal C3

Implement:

CR_C3_SHARED:  
one context-conditioned effect embedding.

CR_C3_GROUP:  
five factorized consequence-group embeddings whose distances are averaged with  
the frozen equal group weights.

Inputs:

- current observable state and masks;
- two-step observable history and masks;
- previous two executed actions and masks;
- current contact;
- nominal H=4 action chunk;
- task identity;
- target or candidate residual.

Forbidden inputs:

- future state;
- candidate/target simulator outcome at inference;
- demonstration phase label;
- episode outcome;
- target ID or candidate ID.

Use:

full-bank listwise loss over all 256 candidates;  
pairwise ranking loss;  
context-reversal loss.

For every target, sample balanced reversal examples across tasks and phases.

The primary reversal loss is:

softplus(m + d_s1(t,i) - d_s1(t,j))  
\+  
softplus(m + d_s2(t,j) - d_s2(t,i))

when the true candidate ordering reverses between s1 and s2.

Train three seeds.

Matched controls must use the same architecture, parameter count, ensemble size  
and training budget:

- action-only;
- context-shuffled;
- nominal-shuffled;
- consequence-label-shuffled;
- reversal-label-shuffled;
- no-reversal-loss.

Part 7 — Action-local trust-region decoder

For every K=64 atlas and target residual:

1. rank the 64 codes by train-covariance-whitened action distance;
2. keep the nearest L codes;
3. choose the code with minimum CR-C3 consequence distance.

Calibration may select:

L in {8, 16, 32, 64}

L=64 is the no-trust-region control.

Evaluate:

CR_C3_FULL  
CR_C3_K64  
CR_TR_C3_K64  
ACTION_ONLY_TR_K64  
SHUFFLED_EFFECT_TR_K64

The trust-region method must always output an executable bank action.  
No clipping, pseudoinverse or action synthesis is allowed.

Part 8 — Optional bounded correction diagnostic

Do not use the old raw-L2 C4 as the proposed decoder.

Only after CR-C3 is trained, optionally test:

d_final = d_C3 \* exp(gamma \* tanh(g_symmetric))

with:

gamma in {0.0, 0.1, 0.2}

Gamma is selected on calibration only.  
Gamma=0 is the exact sham.  
The correction may not replace or bypass C3 geometry.

This diagnostic is not required for Stage 4 success.

Part 9 — Development evaluation

Use episodes 36–39 as the primary development set.

Episodes 40–49 may be used only as a separately reported historical  
exploratory replication set.

Never pool them silently.

Metrics:

- BALANCED_TASK_EFFECT;
- per-group effect error;
- oracle regret;
- Spearman and Kendall;
- NDCG@16;
- Recall@1 and Recall@8;
- action reconstruction RMSE;
- contact-mode preservation;
- task-progress error;
- normalized code utilization;
- code perplexity;
- clipping;
- inference latency;
- context-reversal accuracy;
- context-shuffle gain retention.

Gate A — oracle headroom:

O_K64 must improve at least 20% over B2, with 3/4 tasks and 2/3  
contact-sensitive tasks improving.

Failure:

REJECT_CONSEQUENCE_HEADROOM

Gate B — learned consequence metric:

The best CR-C3 FULL method must:

- improve at least 5% over B2 on episodes 36–39;
- improve at least 5% over B2 on historical exploratory episodes 40–49;
- improve at least 8% on pooled development;
- improve at least 3/4 tasks;
- improve at least 2/3 contact-sensitive tasks;
- improve at least 5% over frozen C3;
- improve context-reversal accuracy by at least 10 percentage points;
- retain at most 50% of its gain under joint state+nominal shuffle;
- not be reproduced by action-only or label-shuffled controls.

Failure:

REJECT_LEARNED_CONSEQUENCE_METRIC

If the effect survives but context interventions do not matter, return:

STATIC_EFFECT_METRIC_ONLY

Gate C — K=64 alphabet:

CR-TR-C3 K64 must:

- improve BALANCED_TASK_EFFECT by at least 8% over B2;
- retain at least 75% of the CR-C3 FULL gain;
- improve at least 3/4 tasks;
- improve at least 2/3 contact-sensitive tasks;
- degrade action RMSE by at most 20%;
- reduce contact preservation by at most 1 percentage point;
- have normalized utilization >= 0.25;
- have clipping = 0.

If CR-C3 FULL passes but every K=64 version fails, return:

PIVOT_TO_CONSEQUENCE_RETRIEVAL_STEERING

If only contact-onset/post-contact pass, return:

NARROW_TO_CONTACT_CONSEQUENCE_METRIC

Part 10 — Obtain genuinely fresh confirmation evidence

Preferred source order:

1. unused successful demonstrations with IDs >= 50;
2. new successful trajectories generated under a frozen nominal state generator  
and new simulator seeds;
3. fresh perturbed states generated from previously unused timesteps with  
pre-registered bounded physical perturbations.

The third option must be labeled:

FRESH_PERTURBED_STATE_CONFIRMATION

It is not a new-episode claim.

If none is possible, return:

BLOCKED_NO_FRESH_CONFIRMATION

Before any confirmation branch execution:

- freeze exact states, seeds, hashes and perturbations;
- freeze the selected model, ensemble members, atlas algorithm, K, L, metrics  
and thresholds;
- commit FRESH_CONFIRMATION_SPLIT.json;
- use separate sacrificial states for replay validation.

Part 11 — Final confirmation gate

Use 10,000 paired episode/state-clustered bootstrap replicates.

GO_TO_SMALL_BC requires:

- pooled effect gain >= 10%;
- paired 95% CI lower bound > 0;
- at least 3/4 tasks improve;
- at least 2/3 contact-sensitive tasks improve;
- action-RMSE degradation <= 20%;
- contact-preservation drop <= 1 percentage point;
- normalized utilization >= 0.25;
- clipping = 0;
- context-shuffled controls retain <= 25% of the gain;
- all three training seeds have the same improvement direction.

Otherwise return:

CONFIRMATION_FAILED

Return exactly one final disposition:

REJECT_CONSEQUENCE_HEADROOM  
REJECT_LEARNED_CONSEQUENCE_METRIC  
STATIC_EFFECT_METRIC_ONLY  
PIVOT_TO_CONSEQUENCE_RETRIEVAL_STEERING  
NARROW_TO_CONTACT_CONSEQUENCE_METRIC  
BLOCKED_NO_FRESH_CONFIRMATION  
CONFIRMATION_FAILED  
GO_TO_SMALL_BC

Required report

STAGE4_REPORT.md must explicitly answer:

1. How much error comes from bank coverage, learned metric, K=64 compression and  
C4 override?
2. Did the frozen C3 development gain replicate on historical exploratory data?
3. Is true consequence ordering state-dependent?
4. Does the learned model actually use state, nominal action and history?
5. Does independent C3 objective selection improve the result?
6. Does context-reversal training improve candidate ordering?
7. Does the trust region reduce action deviation without erasing effect gain?
8. Does full-bank retrieval work when K=64 fails?
9. Is the remaining method an alphabet, retrieval steering, contact-only method,  
static metric, or rejected hypothesis?
10. Was the result confirmed on genuinely fresh states?
11. Is the mechanism ready for a small state-based BC experiment?

Stop after Stage 4.

Do not automatically start ACT, Diffusion Policy, SmolVLA or pi0.5.  
Do not claim paper readiness or inherit the old N3 novelty label.
