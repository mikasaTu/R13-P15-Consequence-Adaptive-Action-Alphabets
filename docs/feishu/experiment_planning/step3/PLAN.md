# step3

Task: Execute R13-P15 Stage 2:  
Fresh-Support Nonlinear Consequence Atlas Audit.

Repository:  
[https://github.com/mikasaTu/R13-P15-Consequence-Adaptive-Action-Alphabets](https://github.com/mikasaTu/R13-P15-Consequence-Adaptive-Action-Alphabets)

Create a new branch:

r13-p15-stage2-nonlinear-consequence-atlas

Scientific status:

- Stage 1 remains REJECT_CORE_HYPOTHESIS.
- Stage 1.5 remains REJECT_P15_FAMILY.
- Do not modify, overwrite, delete, reinterpret, or relabel any Stage 1 or  
Stage 1.5 artifact.
- This is a new preregistered experiment testing the broader consequence-aware  
action-alphabet hypothesis and new nonlinear method hypotheses.
- It is not a continuation that can retroactively rescue CAAA-v2.

Primary scientific question:

On fresh episodes and genuinely unseen action perturbations, does organizing a  
state-dependent action alphabet by predicted nonlinear physical consequences  
preserve realized task effects better than organizing it by action-space  
distance, covariance, phase, or action-only representation learning?

New method hypotheses:

H0 — Fresh-support consequence equivalence:  
True physical consequence equivalence should outperform action-space similarity  
on action directions never used for fitting.

H1 — Nonlinear Consequence-Equivalence Alphabet, NCEA:  
A nonlinear state-action consequence model can dynamically choose executable  
action codewords that cover predicted consequence space without Jacobian  
pseudoinversion.

H2 — Mode-Conditioned NCEA, MC-NCEA:  
Separate contact-mode experts should outperform one global consequence model in  
pre-contact, contact-onset and post-contact states.

H3 — Uncertainty-Gated NCEA, UG-NCEA:  
At a fixed quantization coverage, predictor uncertainty should identify states  
where consequence-based quantization must fall back to a matched baseline.

Hard scope:

- Use the existing four frozen LIBERO tasks:  
bowl_on_plate  
plate_push  
stove_turn_on  
wine_rack
- Use the existing deterministic LIBERO snapshot/restore implementation.
- Do not train ACT, Diffusion Policy, SmolVLA, pi0.5, DINO-WM or any large policy.
- A small state-action consequence predictor is allowed.
- Use CPU simulation and at most one GPU for predictor training.
- Do not submit a two-GPU job.
- Do not generate HTML.
- Do not build formal activation, custom publication systems, cryptographic  
provenance frameworks, mutation frameworks, or PAI orchestration machinery.
- Use normal Git, Markdown, JSON, CSV, Parquet, NPZ/Zarr and ordinary pytest.
- Stop after Stage 2.
- Do not automatically start behavior cloning.

Output root:

experiments/r13_p15_ncea/stage2/  
├── PREREGISTRATION.md  
├── INPUT_BINDING.json  
├── fresh_episode_inventory.json  
├── development_split.json  
├── confirmation_split.json  
├── perturbation_banks.npz  
├── action_bank.npz  
├── consequence_metrics.json  
├── development_rollouts.zarr  
├── predictor_metrics.csv  
├── development_quantization.csv  
├── development_controls.csv  
├── confirmation_rollouts.zarr  
├── confirmation_quantization.csv  
├── bootstrap_results.json  
└── STAGE2_REPORT.md

Commit PREREGISTRATION.md, INPUT_BINDING.json, episode IDs, all direction-bank  
seeds, metric definitions, method definitions and success gates before  
computing or inspecting any method result.

Part 1 — Bind old evidence without changing it

Record:

- current repository commit/tree;
- Stage 1 and Stage 1.5 commits and final dispositions;
- LIBERO commit;
- environment lock;
- simulator/controller settings;
- all Stage 1/1.5 artifact hashes.

Verify that all old files remain byte-identical.

Part 2 — Freeze genuinely fresh episodes

Historical IDs 0–15 may not be used to fit, calibrate or confirm a Stage 2  
method.

Preferred split per task:

train:  
episode IDs 16–23

calibration:  
episode IDs 24–27

internal development test:  
episode IDs 28–31

untouched confirmation:  
episode IDs 32–39

Before reading any result:

- verify every selected demo is successful;
- if an ID is invalid, choose the smallest unused successful episode ID by a  
deterministic ascending-ID rule;
- freeze exact IDs, hashes and phase snapshot indices;
- never replace an episode after seeing method results.

If fewer than 24 unused successful episodes exist for any task, stop with:

BLOCKED_INSUFFICIENT_FRESH_DEMOS

Part 3 — Generate split-specific unseen action supports

For every snapshot generate 24 directions:

- 12 smooth temporal directions generated from random DCT combinations;
- 6 suffix-localized contact directions;
- 6 random low-rank temporal-action directions.

For every direction:

- generate antithetic + and - branches;
- use two radii sampled deterministically from [0.04, 0.12];
- derive seeds from task, episode, phase and split;
- use independent seeds for train, calibration, development test and  
confirmation.

Hard checks:

- no exact residual-action hash may occur in two splits;
- no direction may occur in two splits;
- report the maximum cross-split absolute cosine similarity;
- exclude any target residual that exactly equals an action-bank member;
- every branch must restore the identical simulator snapshot before execution.

Use H=4 and preserve the original gripper command.

Part 4 — Freeze two consequence metrics

Primary metric:

BALANCED_TASK_EFFECT

Use five equal-weight groups:

1. object pose;
2. TCP-object relative pose;
3. contact mode and penetration;
4. gripper and articulation;
5. task progress and constraint violation.

Within each group:

- use train-only robust scaling;
- apply a scale floor to prevent near-zero MAD explosion;
- use a capped or Huber error;
- average dimensions within group;
- average active groups equally.

Raw contact force must not dominate this primary metric.

Secondary metrics:

- CONTACT_FORCE_EFFECT;
- the frozen Stage 1 consequence metric for continuity only;
- contact-mode preservation;
- task-progress preservation;
- action reconstruction error.

Do not tune metric weights after results are visible.

Part 5 — Build a common executable residual-action bank

Build a single train-only bank of M=256 executable residual action chunks.

Requirements:

- balance samples across task, phase and direction family;
- deduplicate near-identical residuals;
- retain only actions valid when added to the current nominal chunk;
- every compared alphabet method receives the same candidate bank;
- report per-state valid-bank size;
- require at least 128 valid candidates before selecting K=64.

Primary alphabet size:

K=64

Do not inspect K=32 or K=128 until the primary Stage 2 disposition is frozen.

Part 6 — Oracle and baseline methods

B0:  
continuous target action, reported only as an unquantized upper bound.

B1:  
centered covariance residual k-means, K=64.

B2:  
phase-conditioned residual k-means, K=64.

B3:  
dynamic action-space medoids:  
select K=64 candidates from the same bank using action-space distance.

B4:  
capacity-matched state-conditioned action VQ:  
learns only action reconstruction and does not use consequence labels.

LJ:  
linear-J consequence alphabet:  
use the frozen local-linear mechanism on the fresh data, but never pseudoinvert  
or generate actions outside the common bank.

O1:  
true-effect oracle consequence atlas:  
execute the common candidate bank from the current snapshot, use true simulator  
effects for both the target and candidate actions, select K candidates and the  
nearest codeword in true consequence space.

O1 is diagnostic only.

O2:  
linear-J oracle atlas:  
use the same candidate bank and selection operator as O1, but replace true  
effects with local linear-J predictions.

Part 7 — Train nonlinear consequence predictors

Input:

- current simulator state features only;
- current contact/phase features;
- residual action chunk delta_a.

Do not use any future or target consequence as an input.

Train:

P1:  
single nonlinear predictor, five-member MLP ensemble.

P2:  
mode-conditioned predictor with a shared trunk and separate  
free-space/pre-contact/contact-onset/post-contact heads.

P3:  
mode-shuffled control with identical architecture and parameters.

P4:  
state-shuffled control.

P5:  
effect-label-shuffled control.

P6:  
capacity-matched random-latent predictor.

Use train episodes only for fitting and calibration episodes only for:

- architecture choice;
- early stopping;
- uncertainty threshold;
- contact-mode gating threshold;
- medoid/FPS settings.

Primary predictor outputs:

- balanced continuous consequence groups;
- contact-mode logits;
- ensemble uncertainty.

Report:

- NRMSE;
- balanced effect error;
- contact transition accuracy;
- uncertainty calibration;
- per-task and per-phase error;
- fraction of the O1–O2 gap closed by each nonlinear model.

Part 8 — Construct the nonlinear consequence alphabets

NCEA:

For each state and every valid residual action bank member, predict consequence  
mean and uncertainty.

Select K=64 bank actions whose predicted consequences cover the local predicted  
consequence space using deterministic k-medoids or farthest-point selection.

For a target residual action, predict its consequence and execute the selected  
bank action with nearest predicted consequence.

Never use Jacobian pseudoinversion.  
Never generate an action outside the common bank.  
Never clip an invalid decoded action into validity.

MC-NCEA:

Use the mode-conditioned predictor and mode-specific consequence comparison.

UG-NCEA:

Filter or fall back when:

- target uncertainty exceeds a calibration-only threshold;
- candidate uncertainty exceeds threshold;
- nearest consequence-code distance exceeds threshold;
- local atlas coverage is insufficient.

Evaluate UG-NCEA at fixed quantization coverage levels:

50%, 70%, 90%.

Compare against:

- random fallback at the same coverage;
- action-space-distance fallback at the same coverage;
- covariance-uncertainty fallback at the same coverage.

Do not claim an uncertainty-gated gain at coverage below 50%.

Part 9 — Development internal screen

Use only episodes 28–31.

Gate A — broad oracle value:

O1 must reduce BALANCED_TASK_EFFECT error by at least 10% relative to the  
strongest of B1, B2 and B3, with lower error on at least three of four tasks.

If Gate A fails:

REJECT_BROAD_CONSEQUENCE_HYPOTHESIS

and stop.

Gate B — nonlinear model value:

The best nonlinear predictor must:

- reduce pooled prediction error by at least 20% relative to O2/linear-J;
- reduce error in at least two of the three contact-sensitive tasks;
- close at least 50% of the O1–O2 oracle gap;
- outperform state-, mode- and effect-shuffled controls.

If Gate B fails:

ORACLE_ONLY_NO_DEPLOYABLE_MODEL

and stop.

Gate C — deployable alphabet value:

At least one of NCEA or MC-NCEA must:

- reduce realized BALANCED_TASK_EFFECT error by at least 8% relative to the  
strongest matched deployable baseline;
- improve at least three of four tasks;
- improve at least two contact-sensitive tasks;
- not degrade bowl_on_plate by more than 5%;
- have clipping rate below 1%;
- have normalized code utilization/perplexity above 0.25;
- not be reproduced by shuffled or random controls;
- have action reconstruction degradation below 10%.

If no method passes, stop without reading confirmation episodes and return:

REJECT_NONLINEAR_CONSEQUENCE_ALPHABET

Part 10 — Untouched confirmation

Only if the development screen passes may episodes 32–39 be loaded.

Do not retune:

- predictor;
- codebook;
- action bank;
- metrics;
- uncertainty threshold;
- mode gate;
- K;
- fallback coverage.

Run all primary methods and controls on confirmation episodes.

Use 10,000 paired episode-cluster bootstrap replicates within task.

Final dispositions

Return exactly one:

REJECT_BROAD_CONSEQUENCE_HYPOTHESIS

ORACLE_ONLY_NO_DEPLOYABLE_MODEL

REJECT_NONLINEAR_CONSEQUENCE_ALPHABET

NARROW_TO_CONTACT_MODE_ALPHABET

GO_TO_SMALL_BC

NARROW_TO_CONTACT_MODE_ALPHABET requires:

- pooled full-task gate fails;
- contact-onset and post-contact improve by at least 10%;
- paired 95% confidence interval supports the contact-phase gain;
- free-space degradation is at most 5%;
- mode-shuffled and state-shuffled controls retain at most 25% of the gain.

GO_TO_SMALL_BC requires all:

1. confirmation BALANCED_TASK_EFFECT improves by at least 10% over the strongest  
matched baseline;
2. paired 95% confidence interval lower bound is above zero;
3. at least three of four tasks improve;
4. at least two contact-sensitive tasks improve;
5. bowl_on_plate degradation is at most 5%;
6. shuffled/random controls retain at most 25% of the gain;
7. action reconstruction degradation is at most 10%;
8. quantization coverage is at least 70%;
9. clipping is below 1%;
10. effective code utilization is at least 25%;
11. predictor uncertainty is calibrated;
12. results do not depend on the frozen Stage 1 force-dominated metric.

Required report

STAGE2_REPORT.md must clearly separate:

- historical rejected evidence;
- development evidence;
- untouched confirmation evidence;
- oracle-only results;
- deployable results;
- predictor results;
- mechanism controls;
- metric sensitivity;
- all negative runs;
- action-support overlap checks;
- exact final disposition.

The report must explicitly answer:

1. Does true consequence equivalence generalize to unseen action directions?
2. Is nonlinear prediction materially better than linear Jacobians?
3. Is contact-mode conditioning necessary?
4. Does consequence-aware action selection outperform action-only VQ?
5. Do shuffled and random controls destroy the gain?
6. Is any gain preserved under a balanced task-effect metric?
7. Is the method ready for a small BC policy experiment?

Stop after Stage 2.

Do not begin ACT, Diffusion Policy, SmolVLA or pi0.5 automatically.
