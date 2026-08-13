# R13-P15 Stage 2 Preregistration

## Status and non-retroactivity

This document freezes the Fresh-Support Nonlinear Consequence Atlas Audit before any Stage 2 method result is computed or inspected. Stage 1 remains `REJECT_CORE_HYPOTHESIS`; Stage 1.5 remains `REJECT_P15_FAMILY`. Their artifacts, labels, and interpretations are immutable. Stage 2 tests a broader consequence-aware action-alphabet hypothesis and cannot retroactively rescue CAAA-v2 or P15.

The only permitted final dispositions are:

- `REJECT_BROAD_CONSEQUENCE_HYPOTHESIS`
- `ORACLE_ONLY_NO_DEPLOYABLE_MODEL`
- `REJECT_NONLINEAR_CONSEQUENCE_ALPHABET`
- `NARROW_TO_CONTACT_MODE_ALPHABET`
- `GO_TO_SMALL_BC`

The experiment stops after Stage 2. No behavior cloning or policy training will be started.

## Frozen scope

- Benchmark: the existing deterministic LIBERO snapshot/restore implementation.
- Tasks: `bowl_on_plate`, `plate_push`, `stove_turn_on`, and `wine_rack`.
- Robot/controller: Panda, `OSC_POSE`, 20 Hz, 7-D actions.
- Chunk: `H=4`; perturb only the first six continuous coordinates at each step; copy the nominal demonstration gripper command unchanged.
- Settling: three simulator steps.
- Simulation: CPU only. Predictor training may use at most one GPU. No PAI job, multi-GPU job, policy, HTML report, activation system, publication framework, cryptographic audit framework, or mutation framework.
- Storage: Git, Markdown, JSON, CSV, Parquet, NPZ/Zarr, and ordinary pytest.
- Primary alphabet size: `K=64`. `K={32,128}` is locked until the primary disposition is frozen.

## Input binding

`INPUT_BINDING.json` will bind the repository input commit/tree, the exact Stage 1 and Stage 1.5 commits and dispositions, every required historical artifact hash, LIBERO source-tree hash, environment lock, and simulator/controller settings. The finalizer must fail before result computation if the historical paths differ from input commit `154d4a89e071d94208f5302955c55c13e3cff7f3`, the Stage 1 tree hash changes, or the LIBERO source tree changes.

## Fresh episodes and locked splits

Historical IDs 0–15 are excluded from all Stage 2 fitting, calibration, development, and confirmation analyses. The preferred per-task split is:

| Split | Episode IDs | Purpose |
|---|---:|---|
| train | 16–23 | fitting and robust consequence scaling |
| calibration | 24–27 | architecture, early stopping, thresholds, baseline selection |
| development | 28–31 | internal success gates |
| confirmation | 32–39 | untouched confirmation, conditionally unlocked |

Before method computation, all demonstrations must be verified successful and hashed, and four phase snapshots per episode must be frozen: `free_space`, `pre_contact`, `contact_onset`, and `post_contact`. An invalid preferred episode is replaced only by the smallest ascending unused successful ID, before results are visible. Fewer than 24 usable fresh episodes for any task yields `BLOCKED_INSUFFICIENT_FRESH_DEMOS`.

Every frozen snapshot is tested by replaying A twice and by both A→B→A and B→A→B execution orders at tolerance `1e-12`. Any failure yields `BLOCKED_NONDETERMINISTIC_BRANCHING`; every failure is retained in the report.

## Split-specific unseen supports

For each snapshot, derive an independent seed as SHA-256 of the global seed `13150200`, task, episode, phase, and split. Generate exactly 24 unit-norm 24-D directions:

- 12 smooth temporal directions from random DCT combinations;
- 6 suffix-localized contact directions;
- 6 random rank-two temporal-action directions.

For each direction, deterministically sample two radii from `[0.04, 0.12]` and execute antithetic signs `{-1,+1}`. Each target support therefore has 96 perturbed branches plus one nominal branch. Every branch begins from the identical restored snapshot. Directions whose maximum coordinate exceeds `0.82` are deterministically rejected and redrawn so valid frozen snapshots remain in bounds.

Hard pre-result checks require zero exact direction overlap and zero exact residual hash overlap between any two splits. The maximum absolute cross-split cosine similarity is reported. Development/confirmation target residuals may not exactly match a common-bank member.

## Common executable action bank

Construct one train-only residual bank with `M=256`. Candidate residuals are balanced over the 4 tasks × 4 phases × 3 direction families, deduplicated by exact typed-array hash, and selected within each stratum by deterministic farthest-point sampling. Only residuals valid when added to every frozen train nominal chunk enter the pool.

All methods receive this same bank. For each evaluated state, invalid candidates are removed without clipping. At least 128 candidates must remain before selecting `K=64`; otherwise the experiment stops as an input-validity failure. No decoder may pseudoinvert a metric, generate an off-bank action, or clip an invalid action.

## Frozen consequence metrics

The primary `BALANCED_TASK_EFFECT` compares target and decoded settled consequences using five equally weighted active groups:

1. object pose;
2. TCP–object relative pose;
3. contact mode and penetration;
4. gripper and articulation;
5. task progress and constraint violation.

Continuous dimensions are scaled using train-only `max(1.4826×MAD, IQR/1.349, physical-unit floor)`. The normalized difference uses Huber loss with delta `1.5`, capped at absolute normalized value `4.0`. Dimensions are averaged within each group and active groups are averaged equally. Contact-mode mismatch is a 0/1 term averaged with penetration in group 3. Raw contact force is excluded from the primary metric.

Secondary metrics are `CONTACT_FORCE_EFFECT`, the frozen Stage 1 metric for continuity only, contact-mode preservation, task-progress preservation, and continuous-action reconstruction RMSE. Metric weights and scale floors in `consequence_metrics.json` cannot be tuned after results become visible.

## Frozen methods

- `B0`: unquantized continuous target; upper bound only.
- `B1`: one centered-covariance residual alphabet selected from the common bank.
- `B2`: phase-conditioned residual alphabets selected from the common bank.
- `B3`: per-state action-space farthest-point medoids over the valid bank.
- `B4`: state-conditioned action-autoencoder latent, trained only for action reconstruction and never with consequences.
- `LJ`/`O2`: train-only local ridge Jacobians, interpolated from nearest frozen train states using calibration-selected ridge and neighbor count; bank predictions only, with no pseudoinversion.
- `O1`: diagnostic true-effect oracle; it uses true target/candidate simulator effects to select and assign the local atlas.
- `NCEA`: five-member global nonlinear state–residual consequence ensemble.
- `MC-NCEA`: shared architecture with phase-conditioned heads.
- `UG-NCEA`: calibration-frozen uncertainty gate at fixed 50%, 70%, and 90% consequence-quantization coverage, falling back to the strongest frozen deployable baseline.
- `P3`: mode/head labels shuffled within task strata.
- `P4`: state features shuffled within task/phase.
- `P5`: continuous effects and contact labels jointly shuffled within task/phase.
- `P6`: frozen random nonlinear features with ridge readout, matched in width and ensemble count.

Atlas selection is deterministic farthest-point medoid selection in the relevant predicted/true effect space, followed by nearest-effect assignment of the target. Ties use ascending common-bank index.

## Predictor protocol

Each predictor input contains only the current measured state vector and mask, current task/phase/contact indicators, and the 24-D residual chunk. It contains no future feature, target consequence, episode outcome, or target-bank identity. Outputs are the 34 preregistered continuous primary consequence dimensions plus contact-mode logits. Targets are settled consequence changes relative to the nominal restored-snapshot branch.

Train episodes alone fit parameters and scaling. Calibration episodes alone choose between hidden widths `(128,128)` and `(256,256)`, select early stopping (maximum 160 epochs, patience 20, minimum delta `1e-5`), and freeze uncertainty/contact thresholds. Other fixed settings are ensemble size 5, batch size 512, Adam learning rate `1e-3`, weight decay `1e-5`, and contact-loss weight `0.25`. P6 uses ridge `1e-3`. Random seeds and all selected checkpoints are recorded.

Report pooled, per-task, and per-phase NRMSE, balanced prediction error, contact-transition accuracy, uncertainty/error Spearman correlation, coverage calibration at 50/70/90%, and the fraction of the O1–O2 gap closed. No development result can choose a model or threshold.

## Calibration-only choices

Calibration chooses the strongest deployable baseline by pooled `BALANCED_TASK_EFFECT`, the nonlinear architecture/checkpoint, local-J ridge in `{1e-6,1e-4,1e-2,1e-1}`, local-J state-neighbor count in `{1,3,5,9}`, and contact confidence in `{0.5,0.6,0.7,0.8,0.9}`. UG thresholds are calibration empirical quantiles yielding fixed 50/70/90% coverage. These choices are serialized before development metrics are read.

## Development gates

Development uses only IDs 28–31.

Gate A passes only if O1 lowers pooled primary error by at least 10% versus the strongest of B1/B2/B3 and lowers error on at least three tasks. Failure immediately freezes `REJECT_BROAD_CONSEQUENCE_HYPOTHESIS`; confirmation remains unread.

Gate B passes only if the best nonlinear model lowers pooled prediction error by at least 20% versus O2/LJ, improves at least two of the three contact-sensitive tasks, closes at least 50% of the positive O1–O2 realized-error gap, and beats P3/P4/P5/P6. Failure freezes `ORACLE_ONLY_NO_DEPLOYABLE_MODEL`.

Gate C passes only if NCEA or MC-NCEA lowers realized primary error by at least 8% versus the frozen strongest deployable baseline, improves at least three tasks and at least two contact-sensitive tasks, degrades `bowl_on_plate` by no more than 5%, clips below 1%, has normalized utilization/perplexity above 0.25, is not reproduced by a shuffled/random control, and degrades action reconstruction by less than 10%. Failure freezes `REJECT_NONLINEAR_CONSEQUENCE_ALPHABET`.

Relative gain is `(baseline_error − method_error) / baseline_error`. “Improves” means strictly lower error. Gap closed is `(O2_error − method_error)/(O2_error − O1_error)` when the denominator is positive; otherwise it is zero. Control gain retention is `max(0, baseline_error − control_error) / max(baseline_error − method_error, 1e-12)`.

## Conditional confirmation and bootstrap

Episodes 32–39 cannot be loaded for support or candidate-effect execution until a serialized development gate explicitly unlocks them. If development stops, required confirmation artifacts are emitted with zero observations and a machine-readable `NOT_RUN_DEVELOPMENT_GATE_FAILED` status.

If unlocked, all choices remain frozen. Confirmation uses 10,000 paired episode-cluster bootstrap replicates, sampling episodes with replacement within task and retaining all phases/targets for each sampled episode.

`GO_TO_SMALL_BC` requires all 12 confirmation conditions in the supplied Stage 2 protocol: ≥10% pooled gain, positive 95% CI lower bound, ≥3 tasks improved, ≥2 contact-sensitive tasks improved, bowl degradation ≤5%, control retention ≤25%, action reconstruction degradation ≤10%, coverage ≥70%, clipping <1%, utilization ≥25%, calibrated uncertainty, and robustness to the balanced metric rather than the Stage 1 force-dominated metric.

If the pooled confirmation gate fails, `NARROW_TO_CONTACT_MODE_ALPHABET` requires ≥10% contact-onset/post-contact gain with a supporting paired 95% CI, free-space degradation ≤5%, and mode/state-shuffled control retention ≤25%. Otherwise the applicable development-stage rejection remains final.

## Preregistered mechanism localization (not idea generation)

The report will reverse-localize every material improvement or degradation using only these frozen diagnostics:

- decompose realized error by task, phase, direction family, radius, contact transition, and each of the five balanced-effect groups;
- compare oracle O1 with O2 to isolate nonlinear representational headroom;
- compare NCEA with MC-NCEA and P3 to isolate phase/contact routing;
- compare NCEA with P4 to test current-state dependence;
- compare NCEA with P5/P6 to test learned consequence signal versus capacity/random geometry;
- compare NCEA/MC-NCEA with B3/B4 to distinguish consequence organization from action geometry/action-only representation;
- relate ensemble disagreement, nearest-atlas distance, valid-bank size, code utilization, and action reconstruction to realized error;
- report antithetic asymmetry and radius scaling as local nonlinearity diagnostics.

An explanation is called supported only when its preregistered control changes the relevant gain in the predicted direction and the task/phase decomposition agrees. Otherwise the mechanism is reported as unresolved. These analyses explain observed code mechanisms and do not generate or promote a new research idea.

## Required artifacts and stopping rule

The output root is `experiments/r13_p15_ncea/stage2/`. All required files from the supplied protocol will be emitted, including explicit empty/locked confirmation artifacts when confirmation is not authorized. `STAGE2_REPORT.md` will separate historical rejected evidence, development evidence, untouched confirmation evidence, oracle-only results, deployable results, predictor results, mechanism controls, metric sensitivity, negative runs, overlap checks, and the exact single final disposition.
