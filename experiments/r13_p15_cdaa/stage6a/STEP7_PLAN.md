# R13-P15 Stage 6-A Agent Prompt — Quantizer Repair and C4-Free Replay

## Repository

Continue work in:

`https://github.com/mikasaTu/R13-P15-Consequence-Adaptive-Action-Alphabets`

Create a new branch:

`r13-p15-stage6a-quantizer-repair-c4-free-replay`

Create the output root:

`experiments/r13_p15_cdaa/stage6a/`

## What this stage is

This stage is a **defect-repair replay**. It is not a new method, not a rescue of any rejected
method, and not a re-interpretation of any frozen result. It re-executes the selection chain over
**already-simulated** candidate consequences with two localized defects removed.

Defect D1 — quantizer degeneracy. Stage 1's M0 assigned 1 of 64 codes at the median state
(`assignment_utilization = 0.015625`), clipped 83.42% of continuous coordinates
(`realized_clipped_coordinate_fraction = 0.834201`), and scored under a metric whose
`contact_and_force` group held 0.999953 of mean squared normalized error. Stage 1 therefore
measured a broken quantizer and a degenerate metric, not the hypothesized mechanism.

Defect D2 — pipeline coverage. In Stage 3, C3 alone reduced realized effect error from B2's 0.30817
to 0.28315, with regret 0.24297 and NDCG@16 0.61958. The frozen C5 used C3 only to choose the
64-code atlas and then let the strictly worse C4 pair ranker decide the final candidate
(regret 0.31872, NDCG@16 0.4449, realized 0.3589 → 0.37497 at K=64). No stage has ever evaluated
the chain with C4 removed.

Stage 6-A answers exactly one question:

> Under the frozen Stage 2 five-group equal-weight metric and common executable bank, with a
> health-gated K=64 alphabet and the C4 pair ranker deleted from the selection code path, what
> fraction of the Stage 5 Gate 0 state-adaptive headroom is recovered on already-executed
> candidates?

Stage 6-A trains nothing, simulates nothing, and submits nothing.

## Historical scientific status — immutable

- Stage 1: `REJECT_CORE_HYPOTHESIS`
- Stage 1.5: `REJECT_P15_FAMILY`
- Stage 2: `ORACLE_ONLY_NO_DEPLOYABLE_MODEL`
- Stage 3: `ORACLE_ONLY_NO_LEARNABLE_RANKER`
- Stage 4: `STATIC_EFFECT_METRIC_ONLY`
- Stage 5: `STATIC_CONSEQUENCE_METRIC_ONLY`

Do not edit, delete, reinterpret, relabel or overwrite any historical artifact. Stage 6-A cannot
upgrade any of these dispositions. A Stage 6-A pass authorizes Stage 6-B and nothing else.

## Hard scope

- Four frozen LIBERO tasks only: `bowl_on_plate`, `plate_push`, `stove_turn_on`, `wine_rack`.
- Panda `OSC_POSE`, 20 Hz, H=4, three settle steps.
- No new simulation branches. No new episodes. No new target residuals.
- No training, fine-tuning, checkpoint re-selection or hyperparameter search of any kind.
- Do not train or modify ACT, Diffusion Policy, SmolVLA, pi0.5, DINO-WM or any VLA.
- No PAI job. No GPU except to load a frozen Stage 3/Stage 5 checkpoint for inference.
- No HTML, no publication/activation/mutation-farm/cryptographic infrastructure.
- Ordinary Git, Markdown, JSON, CSV, Parquet, NPZ and pytest only.
- Every number in the report must be produced by a file under this stage's output root. No number
  may be copied from a historical report without independent recomputation.

## Part 0 — Bind history and locate executed consequences

Record the repository commit and tree. Verify Stage 1, 1.5, 2, 3, 4 and 5 artifacts are
byte-identical to their published values using `scripts/verify_published_artifacts.py`,
`scripts/verify_stage1_5_artifacts.py --full-stage1-hash`, and the Stage 5 release verifier.
Write `HISTORICAL_BINDING.json`. Any mismatch stops with
`BLOCKED_HISTORICAL_BINDING_MISMATCH`.

Locate the executed candidate-consequence tables. Try in this order:

1. The four Stage 5 caches recorded in
   `experiments/r13_p15_cicr_dla/stage5/STAGE5_CACHE_MANIFEST.json`
   (M=128 local bank, 96 targets per state). Verify every `sha256`.
2. The vendored Stage 2 support shards under
   `experiments/r13_p15_ncea/stage2/work/support_shards/**/*.npz`
   (256 shards, 97 executed branch rows per state).

Record the chosen source, its hashes, and the exact state/target/candidate index space in
`DATA_SOURCE_BINDING.json`. Require at least 96 valid candidates at every evaluated state and
complete coverage of the candidate set for every evaluated (state, target) pair.

If neither source validates, or coverage is partial, stop with
`BLOCKED_NO_EXECUTED_CANDIDATE_CACHE`. Do not regenerate anything by simulation and do not
substitute a predicted consequence for an executed one.

## Part 1 — Reproduce the defects before repairing them

Write `DEFECT_REPRODUCTION.json` containing independently recomputed values for:

- D1: median `assignment_utilization`, median realized clipped-coordinate fraction, and the
  per-group share of mean squared normalized error under the Stage 1 metric. Must reproduce
  `0.015625`, `0.834201`, and a `contact_and_force` share of `0.999953` within 1e-4 relative.
- D2: on the frozen Stage 3 development rows, recompute oracle regret, Spearman, NDCG@16 and
  realized effect error for C3, C4, C5 and B2. Must reproduce `0.24297 / 0.31872`,
  `0.61958 / 0.4449`, and `0.30817 / 0.28315 / 0.3589 / 0.37497` within 1e-4 relative.
- D3 (coverage evidence only, no claim): Stage 5 P1's mean modulation norm `5.903557` against its
  theoretical bound `6.123724`, i.e. 96.405% of the parameterization's own ceiling.

If any reproduction falls outside tolerance, stop with `BLOCKED_DEFECT_NOT_REPRODUCED`. A defect
that cannot be reproduced cannot be repaired.

## Part 2 — Freeze the repaired definition before any Stage 6-A metric is computed

Commit all of the following before reading a single performance number.

**Metric.** The frozen Stage 2–5 `BALANCED_TASK_EFFECT`: object pose; TCP-object relative pose;
contact mode and penetration; gripper and articulation; task progress and constraint violation;
equal group weights; train-only robust scales; capped Huber; raw force excluded. Do not refit,
retune or reweight. The Stage 1 force-dominated metric may appear only as a secondary continuity
diagnostic.

**Candidate set.** The frozen executable bank identified in Part 0. Selection returns an executed
bank index. No pseudo-inverse decode, no coordinate clipping, no action synthesis, no target
residual equality.

**Alphabet.** `K=64`, selected from the bank by deterministic ID-stable K-medoids in the frozen C3
bi-encoder embedding, with a committed seed. No farthest-point-then-rerank construction. Do not
inspect K=32, K=96 or K=128 until the Stage 6-A disposition is frozen.

**Selection.** Intra-atlas selection uses C3 bi-encoder distance only. The C4 pair ranker must be
**removed from the code path**, not down-weighted, not blended, not gated. No listwise reranker, no
soft mixture, no C5 or C6 path.

**Checkpoints.** Reuse the frozen Stage 3 C3 checkpoints and the frozen Stage 5 B1/B2 checkpoints
exactly as published. Any retraining voids the stage.

Write `REPAIRED_DEFINITION.json` and `PREREGISTRATION.md` and commit them before Part 3.

## Part 3 — Gate H, quantizer health, evaluated first and enforced hard

Compute the health table before any effect-error comparison is read. Required, on the development
split:

- median normalized assignment utilization `> 0.50`, where utilization is the count of distinct
  codes selected across the 96 targets at a state divided by K;
- median realized clipped-coordinate fraction `< 0.05`;
- pooled dead-code fraction `< 0.10`;
- action reconstruction RMSE at most `1.25 ×` the strongest deployable K=64 baseline;
- at least 96 valid candidates at every evaluated state.

Note that 0.50 is deliberately stricter than the Stage 5 gate value of 0.25. Use 0.50.

If Gate H fails, write `QUANTIZER_HEALTH.json`, halt immediately with `QUANTIZER_STILL_DEGENERATE`,
and do **not** compute or report any effect-error improvement. A degenerate quantizer produces no
mechanism evidence in either direction.

## Part 4 — Comparator ladder

Evaluate every arm on identical states, targets and candidates, by lookup into the executed
consequence table. Recompute, do not import:

- `O_STATE_FULL`, `O_STATE_K64`, `O_STATIC_FULL`, `O_CONTACT_FULL`, `O_PHASE_FULL`.
  These must reproduce the Stage 5 Gate 0 pooled values `0.054725`, `0.069166`, `0.442387`,
  `0.449735`, `0.447839` within 1e-4 relative.
- `B1_ACTION_ONLY` FULL and K=64.
- `B2_STATIC_CONSEQUENCE` FULL and K=64.
- `C3_FULL` — C3 distance over the whole bank, no atlas.
- `C5_FROZEN` — exact reproduction of the frozen Stage 3/5 chain, for the delta attributable to
  deleting C4.
- `R1_REPAIRED_K64` — the proposed object: health-gated K=64 atlas, C3-only selection.
- `A0_ACTUATOR_UNIFORM` — K=64 selected by uniform spacing in actuator space. This is the uniform
  symbol-density comparator the original idea claims to beat.

## Part 5 — Recovery accounting

Write `RECOVERY_ACCOUNTING.json` with:

- `headroom_full = O_STATIC_FULL − O_STATE_FULL` and its relative form (Stage 5 reported 87.6%);
- `headroom_k64 = B2_STATIC_CONSEQUENCE_K64 − O_STATE_K64`;
- `recovered_fraction = (B2_K64 − R1_REPAIRED_K64) / headroom_k64`;
- `c4_removal_delta = C5_FROZEN − R1_REPAIRED_K64`;
- `compression_loss = R1_REPAIRED_K64 − C3_FULL`;
- episode-clustered paired bootstrap, 10,000 replicates, frozen seed, 95% CI on absolute error
  differences for every comparison above;
- per-task (4) and per-phase (4: free_space, pre_contact, contact_onset, post_contact) breakdown,
  with the three contact-sensitive tasks reported separately.

## Part 6 — Mandatory controls

Run each under the identical chain, changing exactly one thing:

- `CTRL_RANDOM_ATLAS` — K=64 drawn uniformly from the bank, 20 seeded draws, report mean and spread;
- `CTRL_SHUFFLED_EMBEDDING` — C3 embedding rows permuted within task;
- `CTRL_LABEL_SHUFFLED` — true-distance rows permuted;
- `CTRL_RAW_ACTION_DISTANCE` — selection by raw residual distance, no learned embedding.

Decision rule: if `CTRL_RANDOM_ATLAS` or `A0_ACTUATOR_UNIFORM` retains at least 75% of R1's gain
over `B2_K64`, the gain is not specific to consequence-adaptive density. Stop with
`GAIN_NOT_DENSITY_SPECIFIC` and report it as such.

## Part 7 — Gate A

All of the following must hold:

1. Gate H passed.
2. `R1_REPAIRED_K64` improves at least 8% over the strongest deployable K=64 baseline, pooled.
3. Episode-clustered paired 95% CI lower bound strictly above zero.
4. At least 3 of 4 tasks improve, and at least 2 of 3 contact-sensitive tasks improve.
5. `recovered_fraction >= 0.20`.
6. `CTRL_RANDOM_ATLAS` and `A0_ACTUATOR_UNIFORM` each retain at most 50% of the gain.
7. `CTRL_LABEL_SHUFFLED` retains at most 25% of the gain.
8. Action RMSE degradation at most 20%; contact preservation drop at most one percentage point.

All pass → `REPAIR_CONFIRMED_ADVANCE_TO_STAGE6B`.

Gate H passed, C4 removed, but conditions 2–5 fail → `C4_REMOVAL_INSUFFICIENT`.

## Exact final dispositions

Return exactly one, in `FINAL_DISPOSITION.json`, with an explicit precedence trace:

- `BLOCKED_HISTORICAL_BINDING_MISMATCH`
- `BLOCKED_NO_EXECUTED_CANDIDATE_CACHE`
- `BLOCKED_DEFECT_NOT_REPRODUCED`
- `QUANTIZER_STILL_DEGENERATE`
- `GAIN_NOT_DENSITY_SPECIFIC`
- `C4_REMOVAL_INSUFFICIENT`
- `REPAIR_CONFIRMED_ADVANCE_TO_STAGE6B`

## Required artifacts

```text
experiments/r13_p15_cdaa/stage6a/
├── PREREGISTRATION.md
├── HISTORICAL_BINDING.json
├── DATA_SOURCE_BINDING.json
├── DEFECT_REPRODUCTION.json
├── REPAIRED_DEFINITION.json
├── ATLAS_K64.json
├── QUANTIZER_HEALTH.json
├── DEVELOPMENT_RANKING.csv
├── DEVELOPMENT_REALIZED.csv
├── DEVELOPMENT_CONTROLS.csv
├── development_realized_rows.parquet
├── RECOVERY_ACCOUNTING.json
├── BOOTSTRAP_RESULTS.json
├── GATE_H.json
├── GATE_A.json
├── FINAL_DISPOSITION.json
├── STAGE6A_RELEASE_VERIFICATION.json
└── STAGE6A_REPORT.md
```

## Required tests

Add pytest coverage for: C4 absence from the selection call graph; deterministic K-medoids;
ID-stable atlas indices; candidate-order permutation invariance; no candidate or target ID in model
inputs; zero clipping; utilization computed over distinct codes; lookup returns an executed
consequence and never a predicted one; historical path immutability; bootstrap cluster identity;
exactly-one-disposition logic. Run the complete existing suite plus the Stage 6-A tests.

## Required report

`STAGE6A_REPORT.md`, in Chinese with English artifact keys, matching the Stage 5 report structure.
It must answer:

1. Were D1 and D2 reproduced exactly? With what numbers?
2. Does the repaired K=64 alphabet pass Gate H, and by how much?
3. What is `c4_removal_delta` — how much did deleting one component change the result?
4. What fraction of the Gate 0 K=64 headroom is recovered?
5. Do the random-atlas and actuator-uniform controls survive? What does that imply about
   consequence-adaptive density specifically?
6. Which tasks and which phases carry the effect?
7. Does anything here justify revisiting the Stage 1 `REJECT_CORE_HYPOTHESIS` verdict, and on what
   restricted grounds?
8. What is the single cheapest experiment that would falsify the Stage 6-A conclusion?

## Stop

After writing `FINAL_DISPOSITION.json` and `STAGE6A_REPORT.md`, stop and wait for review.

Do not begin Stage 6-B. Do not train any model. Do not generate fresh trajectories. Do not submit a
PAI job. Do not claim novelty restoration, paper readiness, VLA improvement or task-success
improvement from replay evidence on already-executed candidates.
