# R13-P15 Stage 6-A preregistration

## Scientific boundary

This is a defect-repair replay over already-executed LIBERO candidate consequences. It does not
train a model, run a simulator, submit PAI work, revise any historical disposition, or begin
Stage 6-B. The sole question is how much Stage 5 Gate 0 K=64 adaptive headroom is recovered after
repairing the quantizer and deleting C4 from the proposed selection call graph.

## Frozen inputs

- Tasks: `bowl_on_plate`, `plate_push`, `stove_turn_on`, `wine_rack`.
- Controller: Panda `OSC_POSE`, 20 Hz, `H=4`, three settle steps.
- Data: the hash-verified Stage 5 M=128 executed-consequence caches, 96 targets per state.
- Metric: frozen `BALANCED_TASK_EFFECT`; equal group weights, train-only robust scales, capped
  Huber, raw force excluded. No refit, retuning, or reweighting.
- Checkpoints: published Stage 3 C3 and Stage 5 B1/B2 bytes only.

## Repaired object

- Alphabet: `K=64` deterministic ID-stable K-medoids in the frozen C3 bi-encoder embedding.
- Selection: C3 squared embedding distance only inside the atlas.
- C4 is absent from the R1 import/call graph; no reranker, blend, gate, C5, C6, or synthesized
  action is permitted.
- The decoded local index is mapped through `candidate_source_index`; physical effects are looked
  up only from executed `true_distance` entries.

## Ordering firewall

1. Verify history and data hashes.
2. Reproduce D1/D2/D3 within `1e-4` relative tolerance.
3. Commit this file and `REPAIRED_DEFINITION.json`.
4. Compute Gate H without reading effect-error values.
5. Only if Gate H passes, compute the comparator ladder, controls, bootstrap, Gate A, final
   disposition, and report.

## Gate H

On development: median distinct-code utilization divided by 64 must be `>0.50`; clipped fraction
must be `<0.05`; pooled dead-code fraction `<0.10`; action RMSE no more than `1.25x` the strongest
deployable K=64 baseline; every state must expose at least 96 valid candidates. Failure freezes
`QUANTIZER_STILL_DEGENERATE` and forbids any effect-error comparison.

## Comparator ladder and controls

Recompute O_STATE_FULL/K64, O_STATIC_FULL, O_CONTACT_FULL, O_PHASE_FULL, B1/B2 FULL/K64, C3_FULL,
C5_FROZEN, R1_REPAIRED_K64, and A0_ACTUATOR_UNIFORM on identical rows. Run 20 random atlases,
task-wise shuffled C3 embedding, label shuffle, and raw-action distance. Use 10,000 paired
source-episode-clustered bootstrap replicates with seed `13150603`.

## Disposition

Apply the prompt's precedence exactly and emit one of the seven registered dispositions. A pass
only authorizes Stage 6-B review; it does not alter Stage 1-5 conclusions or establish policy/VLA
task-success evidence.
