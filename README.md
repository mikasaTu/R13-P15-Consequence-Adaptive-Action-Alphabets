# R13-P15 consequence-adaptive action alphabets — LIBERO Stages 1–3

This repository contains the complete staged audit of consequence-adaptive
action alphabets on LIBERO: CAAA-v2 (Stage 1), failure localization (Stage
1.5), the nonlinear consequence atlas (Stage 2), and nominal-conditioned
effect ranking (Stage 3). It includes implementations, frozen protocols,
deterministic replay evidence, learned models, row-level realized simulation
results, controls, bootstrap output, and reports.

## Result

**`REJECT_CORE_HYPOTHESIS`**

The calibration-selected comparator was covariance Mahalanobis. On the held-out
test episodes, CAAA-v2's pooled relative improvement in realized settled-effect
error was **-0.39623**, with episode-clustered 95% CI **[-1.6841, -0.055]**.
The preregistered gate therefore rejects the core hypothesis and does not
authorize policy training. No ACT, Diffusion Policy, SmolVLA, π0.5, DINO-WM,
behavior cloning, or other policy training was launched.

The full scientific account is in
[`STAGE1_REPORT.md`](experiments/r13_p15_caaa_v2/stage1/STAGE1_REPORT.md).

## Stage 1.5 failure-localization result

**`REJECT_P15_FAMILY`**

Stage 1.5 preserved Stage 1 byte-for-byte and tested centered residual,
reachability-constrained effect, phase-conditioned, permuted-J and random-SPD
constructions at K=64. It executed 18,432 revised old-test branches from 64
identical restored snapshots and used 10,000 paired episode-cluster bootstrap
replicates. No revised deployable method passed the internal screen: the
permuted-J and random-SPD controls reproduced too much or all of the nominal
gain. The preregistered stopping rule therefore prohibited fresh episodes
16–23; no policy training was launched.

The complete account is in
[`STAGE1_5_REPORT.md`](experiments/r13_p15_caaa_v2/stage1_5/STAGE1_5_REPORT.md).

## Stage 2 fresh-support result

**`ORACLE_ONLY_NO_DEPLOYABLE_MODEL`**

The true-effect oracle generalized strongly to fresh episodes and directions:
balanced realized-effect error fell from 0.332598 to 0.133410 (59.89%; 4/4
tasks). The learned NCEA predictor nevertheless lost 14.98% prediction MSE
against linear-J, improved 0/3 contact-sensitive tasks, and closed -0.72% of
the oracle gap. Gate B failed, so confirmation and policy training stayed
locked. See
[`STAGE2_REPORT.md`](experiments/r13_p15_ncea/stage2/STAGE2_REPORT.md).

## Stage 3 nominal-conditioned ranking result

**`ORACLE_ONLY_NO_LEARNABLE_RANKER`**

All planned experiments were executed even after a gate failed. On development
episodes, the K=64 true-effect oracle improved over B2 by 55.32% (4/4 tasks),
but the selected C4 pair ranker worsened oracle regret by 31.18% relative to
C3 and reduced NDCG@16 by 0.17468. C5 then worsened realized effect error by
21.68% versus B2. The required episodes 40–49 run was completed as
`FORCED_EXPLORATORY_HOLDOUT`: C5 was 22.90% worse than B2 and the pooled paired
95% CI for `B2 error - C5 error` was [-0.07588, -0.04945]. K=32/128 did not
rescue the result. No policy or VLA was trained and no PAI job was submitted.

The scientific report and code-to-result mechanism localization are in
[`STAGE3_REPORT.md`](experiments/r13_p15_ncer_aa/stage3/STAGE3_REPORT.md) and
[`MECHANISM_REVERSE_AUDIT.md`](experiments/r13_p15_ncer_aa/stage3/MECHANISM_REVERSE_AUDIT.md).

## Frozen scope

The authorized benchmark adaptation uses standard `libero_goal` at LIBERO
commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`, not LIBERO-Plus. Every branch
uses Panda `OSC_POSE`, 20 Hz, and `H=4`. The 24 quantized coordinates are the
six delta-pose channels over four control steps; the demonstration gripper
command is copied unchanged.

The four tasks were selected to span distinct physical regimes:

| Task ID | LIBERO task | Mechanism role |
| --- | --- | --- |
| `bowl_on_plate` | put the bowl on the plate | low-constraint pick/place control |
| `plate_push` | push the plate to the front of the stove | sustained sliding contact |
| `stove_turn_on` | turn on the stove | small articulated contact |
| `wine_rack` | put the wine bottle on the rack | precision oriented receptacle |

Sixteen official successful demonstrations per task were frozen and split by
episode into 8 train / 4 calibration / 4 test episodes, with four physical
phase snapshots per episode. Official demonstration HDF5 files are inputs, not
redistributable results, so they are not vendored; their individual SHA-256
values and source paths are frozen in `environment_lock.json`.

## Repository map

```text
caaa_libero/                         implementation
config/                              frozen task/runtime configuration
tests/                               unit and launcher-contract tests
scripts/                             local, PAI, and release verification tools
pai/                                 exact Stage 1 PAI launcher and job template
experiments/r13_p15_caaa_v2/
  stage1/                            complete formal outputs and intermediates
  stage1_5/                          failure-localization outputs and intermediates
  stage1_local_smoke/                development-machine smoke evidence
experiments/r13_p15_ncea/stage2/     fresh-support nonlinear-atlas audit
experiments/r13_p15_ncer_aa/stage3/  nominal-conditioned ranking audit
provenance/
  tests/                             machine-readable pytest results
  pai/                               submitted/read-back PAI records and sentinels
  stage1_5_release_verification.json complete Stage 1.5 validation record
```

The formal code snapshot is commit
`34995e8e7c3069b22785ad04536f0d429e75c0fc` (tree
`ad6fa59b782f63624ee3ccef8e880a2398669ce8`). The publication commit adds
artifacts and documentation without rewriting the frozen formal outputs.

## Verify the published package

```bash
python scripts/verify_published_artifacts.py
python scripts/verify_stage1_5_artifacts.py --full-stage1-hash
python -m pytest -q
python -m caaa_libero.cli stage3-finalize
```

The Stage 1 command is standard-library only. It checks the report-declared
artifact hashes, every atomic NPZ completion marker, formal replay counts,
result-table row counts, PAI completion evidence, and the final disposition.
The Stage 1.5 verifier additionally needs NumPy and pandas/pyarrow for NPZ and
Parquet validation. It checks strict JSON, preregistration order, frozen Stage
1 identity, all old-test plans/results, bootstrap and stopping manifests.
See [`TESTING.md`](TESTING.md) for the exact recorded checks.

## Reproduce simulation collection

Create the frozen environment described in
[`environment_lock.json`](experiments/r13_p15_caaa_v2/stage1/environment_lock.json),
install the standard LIBERO source at the locked commit, and provide the
official demonstrations at the paths recorded there. A local smoke can then be
run with:

```bash
MUJOCO_GL=glx \
PYTHONPATH=/path/to/LIBERO \
python -m caaa_libero.cli smoke --output-root /tmp/caaa_smoke
```

The exact formal launcher is
[`scripts/run_stage1_pai.sh`](scripts/run_stage1_pai.sh). It uses durable,
atomically completed snapshot shards and resumes only when the payload hash and
completion marker validate. The infrastructure wrapper and submitted template
are retained under [`pai/`](pai/).
