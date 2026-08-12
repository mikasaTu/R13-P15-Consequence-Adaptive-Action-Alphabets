# R13-P15 CAAA-v2 — LIBERO Stage 1

This repository contains the complete Stage 1 mechanism audit for
**Consequence-Riemannian Action Alphabets (CAAA-v2)** on LIBERO. It includes
the implementation, tests, frozen preregistration, deterministic replay data,
all intermediate shards, fitted Jacobians and codebooks, realized simulation
results, bootstrap output, PAI launch provenance, and the final report.

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
  stage1_local_smoke/                development-machine smoke evidence
provenance/
  tests/                             machine-readable pytest results
  pai/                               submitted/read-back PAI records and sentinels
```

The formal code snapshot is commit
`34995e8e7c3069b22785ad04536f0d429e75c0fc` (tree
`ad6fa59b782f63624ee3ccef8e880a2398669ce8`). The publication commit adds
artifacts and documentation without rewriting the frozen formal outputs.

## Verify the published package

```bash
python scripts/verify_published_artifacts.py
python -m pytest -q
```

The first command is standard-library only. It checks the report-declared
artifact hashes, every atomic NPZ completion marker, formal replay counts,
result-table row counts, PAI completion evidence, and the final disposition.
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
