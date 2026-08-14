# Validation record

This record distinguishes code checks, simulator replay validation, artifact
integrity, and the formal PAI outcome.

## Code tests

On 2026-08-13 UTC, the eleven-test suite passed in both relevant
interpreters:

| Environment | Command | Result |
| --- | --- | --- |
| frozen LIBERO Python 3.8.13 | `/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero-original/bin/python -m pytest -q` | 11 passed in 1.31 s |
| release-check Python 3.10.19 | `python -m pytest -q` | 11 passed in 0.83 s |

The clean system Python initially had no `pytest`; after installing it, test
collection exposed missing `numpy` and then `h5py`. Those test dependencies
were installed and the suite passed. These were environment setup failures,
not assertion failures. JUnit XML from the passing runs is retained in
`provenance/tests/`.

The suite covers deterministic perturbation directions and ridge math,
coordinate transforms, calibration-only baseline selection, strict resumed
quantized-plan identity, constrained box/ball decoding, within-stratum
Jacobian permutation, strict-JSON screen output, report logic, and the PAI
launcher contract.

## Formal simulator replay gate

The frozen formal replay gate passed **256/256 tests** at tolerance `1e-12`,
with zero failed formal tests. The full test records are in
`experiments/r13_p15_caaa_v2/stage1/branch_replay_validation.json`.

One development replay incident is deliberately retained in the report: the
initial local smoke omitted an integrated Panda gripper command from snapshot
state. Capturing gripper history and solver/control auxiliaries reduced formal
A/A and A/B/A differences to zero.

## Formal experiment and artifact checks

PAI job `dlc1wxel8qjf7ck8` (run
`r13p15-caaa-v2-stage1-20260812-f`) completed with status `Succeeded`. Its
completion sentinel binds the report SHA-256
`5924ee1a77bc8c9339c5f450a8be87b9f8ad1e1a91b683fbf848f5f3f2047dd5`,
69,120 realized quantization rows, and final disposition
`REJECT_CORE_HYPOTHESIS`.

Run `python scripts/verify_published_artifacts.py` after cloning. The verifier
checks all report-declared hashes; 256 branch shards, 128 quantized shards, 128
quantization plans, 256 fitted Jacobians, and 27 codebooks with their markers;
JSONL/CSV row counts; replay counts; the PAI completion sentinel; and the exact
final disposition.

The published formal Stage 1 tree contains 28,068 files and 209,699,045
logical bytes. Its largest file is the 44,498,042-byte realized-quantization
JSONL, below GitHub's 100 MB single-file ceiling, so no Git LFS indirection is
required.

## Stage 1.5 validation

The Stage 1.5 simulator run used CPU only (`CUDA_VISIBLE_DEVICES` empty,
`MUJOCO_GL=glx`, all renderers disabled). Four tasks ran in task-level CPU
parallelism. The corrected shard validator passed:

- 64 frozen old-test plans and 64 realized result shards;
- 288 rows per shard, K=64 only, for 18,432 revised branches total;
- every completion marker and payload SHA-256;
- exact equality of every plan-bound field in its realized shard;
- finite initial, immediate, settled and final-state arrays;
- zero collection failures.

`scripts/verify_stage1_5_artifacts.py --full-stage1-hash` completed with
`PASS`. The machine-readable record is
`provenance/stage1_5_release_verification.json`. It additionally verifies:

- all 12 required Stage 1.5 artifacts and every report-declared hash;
- 143 strict JSON files with zero invalid constants;
- the two-file preregistration commit preceded revised results;
- 256 retrospective diagnostic rows and 59/40/160/45 expected CSV rows;
- a 30,720-row internal screen and nine 10,000-replicate bootstrap comparisons;
- `NOT_COLLECTED_INTERNAL_SCREEN_FAILED` with zero fresh records/states;
- Stage 1 Git path identity and the complete bound Stage 1 tree SHA-256
  `047aae35193339a460cd1dbac0e4495d7f9cff4a1cb2799c58b738e86e0e4c5c`;
- final disposition `REJECT_P15_FAMILY`.

The report retains all development failures: one interrupted inefficient
decoder implementation, one dtype bug in an ad-hoc validator, and one
successful-but-non-strict JSON screen superseded by a deterministic strict
JSON rerun. None changed a simulator payload or the scientific disposition.

## Stage 3 validation

On 2026-08-14 UTC, after the full confirmation collection and before release,
the current suite passed in both relevant runtimes:

| Environment | Command | Result |
| --- | --- | --- |
| analysis Python 3.11 | `CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/openpi_py311/bin/python -m pytest -q` | 30 passed in 49.85 s |
| frozen LIBERO Python 3.8 | `MUJOCO_GL=glx CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero-original/bin/python -m pytest -q tests/test_stage3.py` | 12 passed in 16.74 s |
| analysis Python 3.11 | `python -m py_compile caaa_libero/*.py` | passed |

The Stage 3 release verifier additionally checks all 544 bound states and
their context/support/candidate shard hashes; 256/256 reused training states;
pair-score symmetry and exact zero self-distance; the exact 147,456
development and 368,640 holdout row counts; all 24 methods; candidate-order
invariance; 10,000 episode-clustered bootstrap replicates; post-disposition
K=32/128 sensitivity; one-visible-GPU provenance; zero PAI jobs; and no policy
training. Its machine-readable output is
`experiments/r13_p15_ncer_aa/stage3/STAGE3_RELEASE_VERIFICATION.json`.

The confirmation collection completed all 160 states (4 tasks x 10 episodes x
4 phases), but is deliberately labeled `FORCED_EXPLORATORY_HOLDOUT` because a
pre-result deterministic replay probe violated the literal untouched rule.
This integrity failure is retained and `GO_TO_SMALL_BC` remains unavailable.
