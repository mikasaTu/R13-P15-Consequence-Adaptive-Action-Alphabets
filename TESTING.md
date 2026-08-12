# Validation record

This record distinguishes code checks, simulator replay validation, artifact
integrity, and the formal PAI outcome.

## Code tests

On 2026-08-13 UTC, the unchanged seven-test suite passed in both relevant
interpreters:

| Environment | Command | Result |
| --- | --- | --- |
| frozen LIBERO Python 3.8.13 | `/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero-original/bin/python -m pytest -q` | 7 passed in 0.59 s |
| release-check Python 3.10.19 | `python -m pytest -q` | 7 passed in 0.22 s |

The clean system Python initially had no `pytest`; after installing it, test
collection exposed missing `numpy` and then `h5py`. Those test dependencies
were installed and the suite passed. These were environment setup failures,
not assertion failures. JUnit XML from the passing runs is retained in
`provenance/tests/`.

The suite covers deterministic perturbation directions and ridge math,
coordinate transforms, calibration-only baseline selection, strict resumed
quantized-plan identity, report logic, and the PAI launcher contract.

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
