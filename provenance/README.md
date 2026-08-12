# Provenance

This directory preserves publication and execution evidence that is not part
of the scientific result tables themselves.

- `tests/` contains JUnit XML from passing Python 3.8 and Python 3.10 runs.
- `pai/job_registry_runs/` contains the submitted, resolved, and server
  read-back records for attempts `c` through `f`, including failed attempts.
- `pai/artifact_evidence/` contains first-committed-shard, cleanup, and final
  completion sentinels written to durable storage by the jobs.
- `release_verification.json` is generated from the published tree by
  `scripts/verify_published_artifacts.py`.

No credentials or secret environment values were injected into these PAI jobs.
The copied records were scanned before publication; empty credential fields and
the explicit `secret_env_names: []` contract remain as audit evidence.
