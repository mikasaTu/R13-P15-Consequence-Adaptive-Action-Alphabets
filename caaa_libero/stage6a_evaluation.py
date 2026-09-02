"""Guarded Stage 6-A effect lookup primitives.

The full evaluator is unreachable after a failed Gate H.  Keeping the lookup
primitive here makes the executed-only contract testable without weakening
the preregistered stop rule.
"""

from __future__ import annotations

import numpy as np


def require_gate_h(gate):
    if not bool(gate.get("passed")):
        raise RuntimeError("Gate H failed; effect-error evaluation is forbidden")


def executed_consequence_lookup(executed_true_distance, selected_local_index):
    """Return only entries from an already-executed consequence table."""
    table = np.asarray(executed_true_distance, dtype=np.float64)
    selected = np.asarray(selected_local_index, dtype=np.int64)
    if table.ndim != 3 or selected.shape != table.shape[:2]:
        raise ValueError("executed lookup shape mismatch")
    if np.any(selected < 0) or np.any(selected >= table.shape[2]):
        raise IndexError("selected candidate is outside executed bank")
    state = np.arange(table.shape[0], dtype=np.int64)[:, None]
    target = np.arange(table.shape[1], dtype=np.int64)[None, :]
    return table[state, target, selected]


def selection_input_contract(context, target_residual, candidate_residual):
    """Explicitly omit candidate/target identifiers from learned-model inputs."""
    return (
        np.asarray(context, dtype=np.float32),
        np.asarray(target_residual, dtype=np.float32),
        np.asarray(candidate_residual, dtype=np.float32),
    )
