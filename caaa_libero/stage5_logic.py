"""Frozen Stage 5 disposition precedence."""

from __future__ import annotations

from .stage5_config import FINAL_DISPOSITIONS


def choose_disposition(
    historical_binding_valid,
    oracle_gate_passed,
    context_gate_passed,
    static_consequence_gate_passed,
    dynamic_k64_gate_passed,
    fresh_trajectory_supply_complete,
    confirmation_gate_passed,
):
    if not historical_binding_valid:
        result = "BLOCKED_HISTORICAL_BINDING_MISMATCH"
    elif not oracle_gate_passed:
        result = "STATIC_EFFECT_GEOMETRY_SUFFICIENT"
    elif not context_gate_passed:
        result = (
            "STATIC_CONSEQUENCE_METRIC_ONLY"
            if static_consequence_gate_passed
            else "REJECT_LEARNED_CONSEQUENCE_METRIC"
        )
    elif not dynamic_k64_gate_passed:
        result = "PIVOT_TO_CONSEQUENCE_RETRIEVAL_STEERING"
    elif not fresh_trajectory_supply_complete:
        result = "BLOCKED_NO_FRESH_TRAJECTORIES"
    elif not confirmation_gate_passed:
        result = "CONFIRMATION_FAILED"
    else:
        result = "GO_TO_FIXED_POLICY_RERANKING"
    if result not in FINAL_DISPOSITIONS:
        raise AssertionError(result)
    return result


def exact_one_disposition(value):
    return isinstance(value, str) and value in FINAL_DISPOSITIONS
