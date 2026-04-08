"""Convenience functions for common decision framework patterns.

Pure key/override builders plus verb-based actions that store overrides
into context.settings. Use the full framework directly for unusual cases.
"""
from dataclasses import replace
from frozendict import frozendict

from decisions.decisions import (
    DecisionKey,
    DecisionOverride,
    DecisionType,
    ObjectType,
    BinaryChoice,
)


# ---------------------------------------------------------------------------
# Key builders
# ---------------------------------------------------------------------------

def quarantine_key(
    object_type: ObjectType, object_id: str, parent_id: str | None = None,
) -> DecisionKey:
    """Build the decision key for quarantining an object.

    For run-level quarantine (a run within a stack), parent_id is the
    stack hash. Position [2] must be the stack hash so per-stack
    decision filtering picks it up; position [3] is the run hash so
    _apply_run_exclusions knows which run to exclude.
    """
    return (DecisionType.QUARANTINE, object_type, parent_id or object_id, object_id)


_CAL_DECISION_TYPES = {
    "bias": DecisionType.CALIBRATION_BIAS,
    "dark": DecisionType.CALIBRATION_DARK,
    "flat": DecisionType.CALIBRATION_FLAT,
}


def calibration_key(cal_type: str, stack_id: str, run_id: str) -> DecisionKey:
    """Build the decision key for a calibration override (bias/dark/flat)."""
    return (_CAL_DECISION_TYPES[cal_type], ObjectType.STACK, stack_id, run_id)


# ---------------------------------------------------------------------------
# Override builders (return single-entry dicts, composable with |)
# ---------------------------------------------------------------------------

def exclude(key: DecisionKey) -> dict[DecisionKey, DecisionOverride]:
    """Single-entry override dict that excludes."""
    return {key: DecisionOverride(choice=BinaryChoice.EXCLUDE)}


def include(key: DecisionKey) -> dict[DecisionKey, DecisionOverride]:
    """Single-entry override dict that includes."""
    return {key: DecisionOverride(choice=BinaryChoice.INCLUDE)}


def calibration_override(
    cal_type: str, stack_id: str, run_id: str, cal_stack_id: str,
) -> dict[DecisionKey, DecisionOverride]:
    """Single-entry override dict assigning a calibration stack to a run."""
    return {calibration_key(cal_type, stack_id, run_id): DecisionOverride(choice=cal_stack_id)}


# ---------------------------------------------------------------------------
# Actions (read-modify-write context.settings.decision_overrides)
# ---------------------------------------------------------------------------

def _add_overrides(overrides: dict[DecisionKey, DecisionOverride]) -> None:
    """Merge overrides into context.settings.decision_overrides."""
    from context import context
    current = dict(context.settings.decision_overrides)
    current.update(overrides)
    context.settings = replace(context.settings, decision_overrides=frozendict(current))


def set_quarantine(
    object_type: ObjectType, object_id: str, parent_id: str | None = None,
) -> None:
    """Quarantine an object (stack, target, or mosaic)."""
    _add_overrides(exclude(quarantine_key(object_type, object_id, parent_id)))


def set_calibration(cal_type: str, stack_id: str, run_id: str, cal_stack_id: str) -> None:
    """Assign a calibration stack (bias/dark/flat) to a run."""
    _add_overrides(calibration_override(cal_type, stack_id, run_id, cal_stack_id))


def set_flat(stack_id: str, run_id: str, flat_stack_id: str) -> None:
    """Assign a flat calibration stack to a run."""
    set_calibration("flat", stack_id, run_id, flat_stack_id)
