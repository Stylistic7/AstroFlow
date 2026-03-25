from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Any
from frozendict import frozendict
from hashing import hash_dataclass


class DecisionType(str, Enum):
    QUARANTINE = "quarantine"
    ROTATION_OUTLIER = "rotation_outlier"
    SMALL_RUN = "small_run"
    SUBFRAME_QUALITY = "subframe_quality"
    CALIBRATION_BIAS = "calibration_bias"
    CALIBRATION_DARK = "calibration_dark"
    CALIBRATION_FLAT = "calibration_flat"


# Lower precedence = applied first by resolve_decisions.
DECISION_PRECEDENCE: dict["DecisionType", int] = {
    DecisionType.QUARANTINE: 0,
    DecisionType.ROTATION_OUTLIER: 1,
    DecisionType.SMALL_RUN: 2,
    DecisionType.SUBFRAME_QUALITY: 3,
}


class BinaryChoice(str, Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"


class ObjectType(str, Enum):
    STACK = "stack"
    TARGET = "target"
    MOSAIC = "mosaic"


# Unique identifier for a decision — deterministic tuple of enums/hashes.
# e.g. (DecisionType.ROTATION_OUTLIER, ObjectType.STACK, stack_hash, substack_hash)
DecisionKey = tuple[DecisionType, ObjectType, str, str]


@dataclass(frozen=True)
class DecisionPoint:
    key: DecisionKey
    decision_type: DecisionType
    default_choice: Any
    title: str
    description: str
    metadata: frozendict = field(default_factory=frozendict)
    object_type: ObjectType = ObjectType.STACK
    object_id: str = ""
    subject_id: str = ""

    def __hash__(self):
        return hash_dataclass(self)


@dataclass(frozen=True)
class DecisionOverride:
    choice: Any

    def __hash__(self):
        return hash_dataclass(self)


def _decisions_key(object_type: ObjectType, object_id: str):
    """Context key for per-object decision storage."""
    return ("decisions", object_type.value, object_id)


def resolve_decision(
    decision: DecisionPoint,
    overrides: frozendict[DecisionKey, DecisionOverride],
    default: Any,
) -> Any:
    """Publish a decision (if not already published) and resolve its value.

    1. Publishes the decision to per-object storage in context if not already
       there.
    2. Resolves: user override wins, otherwise the caller's default.

    The default comes from the calling code, not from the cached DecisionPoint.
    This ensures code changes to defaults take effect immediately, unless the
    user has explicitly overridden the decision.

    The return value can be any type — BinaryChoice for include/exclude
    decisions, or a numeric threshold for quality decisions, etc.
    """
    from context import context
    obj_key = _decisions_key(decision.object_type, decision.object_id)
    decisions = context.get(obj_key, {})
    if decision.key not in decisions:
        context.or_equals(obj_key, {decision.key: decision})

    override = overrides.get(decision.key)
    if override is not None:
        return override.choice
    return default


def get_decisions_for_object(
    object_type: ObjectType, object_id: str,
) -> dict[DecisionKey, DecisionPoint]:
    """Read all decisions for a specific object from context."""
    from context import context
    return context.get(_decisions_key(object_type, object_id), {})
