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

    @staticmethod
    def make(
        decision_type: DecisionType,
        object_id: str,
        subject_id: str,
        title: str,
        description: str,
        metadata: dict | frozendict = frozendict(),
        object_type: ObjectType = ObjectType.STACK,
    ) -> "DecisionPoint":
        key = (decision_type, object_type, object_id, subject_id)
        return DecisionPoint(
            key=key,
            decision_type=decision_type,
            default_choice=None,  # Filled by resolve_decision's lazy default
            title=title,
            description=description,
            metadata=frozendict(metadata) if isinstance(metadata, dict) else metadata,
            object_type=object_type,
            object_id=object_id,
            subject_id=subject_id,
        )


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
    """Resolve a decision value with read-through caching.

    Three-tier resolution:
    1. User override → return immediately.
    2. Cached default → return the previously-computed default from context.
    3. Compute default → evaluate the callable (or use the value), cache it
       in the DecisionPoint, publish to context, return it.

    The `default` parameter can be either a plain value or a callable. If
    callable, it's only invoked when there's no override and no cached
    default — expensive default computations (like calibration matching)
    are lazy and run at most once.

    The cached DecisionPoint in context serves double duty: it's the
    UI-visible decision metadata AND the cached default value. The UI can
    always show what the system would have done by reading default_choice.
    """
    from context import context
    from dataclasses import replace

    # 1. User override wins.
    override = overrides.get(decision.key)
    if override is not None:
        # Still publish the decision so the UI can show default vs. override.
        obj_key = _decisions_key(decision.object_type, decision.object_id)
        if decision.key not in context.get(obj_key, {}):
            # Compute default to cache it for UI display.
            value = default() if callable(default) else default
            decision = replace(decision, default_choice=value)
            context.or_equals(obj_key, {decision.key: decision})
        return override.choice

    # 2. Check for cached default in context.
    obj_key = _decisions_key(decision.object_type, decision.object_id)
    existing = context.get(obj_key, {})
    if decision.key in existing:
        return existing[decision.key].default_choice

    # 3. Compute default (lazy), cache it, publish it.
    value = default() if callable(default) else default
    decision = replace(decision, default_choice=value)
    context.or_equals(obj_key, {decision.key: decision})
    return value


def get_decisions_for_object(
    object_type: ObjectType, object_id: str,
) -> dict[DecisionKey, DecisionPoint]:
    """Read all decisions for a specific object from context."""
    from context import context
    return context.get(_decisions_key(object_type, object_id), {})


def decisions_for_stack(
    decision_overrides: frozendict, stack_id: str,
) -> frozendict:
    """Filter decision overrides to only those keyed to a specific stack.

    Decision keys are (DecisionType, ObjectType, object_id, subject_id)
    where position [2] is the stack hash. Passing per-stack filtered
    decisions into @node functions ensures that changing one stack's
    decisions doesn't invalidate another stack's cache.
    """
    return frozendict({
        k: v for k, v in decision_overrides.items()
        if len(k) >= 3 and k[2] == stack_id
    })
