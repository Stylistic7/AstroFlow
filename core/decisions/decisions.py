from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Any
from frozendict import frozendict
from hashing import hash_dataclass


class DecisionType(str, Enum):
    QUARANTINE = "quarantine"
    ROTATION_OUTLIER = "rotation_outlier"
    SUBFRAME_QUALITY = "subframe_quality"
    CALIBRATION_BIAS = "calibration_bias"
    CALIBRATION_DARK = "calibration_dark"
    CALIBRATION_FLAT = "calibration_flat"


# Lower precedence = applied first by resolve_decisions.
DECISION_PRECEDENCE: dict["DecisionType", int] = {
    DecisionType.QUARANTINE: 0,
    DecisionType.ROTATION_OUTLIER: 1,
    DecisionType.SUBFRAME_QUALITY: 2,
}


class BinaryChoice(str, Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"


class ObjectType(str, Enum):
    STACK = "stack"
    TARGET = "target"
    MOSAIC = "mosaic"


_CAL_TYPE_TO_DECISION_TYPE: dict[str, DecisionType] = {
    "bias": DecisionType.CALIBRATION_BIAS,
    "dark": DecisionType.CALIBRATION_DARK,
    "flat": DecisionType.CALIBRATION_FLAT,
}


@dataclass(frozen=True)
class DecisionKey:
    """A typed key into the decision overrides dict.

    `parent_stack_id` is the stack hash that owns this decision — used
    by per-stack cache filtering. For stack/target/mosaic-level
    decisions, `parent_stack_id == subject_id`. For run-level decisions,
    `subject_id` is the run hash.
    """
    decision_type: DecisionType
    object_type: ObjectType
    parent_stack_id: str
    subject_id: str

    def __hash__(self):
        return hash_dataclass(self)

    def matches_stack(self, stack_id: str) -> bool:
        """True if this decision belongs to the given stack."""
        return self.parent_stack_id == stack_id

    @classmethod
    def quarantine(
        cls,
        object_type: "ObjectType",
        object_id: str,
        parent_stack_id: str | None = None,
    ) -> "DecisionKey":
        return cls(
            DecisionType.QUARANTINE,
            object_type,
            parent_stack_id or object_id,
            object_id,
        )

    @classmethod
    def rotation_outlier(cls, stack_id: str, run_id: str) -> "DecisionKey":
        return cls(
            DecisionType.ROTATION_OUTLIER, ObjectType.STACK, stack_id, run_id,
        )

    @classmethod
    def subframe_quality(cls, stack_id: str) -> "DecisionKey":
        return cls(
            DecisionType.SUBFRAME_QUALITY, ObjectType.STACK, stack_id, stack_id,
        )

    @classmethod
    def calibration(
        cls, cal_type: str, stack_id: str, run_id: str,
    ) -> "DecisionKey":
        return cls(
            _CAL_TYPE_TO_DECISION_TYPE[cal_type],
            ObjectType.STACK,
            stack_id,
            run_id,
        )


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
        key: DecisionKey,
        title: str,
        description: str,
        metadata: dict | frozendict = frozendict(),
    ) -> "DecisionPoint":
        return DecisionPoint(
            key=key,
            decision_type=key.decision_type,
            default_choice=None,  # Filled by resolve_decision's lazy default
            title=title,
            description=description,
            metadata=frozendict(metadata) if isinstance(metadata, dict) else metadata,
            object_type=key.object_type,
            object_id=key.parent_stack_id,
            subject_id=key.subject_id,
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

    Passing per-stack filtered decisions into @node functions ensures
    that changing one stack's decisions doesn't invalidate another
    stack's cache.
    """
    return frozendict({
        k: v for k, v in decision_overrides.items()
        if k.matches_stack(stack_id)
    })
