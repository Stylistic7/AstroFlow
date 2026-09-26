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
    # Not a calibration CHOICE but a promotion: "the frames of this stack are
    # blank sky, build a NIGHT-SKY FLAT from them". The user is the only one
    # who can know that — blank-sky frames arrive stamped `Light` like any
    # other, so nothing in the header distinguishes them. The result is an
    # ordinary flat and is chosen through CALIBRATION_FLAT like any other
    # (see `stacks/sky_flat.py`).
    #
    # The VALUE is a persistence identifier: DecisionKeys are pickled into
    # the durable store and a str-Enum unpickles BY VALUE, so a member whose
    # value vanishes fails the ENTIRE settings blob on load (every override,
    # not just its own). It therefore keeps the value it was first stored
    # under. Rename it only with a migration that rewrites the stored keys.
    SKY_FLAT_SOURCE = "fringe_source"


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


@dataclass(frozen=True, order=True)
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
    def sky_flat_source(cls, stack_id: str) -> "DecisionKey":
        """Whether this stack's frames are blank sky to build a NIGHT-SKY
        FLAT from. A whole-stack decision, so subject and parent are the same
        — the same shape `subframe_quality` uses."""
        return cls(
            DecisionType.SKY_FLAT_SOURCE, ObjectType.STACK, stack_id, stack_id,
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
    """Resolve a decision value.

    Two-tier resolution:
    1. User override → return immediately.
    2. Otherwise → compute the default fresh from the supplied callable
       (or use the value).

    The cache in context is kept up to date for the UI's benefit (what
    list_decisions reads) but is never consulted as the source of truth
    here. A "default" is by definition "what the current code would
    pick now"; serving a stale cached default for a calibration choice
    that no longer exists in cal_stacks is the bug class this avoids.
    Computing fresh on every call is cheap relative to the work that
    follows it — the heavy index-building inside callers like
    resolve_calibration is amortized once per call site, not per
    decision.
    """
    from context import context
    from dataclasses import replace

    value = default() if callable(default) else default

    # Refresh the cached DecisionPoint so the UI display path sees the
    # current default. Pipeline never reads back from this cache.
    decision = replace(decision, default_choice=value)
    obj_key = _decisions_key(decision.object_type, decision.object_id)
    context.or_equals(obj_key, {decision.key: decision})

    override = overrides.get(decision.key)
    if override is not None:
        return override.choice

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


def decisions_for_objects(
    decision_overrides: frozendict, object_ids,
) -> frozendict:
    """Filter decision overrides to those OWNED by any id in `object_ids` —
    the containment closure of a pipeline's entity (the object itself plus
    every contained object: a mosaic's targets, their stacks).

    `parent_stack_id` is the owning object's id for every DecisionKey shape
    (== subject_id for object-level keys, == the owning stack for run-level
    keys), so ownership is a single-field membership test.

    The mosaic/target analogue of `decisions_for_stack`, and the same
    invariant: a decision on an UNRELATED object must never move this
    pipeline's cache keys. Before this existed, mosaic pipelines baked the
    global overrides dict — one quarantine click on any object re-keyed
    every mosaic's full-config consumers (place_image and downstream)."""
    ids = frozenset(object_ids)
    return frozendict({
        k: v for k, v in decision_overrides.items()
        if k.parent_stack_id in ids
    })
