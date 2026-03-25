from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from frozendict import frozendict
from hashing import hash_dataclass


class DecisionType(str, Enum):
    ROTATION_OUTLIER = "rotation_outlier"
    SMALL_RUN = "small_run"
    SUBFRAME_QUALITY = "subframe_quality"


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
    default_choice: BinaryChoice
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
    choice: BinaryChoice

    def __hash__(self):
        return hash_dataclass(self)


def resolve_decision(
    decision: DecisionPoint,
    overrides: frozendict[DecisionKey, DecisionOverride],
) -> BinaryChoice:
    override = overrides.get(decision.key)
    if override is not None:
        return override.choice
    return decision.default_choice


def resolve_decisions_for_stack(
    decisions: dict[DecisionKey, DecisionPoint],
    overrides: frozendict[DecisionKey, DecisionOverride],
    stack_id: str,
) -> frozendict[DecisionKey, BinaryChoice]:
    """Resolve all decisions affecting a specific stack.

    Returns a frozen dict suitable for use as a task argument (hashable).
    """
    relevant = {
        k: resolve_decision(d, overrides)
        for k, d in decisions.items()
        if k[1] == ObjectType.STACK and k[2] == stack_id
    }
    return frozendict(relevant)
