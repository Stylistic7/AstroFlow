"""Apply decisions to a CompositeStack to produce filtered or annotated views."""
from __future__ import annotations

from dataclasses import replace
from frozendict import frozendict

from stacks.composite_stack import CompositeStack
from stacks.simple_stack import RunsDict
from customization.decisions import (
    BinaryChoice, DecisionKey, DecisionOverride, DecisionType,
    ObjectType, DECISION_PRECEDENCE, resolve_decision, get_decisions_for_object,
)
from customization.subframe_selection import select_subframes
from quarantining.quarantining import apply_quarantine


def _apply_run_exclusions(
    stack: CompositeStack,
    decision_overrides: frozendict[DecisionKey, DecisionOverride],
    decision_type: DecisionType,
) -> CompositeStack | None:
    """Remove runs excluded by a specific decision type."""
    excluded = {
        key[3]  # subject_id = run hash
        for key, override in decision_overrides.items()
        if (
            key[0] == decision_type
            and key[1] == ObjectType.STACK
            and override.choice == BinaryChoice.EXCLUDE
        )
    }
    if not excluded:
        return stack
    return apply_quarantine(stack, excluded)


def resolve_decisions(
    stack: CompositeStack,
    decision_types: tuple[DecisionType, ...],
    decision_overrides: frozendict[DecisionKey, DecisionOverride],
    min_frames_in_stack: int,
) -> CompositeStack | None:
    """Apply the specified decision types to produce a filtered view.

    Decision types are automatically sorted by DECISION_PRECEDENCE (lower
    first): run-level exclusions before frame-level exclusions.
    """
    sorted_types = sorted(
        decision_types,
        key=lambda dt: DECISION_PRECEDENCE.get(dt, 999),
    )

    for dt in sorted_types:
        if stack is None:
            return None
        if dt in (DecisionType.QUARANTINE, DecisionType.ROTATION_OUTLIER):
            stack = _apply_run_exclusions(stack, decision_overrides, dt)
        elif dt == DecisionType.SUBFRAME_QUALITY:
            stack = select_subframes(
                stack, min_frames_in_stack,
                decision_overrides=decision_overrides,
            )

    return stack


def is_frame_included(
    frame_hash: str,
    stack: CompositeStack,
    decision_overrides: frozendict[DecisionKey, DecisionOverride],
) -> bool:
    """Check if a frame survives the quality filter. For per-frame annotation."""
    metadata = stack.all_metadata()
    if frame_hash not in metadata:
        return False
    m = metadata[frame_hash]
    if m.plate_solution is None:
        return False

    all_metadata = stack.all_metadata()
    star_counts = [
        md.plate_solution.num_stars
        for md in all_metadata.values()
        if md.plate_solution is not None
    ]
    if not star_counts:
        return True

    from customization.subframe_selection import _QUALITY_THRESHOLD
    best = max(star_counts)
    threshold = int(_QUALITY_THRESHOLD * best)

    # Check if the quality filter is overridden (user chose to skip it).
    quality_key = (
        DecisionType.SUBFRAME_QUALITY, ObjectType.STACK,
        stack.stable_hash, stack.stable_hash,
    )
    override = decision_overrides.get(quality_key)
    if override is not None and override.choice == BinaryChoice.EXCLUDE:
        return True  # Filter skipped → all frames included.

    return m.plate_solution.num_stars >= threshold


def is_run_included(
    run_hash: str,
    stack: CompositeStack,
    decision_overrides: frozendict[DecisionKey, DecisionOverride],
) -> bool:
    """Check if a run survives rotation/quarantine decisions. For per-run annotation."""
    for dt in (DecisionType.QUARANTINE, DecisionType.ROTATION_OUTLIER):
        key = (dt, ObjectType.STACK, stack.stable_hash, run_hash)
        override = decision_overrides.get(key)
        if override is not None and override.choice == BinaryChoice.EXCLUDE:
            return False

    # Check published decisions — if the default is EXCLUDE (unlikely but possible).
    decisions = get_decisions_for_object(ObjectType.STACK, stack.stable_hash)
    for dt in (DecisionType.QUARANTINE, DecisionType.ROTATION_OUTLIER):
        key = (dt, ObjectType.STACK, stack.stable_hash, run_hash)
        dp = decisions.get(key)
        if dp is not None:
            choice = resolve_decision(dp, decision_overrides, default=dp.default_choice)
            if choice == BinaryChoice.EXCLUDE:
                return False

    return True
