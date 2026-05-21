"""Per-(target, filter) artifact identity and grouping helpers.

A `TargetChannel` is the canonical "this target's R-channel image" — one
master per (target, filter) pair, regardless of whether the user shot
one stack at one exposure or multiple stacks across exposures (HDR).

The sentinel dataclass exists only so `@makes_image(TargetChannel, ...)`
can register the merge producer. The actual merge implementation lives
in `merge_target_channel.py` (PIXINSIGHT) and `merge_target_channel_oss.py`
(OSS).

The graph-building that pipelines do (stack + merge per channel) is in
each `@pipeline` body now — see `core/pipelines/*.py`.
"""
from dataclasses import dataclass

from context import context
from stacks.composite_stack import Stack
from targets.types import Target


@dataclass(frozen=True)
class TargetChannel:
    """Sentinel for the producer registry; payload is on disk."""
    pass


def group_stacks_by_filter(stacks: list[Stack]) -> dict[str, list[Stack]]:
    """Group light stacks by their filter, dropping non-light stacks.

    Stable ordering: filters alphabetical, stacks-within-filter by
    stable_hash so the cache key is deterministic.
    """
    by_filter: dict[str, list[Stack]] = {}
    for s in stacks:
        if s.filter is None:
            continue
        by_filter.setdefault(s.filter, []).append(s)
    return {
        f: sorted(by_filter[f], key=lambda s: s.stable_hash)
        for f in sorted(by_filter)
    }


def channel_stacks_for(target: Target, filter: str) -> tuple[Stack, ...]:
    """The constituent Stacks for one (target, filter) channel.

    Resolves from the live `context.stacks`. Callers above the Target
    layer (mosaic placement, target post-processing) use this instead
    of inspecting Stacks directly.
    """
    target_stacks = [context.stacks[sid].get() for sid in target.stack_ids]
    return tuple(group_stacks_by_filter(target_stacks).get(filter, []))


def filters_for_target(target: Target) -> list[str]:
    """Filters this target was imaged in, in deterministic order."""
    target_stacks = [context.stacks[sid].get() for sid in target.stack_ids]
    return list(group_stacks_by_filter(target_stacks).keys())
