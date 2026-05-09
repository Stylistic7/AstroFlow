"""Per-(target, filter) merged-master artifact and subgraph builder.

A `TargetChannel` is the canonical "this target's R-channel image" — one
master per (target, filter) pair, regardless of whether the user shot
one stack at one exposure or multiple stacks across exposures (HDR).

The sentinel dataclass exists only so `@makes_image(TargetChannel, ...)`
can register the merge producer. The actual merge implementation lives
in `merge_target_channel.py` (PIXINSIGHT) and `merge_target_channel_oss.py`
(OSS).

This module is the encapsulation boundary: callers above the Target
layer (e.g. mosaic placement) ask `channel_stacks_for(target, filter)`
or invoke `build_target_channel_subgraph(...)` and never poke at Stack
internals directly.
"""
from dataclasses import dataclass, replace

import ray
from beartype import beartype

from context import context
from decisions.decisions import decisions_for_stack
from decisions.settings import ProcessingConfig
from dependency_graph import Graph, Node, chain
from metadata.image_metadata import compute_image_metadata
from previews.types import QualityLevel
from products import get_producer, MediaType
from stacks.composite_stack import Stack
from stacks.compute_graph import build_stacking_subgraph, get_stack_subframe_image
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


@ray.remote(num_cpus=0.1)
@beartype
def build_target_channel_subgraph(
    target: Target,
    filter: str,
    stacks: tuple[Stack, ...],
    config: ProcessingConfig,
) -> Graph:
    """Per-(target, filter) subgraph: stack each constituent, then merge.

    SUBFRAME mode short-circuits to the first stack's best frame —
    "merge a single preview frame" doesn't make scientific sense, and
    a SUBFRAME mosaic preview just wants something fast and
    representative. MASTER / DOWNSCALED_MASTER modes go through the
    full stack-then-merge path.
    """
    if config.quality == QualityLevel.SUBFRAME:
        # Pick the first stack as the representative for this channel
        # in preview mode. Same behavior the old code had implicitly
        # via "first matching stack wins".
        assert stacks, f"channel ({target}, {filter}) has no stacks"
        return Graph({
            **chain(
                (get_stack_subframe_image, stacks[0]),
                Node(
                    compute_image_metadata,
                    name=f"channel_subframe_{target.stable_hash}_{filter}",
                ),
            )
        })

    def config_for(stack: Stack) -> ProcessingConfig:
        return replace(
            config,
            decisions=decisions_for_stack(config.decisions, stack.stable_hash),
        )

    per_stack_tasks = [
        (build_stacking_subgraph, stack, config_for(stack))
        for stack in stacks
    ]

    merge_fn = get_producer(
        MediaType.IMAGE, TargetChannel, config.quality, config.implementation,
    )
    assert merge_fn is not None, (
        f"No TargetChannel producer registered for "
        f"({config.quality}, {config.implementation})"
    )
    merge_task = (merge_fn, target, filter, stacks, config)

    graph = Graph()
    for task in per_stack_tasks:
        graph[task] = set()
    graph[merge_task] = set(per_stack_tasks)
    return graph
