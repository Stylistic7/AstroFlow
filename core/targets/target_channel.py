"""Per-(target, filter) artifact identity and grouping helpers.

A `TargetChannel` is the canonical "this target's R-channel image" — one
master per (target, filter) pair, regardless of whether the user shot
one stack at one exposure or multiple stacks across exposures (HDR).

The sentinel dataclass exists only so `@makes_image(TargetChannel, ...)`
can register the merge producer. The actual merge implementation lives
in `targets/merging/pixinsight.py` (PixInsight HDRComposition) and
`targets/merging/oss.py` (the PixInsight-free HDR combine).

The graph-building that pipelines do (stack + merge per channel) is in
each `@pipeline` body now — see `core/pipelines/*.py`.
"""
from dataclasses import dataclass
from functools import lru_cache

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


@lru_cache(maxsize=None)
def _resolved_by_filter(stack_ids: tuple) -> dict[str, tuple[Stack, ...]]:
    """Read every Stack referenced by `stack_ids` from context and
    group them by filter. Cached on `stack_ids`: a target's stack-id
    set is stable in a session (Target is frozen), so this loads
    each constituent Stack at most once per process.

    Measured cost without the cache: 10.8s for a 24-target pipeline
    trace because the pipeline body's nested loops re-dereference
    every stack each time (~13ms per ContextRef.get() over Ray RPC
    + pickle on the web server). With the cache the same trace
    pays the deref cost once per (target.stack_ids) tuple and
    re-uses the by-filter grouping.

    Returned dict is keyed by filter; values are tuples so callers
    that index `[filter]` get an immutable result. Cache stores
    a single shared object per key — don't mutate it.
    """
    stacks = [context.stacks[sid].get() for sid in stack_ids]
    return {
        f: tuple(group)
        for f, group in group_stacks_by_filter(stacks).items()
    }


def channel_stacks_for(target: Target, filter: str) -> tuple[Stack, ...]:
    """The constituent Stacks for one (target, filter) channel."""
    return _resolved_by_filter(target.stack_ids).get(filter, ())


def filters_for_target(target: Target) -> list[str]:
    """Filters this target was imaged in, in deterministic order."""
    return list(_resolved_by_filter(target.stack_ids).keys())


def _quarantine(decision_overrides):
    from decisions.quarantine import Quarantine
    return Quarantine(decision_overrides, lambda sid: context.stacks[sid].get())


def live_channels(target: Target, decision_overrides) -> dict[str, tuple[Stack, ...]]:
    """`filter -> member stacks` with the quarantined ones out
    (`decisions/quarantine.py`): a filter with no member left is absent,
    and a quarantined target has no channels at all. To every pipeline the
    panel simply was not imaged in that filter.

    Membership is settled HERE, at trace time, not inside the integration:
    a stack that stays promised with nothing to integrate raises "no frames
    after applying decisions", and one such panel channel held every finish
    of its mosaic at the previous generation (549e 2026-10-24, e308
    2026-10-27). Identities are untouched — the stack stays in its Target,
    the Target in its Mosaic — so clearing the key brings the channel
    straight back from cache. The computed subframe gate reads metadata and
    stays the integration's to refuse."""
    quarantine = _quarantine(decision_overrides)
    if quarantine.target(target.stable_hash, target.stack_ids):
        return {}
    channels = {}
    for filter, stacks in _resolved_by_filter(target.stack_ids).items():
        members = tuple(s for s in stacks if not quarantine.stack(s.stable_hash))
        if members:
            channels[filter] = members
    return channels


def live_panels(mosaic, decision_overrides) -> tuple[Target, ...]:
    """The mosaic's targets with the quarantined ones out; none when the
    mosaic itself is. Same rule and reason as `live_channels`."""
    quarantine = _quarantine(decision_overrides)
    targets = mosaic.targets
    if quarantine.mosaic(mosaic.stable_hash, ((t.stable_hash, t.stack_ids) for t in targets)):
        return ()
    return tuple(t for t in targets if not quarantine.target(t.stable_hash, t.stack_ids))
