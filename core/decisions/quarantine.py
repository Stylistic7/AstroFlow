"""Effective quarantine, derived.

An object is quarantined when it carries the user's own key, or when every
child is: run → stack → target → mosaic. Nothing is stored. The overrides
and the object tree are the only inputs, so processing (which runs a stack
integrates, which stacks a channel has, which targets a mosaic places), the
viewer and the MCP sidecar all read this one rule.

Before 2026-10-28 each site had a partial version of its own: the
integration dropped only the keyed runs, so a STACK key dropped just the run
carrying the stack's id and the rest still integrated; the viewer dimmed a
target only when every stack carried a STACK key; nothing anywhere
quarantined a mosaic; and a stack whose every run was keyed stayed promised
to its mosaic with nothing to integrate, which held every finish of e308 at
the previous generation.

Upward only. A key on a target keeps it out of every mosaic and gives it no
products of its own; its stacks still integrate (their masters are cheap and
stay useful) and their frames are not the trash collector's.
"""
from collections import defaultdict
from collections.abc import Callable, Iterable

from decisions.decisions import BinaryChoice, DecisionType, ObjectType


class Quarantine:
    """The user's quarantine keys, read once, answered for any level.

    `stack_of(id) -> Stack | None` is consulted only for a stack that RUN
    keys name — a stack nobody keyed is not quarantined, and neither is
    anything built from such stacks alone — so the viewer answers for the
    whole library at the cost of a few fetches."""

    def __init__(self, overrides, stack_of: Callable | None = None):
        keyed = [
            key for key, override in overrides.items()
            if key.decision_type == DecisionType.QUARANTINE
            and override.choice == BinaryChoice.EXCLUDE
        ]
        self.explicit = {
            t: frozenset(k.subject_id for k in keyed if k.object_type == t)
            for t in (ObjectType.RUN, ObjectType.STACK, ObjectType.TARGET, ObjectType.MOSAIC)
        }
        self.runs_by_stack: dict[str, set[str]] = defaultdict(set)
        for k in keyed:
            if k.object_type == ObjectType.RUN:
                self.runs_by_stack[k.parent_stack_id].add(k.subject_id)
        self.stack_of = stack_of

    def runs_of(self, stack) -> frozenset[str]:
        """The runs of `stack` that are out: all of them under a STACK key,
        else the ones under their own RUN key."""
        runs = frozenset(stack.contents)
        if stack.stable_hash in self.explicit[ObjectType.STACK]:
            return runs
        return runs & frozenset(self.runs_by_stack.get(stack.stable_hash, ()))

    def stack(self, stack_id: str) -> bool:
        if stack_id in self.explicit[ObjectType.STACK]:
            return True
        if stack_id not in self.runs_by_stack:
            return False
        assert self.stack_of is not None, "a keyed run needs its stack to answer"
        stack = self.stack_of(stack_id)
        return (stack is not None and bool(stack.contents)
                and self.runs_of(stack) == frozenset(stack.contents))

    def target(self, target_id: str, stack_ids: Iterable[str]) -> bool:
        stack_ids = tuple(stack_ids)
        return (target_id in self.explicit[ObjectType.TARGET]
                or (bool(stack_ids) and all(self.stack(sid) for sid in stack_ids)))

    def mosaic(self, mosaic_id: str, panels: Iterable[tuple[str, Iterable[str]]]) -> bool:
        """`panels`: `(target_id, stack_ids)` per target of the mosaic."""
        panels = tuple(panels)
        return (mosaic_id in self.explicit[ObjectType.MOSAIC]
                or (bool(panels) and all(self.target(tid, sids) for tid, sids in panels)))
