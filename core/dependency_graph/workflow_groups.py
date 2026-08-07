"""Lexical workflow-graph groups — pure UI metadata.

`@workflow_group` is a decorator that tags every `record()`-recorded
Node made inside an invocation of the wrapped helper with a group
identifier. The workflow-graph viewer then renders the tagged Nodes
as a single collapsible group labelled with the helper's name.

This module is split out of `dependency_graph/tracing.py` because the
two concerns are orthogonal: tracing is "this is the engine that
turns Python control flow into a Graph"; workflow groups are "purely
how the viewer should render that graph". No runtime or cache
behavior depends on `@workflow_group` — outside an active trace the
decorator is a no-op.

Coupling left in place: `record()` (in `tracing.py`) reads
`_GROUP_STACK` to tag new Nodes. The ContextVar lives here so
`workflow_group` doesn't have to reach across module boundaries to
mutate it; `record()` imports it from this module.
"""
from __future__ import annotations

import contextvars
import functools

from hashing import hash_list_int


# Stack of (group_name, group_id) pairs for the active lexical
# `@workflow_group` scope. Each entry corresponds to one
# @workflow_group-decorated function currently on the call stack;
# `record()` reads this and tags every new Node with the full path,
# which the workflow-graph serializer uses to build collapsible
# group nodes in the viewer.
_GROUP_STACK: contextvars.ContextVar[tuple] = contextvars.ContextVar(
    "workflow_group_stack", default=()
)


def workflow_group(fn=None, *, display_name: str | None = None):
    """Decorator: mark a plain Python helper as a lexical
    workflow-graph group. Every `record()` call made during one
    invocation of the wrapped function (including via nested
    helpers and @node wrappers) tags its new Node with the
    function's name; the workflow-graph serializer renders the
    tagged Nodes as a single collapsible group in the viewer,
    labeled with the function's `__name__`.

    Pure UI grouping — does not change graph topology, runtime
    behavior, or caching. The decorated function still returns
    whatever it would normally return. Outside an active trace
    the decorator is a no-op (just calls the function directly).

    Example:

        @workflow_group
        def _build_cal_masters(stack, config):
            steps, top_ids = calibration_plan(stack, config)
            masters = {}
            for cs, dep_ids in steps:
                masters[str(cs.stable_hash)] = stack_files(
                    downscale_stack(8, cs), config,
                    *(masters[i] for i in dep_ids),
                )
            return [masters[i] for i in top_ids]

    Groups can nest — a helper decorated with `@workflow_group`
    that's called inside another `@workflow_group`-decorated
    helper inherits the outer group as a prefix on its Nodes'
    `_group_path`.

    The `group_id` is a deterministic hash of (function-name,
    parent_path, per-call key) so re-traces of the same lexical
    structure produce the same ids; the viewer's expanded/
    collapsed state stays stable across reruns.
    """
    # Dual-form: `@workflow_group` or `@workflow_group(display_name="...")`.
    # `display_name` replaces the function name as the group's viewer label
    # (and in the deterministic group_id — renaming resets that group's
    # expanded/collapsed state, nothing else; groups are pure UI).
    if fn is None:
        return functools.partial(workflow_group, display_name=display_name)

    # Local import to avoid a tracing → workflow_groups → tracing
    # round-trip at module load time.
    from dependency_graph.tracing import _TRACE
    name = display_name or fn.__name__

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if _TRACE.get() is None:
            return fn(*args, **kwargs)
        parent = _GROUP_STACK.get()
        # Per-call key so two distinct calls of the same
        # `@workflow_group` helper produce distinct group ids.
        # Most pipeline helpers take their primary entity
        # (stack / target / mosaic) as the first arg, all of
        # which expose a precomputed `stable_hash` string — using
        # that directly avoids the O(cal_stacks) cost of recursive
        # arg hashing through ProcessingConfig. Falls back to a
        # full args hash for callers whose first arg has no
        # stable_hash.
        if args and hasattr(args[0], "stable_hash"):
            call_key = args[0].stable_hash
        else:
            call_key = hash_list_int(
                [list(args), sorted(kwargs.items())]
            )
        group_id = str(hash_list_int([name, parent, call_key]))
        token = _GROUP_STACK.set(parent + ((name, group_id),))
        try:
            return fn(*args, **kwargs)
        finally:
            _GROUP_STACK.reset(token)
    return wrapper
