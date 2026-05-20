"""Define-by-run graph tracing.

A graph builder written as straight-line Python: call `record(op, *args)`
wherever you'd otherwise hand-build a `Node` + manually wire
`graph[node] = {deps}`. Any argument that is itself a recorded handle
(a `Node`) becomes a dependency edge; everything else is a literal.
`trace(builder, ...)` runs the builder under a recording context and
returns the accumulated `Graph` — the exact same `Graph`/`Node` objects
`GraphRuntime` already consumes (no new execution engine).

Node identity is structural: `Node.__eq__/__hash__` already key on
`(fn, name, args)`. Literal args distinguish e.g. per-target placements
automatically; for nodes with no distinguishing literal args (e.g.
`compute_image_metadata(handle)`) a deterministic `name` is synthesised
recursively from the op + its dependency handles, so identical
sub-pipelines dedup and different ones stay distinct — no per-call
counters, no hand-assigned names.

Higher-level surface: `@graph_builder` wraps the boilerplate of "run the
builder under `trace(...)` and return the resulting Graph via a Ray
task" into a decorator. `set_workflow_key(...)` is the function-body
helper for attaching the standard workflow callbacks to the active
trace, replacing `graph_update_fn=...` / `node_status_fn=...` kwargs
that used to be passed at Graph construction time.
"""
import contextvars
import functools
from functools import partial
from typing import Any, Callable

import ray

from dependency_graph.classes import Graph, Node
from hashing import hash_list_int

_TRACE: contextvars.ContextVar[Graph | None] = contextvars.ContextVar(
    "graph_trace", default=None
)


def tracing_active() -> bool:
    return _TRACE.get() is not None


def record(task: Callable, *args: Any) -> Node:
    """Record one op invocation as a graph node and return its handle.

    Handle (Node) args become dependency edges, in call order; the rest
    are literal `Node.args`. The runtime's `make_fn_args` appends
    dependency results after the literals, so ops must take their
    upstream inputs as their trailing parameters (place_image /
    merge_images / compute_image_metadata already do).
    """
    g = _TRACE.get()
    if g is None:
        raise RuntimeError("record() called outside trace()")

    literals = [a for a in args if not isinstance(a, Node)]
    deps = [a for a in args if isinstance(a, Node)]

    fn = getattr(task, "_function", task)
    name = str(hash_list_int([
        getattr(fn, "__module__", ""),
        getattr(fn, "__qualname__", getattr(fn, "__name__", "")),
        list(literals),
        [d.name for d in deps],
    ]))

    node = Node(task, *literals, name=name)
    g[node] = list(deps)
    return node


def trace(
    builder: Callable[..., Any],
    *args: Any,
    graph_update_fn=None,
    node_status_fn=None,
    **kwargs: Any,
) -> Graph | None:
    """Run `builder` under a recording context; return the accumulated
    `Graph` (or None if the builder returns None — e.g. a MASTER-mode
    missing panel). The builder uses `record(...)`; every recorded node
    (including the sink) is already a key of the returned graph.
    """
    g = Graph(graph_update_fn=graph_update_fn, node_status_fn=node_status_fn)
    token = _TRACE.set(g)
    try:
        result = builder(*args, **kwargs)
    finally:
        _TRACE.reset(token)
    if result is None:
        return None
    return g


def set_workflow_key(*key_parts: Any) -> None:
    """Inside a `@graph_builder` (or `@pipeline`) body, attach the
    standard workflow-progress callbacks to the active trace's `Graph`,
    keyed on `key_parts`. Replaces the older pattern of passing
    `graph_update_fn=partial(...)` and `node_status_fn=...` at Graph
    construction time.

    Mutates the active `Graph` in place — both attributes are mutable
    per `classes.py`.
    """
    g = _TRACE.get()
    if g is None:
        raise RuntimeError("set_workflow_key() called outside a trace")
    # Local import: workflow_graph_adapter imports from this module
    # transitively, so import at call time to avoid a cycle.
    from dependency_graph.workflow_graph_adapter import (
        store_node_status, store_workflow_graph,
    )
    g.graph_update_fn = partial(store_workflow_graph, tuple(key_parts))
    g.node_status_fn = store_node_status


def graph_builder(*decorator_args: Any, **ray_opts: Any):
    """Decorator: turn a plain-Python builder into a Ray-dispatched
    subgraph builder with automatic trace setup.

    Two usage forms:

        @graph_builder
        def build_X(...): ...

        @graph_builder(num_cpus=0.1)
        def build_X(...): ...

    The decorated function:

    * called from inside another active trace — records a `Node` handle
      so the outer trace sees this builder as a deferred dependency;
      the runtime later dispatches it via `.remote(...)` and splices the
      returned `Graph` into the running computation.
    * called via `.remote(*args)` (the runtime's dispatch path) — runs
      as a Ray task that sets `_TRACE` to a fresh `Graph`, executes the
      wrapped function, and returns the recorded `Graph` (or `None`
      when the body returns `None`, signalling "no graph for this
      input"; `GraphRuntime` already handles the None case).

    Calling the wrapper directly *outside* any trace raises — graph
    builders are not data ops and have nothing useful to do in that
    context; use `.remote(...)` or wrap the call in a `@pipeline`.

    Only positional args are supported in trace context (kwargs would
    need to be threaded through `record()`/`Node` deterministically,
    which today's identity scheme doesn't do; add that in a follow-on
    when it's actually needed).
    """
    # Detect parameter-less @graph_builder vs @graph_builder(num_cpus=…).
    if (
        len(decorator_args) == 1
        and callable(decorator_args[0])
        and not ray_opts
    ):
        return _make_graph_builder(decorator_args[0], ray_opts={})
    if decorator_args:
        raise TypeError(
            "@graph_builder takes only keyword arguments "
            "(Ray remote options); positional args are not supported."
        )

    def decorate(fn):
        return _make_graph_builder(fn, ray_opts=ray_opts)
    return decorate


def _make_graph_builder(fn: Callable, ray_opts: dict):
    """Build the @graph_builder wrapper object with both `__call__`
    (records in trace) and `.remote(...)` (Ray-dispatched, sets up the
    inner trace and returns the recorded Graph)."""

    # The Ray-task entry: sets up trace, runs the user's body, returns
    # the recorded Graph (or None for a "no-graph" signal).
    def _ray_entry(*args: Any, dry_run: bool = False, **kwargs: Any):
        # `dry_run` is meaningless for a graph builder — it doesn't
        # produce data, it produces a graph for the runtime to expand.
        # Just build the graph and let the runtime handle dispatch.
        del dry_run
        g = Graph()
        token = _TRACE.set(g)
        try:
            result = fn(*args, **kwargs)
        finally:
            _TRACE.reset(token)
        return None if result is None else g

    _ray_entry.__name__ = fn.__name__
    _ray_entry.__qualname__ = fn.__qualname__
    _ray_entry.__module__ = fn.__module__

    ray_func = (
        ray.remote(**ray_opts)(_ray_entry) if ray_opts
        else ray.remote(_ray_entry)
    )

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any):
        if _TRACE.get() is None:
            raise RuntimeError(
                f"@graph_builder {fn.__name__} called outside a trace. "
                "Either call from inside another @graph_builder/@pipeline, "
                "or use .remote(...) for Ray-dispatched execution."
            )
        if kwargs:
            raise NotImplementedError(
                f"@graph_builder {fn.__name__} called with kwargs inside a "
                f"trace ({sorted(kwargs)}); only positional args are "
                "supported in trace context."
            )
        return record(wrapper, *args)

    wrapper.remote = ray_func.remote
    wrapper.options = ray_func.options
    wrapper._function = fn
    wrapper._is_graph_builder = True
    return wrapper
