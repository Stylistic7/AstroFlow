"""Define-by-run graph tracing (spike).

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
"""
import contextvars
from typing import Any, Callable

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
