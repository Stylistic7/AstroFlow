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

# Stack of (group_name, group_id) pairs for the active lexical
# `@workflow_group` scope. Each entry corresponds to one
# @workflow_group-decorated function currently on the call stack;
# `record()` reads this and tags every new Node with the full path,
# which the workflow-graph serializer uses to build collapsible
# group nodes in the viewer.
_GROUP_STACK: contextvars.ContextVar[tuple] = contextvars.ContextVar(
    "workflow_group_stack", default=()
)


def tracing_active() -> bool:
    return _TRACE.get() is not None


def workflow_group(fn):
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
            cal_masters = []
            for cs in calibration_stacks_for(stack, config):
                cal_masters.append(stack_files(
                    downscale_stack(cs, 8), config, *cal_masters,
                ))
            return cal_masters

    Groups can nest — a helper decorated with `@workflow_group`
    that's called inside another `@workflow_group`-decorated
    helper inherits the outer group as a prefix on its Nodes'
    `_group_path`.

    The `group_id` is a deterministic hash of (function-name,
    parent_path, per-call key) so re-traces of the same lexical
    structure produce the same ids; the viewer's expanded/
    collapsed state stays stable across reruns."""
    name = fn.__name__

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


def record(task: Callable, *args: Any, _after: tuple = ()) -> Node:
    """Record one op invocation as a graph node and return its handle.

    Handle (Node) args become dependency edges, in call order; the rest
    are literal `Node.args`. The runtime's `make_fn_args` appends
    dependency results after the literals, so ops must take their
    upstream inputs as their trailing parameters (place_image /
    merge_images / compute_image_metadata already do).

    `_after` is a tuple of Node handles to add as *ordering-only*
    dependencies — the recorded node won't run until each `_after` node
    finishes, but their results are not appended to the call args. Use
    this for ops like `post_process_stack(stack, mock)` whose signature
    is fully covered by literals but which depend on the side-effect of
    an upstream node (the master file on disk). Including `_after` in
    the node-name hash ensures that two calls with identical data deps
    but different ordering deps don't dedup.

    When the constructed Node equals one already in the graph, the
    canonical key is returned (so downstream dep lists hold canonical
    objects) and its `_group_path` is cleared to `()` — a shared
    computation has no single canonical lexical scope. The render-
    time `duplicate_shared_into_consumer_scopes` transform (in
    `graph_transforms.py`) decides where to render shared Nodes
    by looking at their consumers' scopes.
    """
    g = _TRACE.get()
    if g is None:
        raise RuntimeError("record() called outside trace()")

    literals = [a for a in args if not isinstance(a, Node)]
    data_deps = [a for a in args if isinstance(a, Node)]
    ordering_deps = [a for a in _after if isinstance(a, Node)]
    deps = data_deps + ordering_deps

    fn = getattr(task, "_function", task)
    # Include `_after` in the name hash so two otherwise-identical calls
    # with different ordering deps don't dedup to one canonical Node.
    # The docstring claims this — but the original implementation only
    # hashed data deps' names. Sort `_after` names so the encoding is
    # order-independent (semantically `_after` is a SET).
    name = str(hash_list_int([
        getattr(fn, "__module__", ""),
        getattr(fn, "__qualname__", getattr(fn, "__name__", "")),
        list(literals),
        [d.name for d in data_deps],
        sorted(d.name for d in ordering_deps),
    ]))

    node = Node(task, *literals, name=name)
    if node in g:
        canonical = next(k for k in g if k == node)
        canonical._group_path = ()
        return canonical
    # `_group_path` tags the Node with the active `@workflow_group`
    # scope. Plain instance attribute — Node has no __slots__ and
    # __eq__/__hash__ ignore it.
    node._group_path = _GROUP_STACK.get()
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
    keyed on `key_parts`.

    `store_workflow_graph` fires on every node completion to update the
    viewer snapshot of the workflow DAG. The sweep CLI and cart→archive
    flow no longer rely on a per-cart-item path snapshot — they re-trace
    the pipeline on demand (see `dependency_graph/trace_paths.py`), which
    is bit-for-bit equivalent to the runtime under the pure-Merkle
    content_hash convention (every `@makes_image` output has
    `content_hash := producer.call_hash` stamped in its file header).

    Mutates the active `Graph` in place.
    """
    g = _TRACE.get()
    if g is None:
        raise RuntimeError("set_workflow_key() called outside a trace")
    # Local import: workflow_graph_adapter imports from this module
    # transitively, so import at call time to avoid a cycle.
    from dependency_graph.workflow_graph_adapter import (
        store_node_status, store_workflow_graph,
    )
    from functools import partial

    key = tuple(key_parts)
    g.graph_update_fn = partial(store_workflow_graph, key)
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
    #
    # IMPORTANT: this function must NOT reference `_TRACE` (a
    # ContextVar) directly. cloudpickle walks `__globals__` of any
    # function it serialises and tries to pickle every referenced
    # name; ContextVar instances are not picklable, which breaks Ray
    # dispatch (`.remote(...)`). Delegate to `trace()` instead -- it
    # references `_TRACE` from its OWN body, and cloudpickle pickles
    # `trace` itself by name/reference rather than by value.
    def _ray_entry(*args: Any, dry_run: bool = False, **kwargs: Any):
        # `dry_run` is meaningless for a graph builder — it doesn't
        # produce data, it produces a graph for the runtime to expand.
        del dry_run
        return trace(fn, *args, **kwargs)

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
        # `_after` is a magic kwarg for ordering-only dependencies; pop
        # it before checking for forbidden user-supplied kwargs.
        _after = kwargs.pop("_after", ())
        if kwargs:
            raise NotImplementedError(
                f"@graph_builder {fn.__name__} called with kwargs inside a "
                f"trace ({sorted(kwargs)}); only positional args are "
                "supported in trace context."
            )
        return record(wrapper, *args, _after=_after)

    wrapper.remote = ray_func.remote
    wrapper.options = ray_func.options
    wrapper._function = fn
    wrapper._is_graph_builder = True
    return wrapper
