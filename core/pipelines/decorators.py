"""User-facing pipeline / stage decorators.

* `@pipeline("Display Name", processes=Mosaic)` — registers a function
  as a user-selectable pipeline entry point. The decorated function
  is a plain Python callable (NOT a Ray task): the cart-branch
  dispatcher in `compute_graph.build_main_processing_graph` calls
  it directly, on the same worker that's already running
  `build_main_processing_graph`, and splices the recorded `Graph`
  into the outer graph. So pipelines have the same execution shape
  as `build_compute_graph`: graph-building code, runs in-process,
  returns a `Graph`. There's no cloudpickle pass over the pipeline
  body — important because the trace machinery references a
  `ContextVar` that doesn't survive pickling.

  The `processes=` arg is the entity class this entry point handles
  (`Stack`, `Target`, or `Mosaic`). One pipeline name can register
  multiple entry points, one per class; the cart-branch dispatcher
  looks up `(display_name, processes)` to pick the right one. A
  pipeline that only handles mosaics declares one entry point; one
  that supports queueing at any granularity declares three. Using
  the actual class (not a string) means a typo is a `NameError` at
  decoration time rather than a silent registry miss.

* `@stage` — pipeline-local cacheable Ray-task op. Today this is just a
  re-export of `@node`, kept as a distinct name so we can fork the
  implementation later (e.g. pipeline-identity in the cache key)
  without touching every `@node` call site in the existing codebase.
"""
import functools
from typing import Callable

from decorators.decorators import node
from dependency_graph.tracing import trace, set_workflow_key
from pipelines.registry import _REGISTRY


# URL-friendly tag for each entity class. Workflow-graph keys are
# stored as `(type_tag, id, pipeline_name)`, mirroring the cart-entry
# tuple shape, so the web server's `/api/workflow_graph/<type>/<id>/<pipeline>`
# routes look up cleanly.
_TYPE_TAG = {
    "Stack": "stack",
    "Target": "target",
    "Mosaic": "mosaic",
}


def _snapshot_workflow_graph(key: tuple, graph) -> None:
    """Build a topology-only workflow graph from `graph` and store
    it under `key`. Uses a stub `GraphRuntime` with no recorded
    results, so every node shows as 'NO_STATE_RUNTIME' — the viewer
    renders the structure without per-node state.

    Identical storage shape to the runtime-driven `store_workflow_
    graph`, so the web server's
    `/api/workflow_graph/<type>/<id>/<pipeline>` endpoints don't
    have to special-case anything.
    """
    from dependency_graph.graph_runtime import GraphRuntime
    from dependency_graph.workflow_graph_adapter import store_workflow_graph
    rt = GraphRuntime(graph)
    store_workflow_graph(key, rt)


def _filter_of(node) -> str | None:
    """Inspect a Node's literal args and return the value bound to
    the `filter` parameter (if any). Used to group sinks by filter
    when slicing a pipeline's graph into per-channel views."""
    import inspect
    fn = getattr(node.task, "_function", node.task)
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    params = list(sig.parameters.keys())
    arg_map = {params[i]: node.args[i]
               for i in range(min(len(node.args), len(params)))}
    return arg_map.get("filter")


def _ancestor_closure(roots, graph) -> set:
    """Set of every node reachable from any node in `roots` by
    walking dep edges (graph[node] is its deps)."""
    seen = set()
    stack = list(roots)
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        for dep in graph.get(n, []):
            stack.append(dep)
    return seen


def _snapshot_per_filter_workflow_graphs(base_key: tuple, graph) -> None:
    """Slice `graph` into per-filter subgraphs and snapshot each
    under `(*base_key, filter)`.

    For mosaic pipelines (and any pipeline whose sinks carry a
    `filter` literal), a sink is a node with no descendants. We
    group sinks by filter, compute each filter's ancestor closure,
    and snapshot it as its own workflow graph. The per-(pipeline,
    filter) viewer page then shows only the work that produces
    that one channel."""
    from dependency_graph import Graph

    has_deps = set()
    for deps in graph.values():
        for d in deps:
            has_deps.add(d)
    sinks = [n for n in graph if n not in has_deps]

    sinks_by_filter: dict[str, list] = {}
    for s in sinks:
        f = _filter_of(s)
        if f is None:
            continue
        sinks_by_filter.setdefault(str(f), []).append(s)

    for filter_key, filter_sinks in sinks_by_filter.items():
        keep = _ancestor_closure(filter_sinks, graph)
        sub = Graph({n: list(graph.get(n, [])) for n in keep})
        _snapshot_workflow_graph((*base_key, filter_key), sub)


def pipeline(display_name: str, processes: type) -> Callable:
    """Decorator: register the wrapped function as an entry point for
    pipeline `display_name`, handling cart entries whose object class
    is `processes`. Re-registration silently overrides (so a user
    file in `~/AstroFlow/pipelines/` can replace a built-in by reusing
    the (name, class) pair). The wrapped function, when called, runs
    the user's body under a fresh trace context and returns the
    recorded `Graph` (or `None` if the body returned `None`).
    """
    def decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def runner(*args, **kwargs):
            """Build and return the pipeline's `Graph` by running the
            user's body under a fresh trace context.

            After the trace completes, snapshot a topology-only
            workflow graph and store it under
            `(type_tag, entity.stable_hash, pipeline_name)` so the
            stack / target / mosaic viewers can render the pipeline's
            structure without waiting for runtime state.

            State updates are a follow-up: today the dispatcher
            splices each pipeline's graph into the main compute
            graph rather than dispatching it as a child runtime, so
            the per-pipeline `graph_update_fn` registered by
            `set_workflow_key` doesn't fire at runtime. The topology
            snapshot at least gives the viewer something to render.
            """
            type_tag = _TYPE_TAG.get(processes.__name__)
            entity = args[0] if args else None
            entity_id = getattr(entity, "stable_hash", None)

            def keyed_fn(*a, **kw):
                if type_tag and entity_id:
                    set_workflow_key(type_tag, entity_id, display_name)
                return fn(*a, **kw)

            graph = trace(keyed_fn, *args, **kwargs)
            if graph is not None and type_tag and entity_id:
                base_key = (type_tag, entity_id, display_name)
                _snapshot_workflow_graph(base_key, graph)
                _snapshot_per_filter_workflow_graphs(base_key, graph)
            return graph

        runner._function = fn
        runner._is_pipeline = True
        runner._pipeline_name = display_name
        runner._pipeline_processes = processes
        _REGISTRY[(display_name, processes)] = runner
        return runner

    return decorate


# `@stage` is `@node` under a friendly name. Same call signature, same
# cache behaviour, same `.remote(...)`. Distinct alias so the future
# pipeline-identity refactor can fork @stage without disturbing the
# rest of the codebase.
stage = node
