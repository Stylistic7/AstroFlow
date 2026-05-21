"""User-facing pipeline / stage decorators.

* `@pipeline("Display Name")` — registers a function as a user-selectable
  pipeline. The decorated function is the trace boundary: it gets
  `@graph_builder` semantics (records ops called inside it as a Graph,
  Ray-dispatched via `.remote(...)` to produce that Graph for the
  runtime to splice). Display name is independent of filename and
  function name; users see exactly the string passed here.

* `@stage` — pipeline-local cacheable Ray-task op. Today this is just a
  re-export of `@node`, kept as a distinct name so we can fork the
  implementation later (e.g. pipeline-identity in the cache key)
  without touching every `@node` call site in the existing codebase.
"""
from typing import Any, Callable

from decorators.decorators import node
from dependency_graph.tracing import graph_builder
from pipelines.registry import _REGISTRY


def pipeline(display_name: str, **ray_opts: Any) -> Callable:
    """Decorator: register the wrapped function as a pipeline named
    `display_name`. Re-registration silently overrides (so a user file
    in `~/AstroFlow/pipelines/` can replace a built-in by reusing the
    name).

    `ray_opts` are forwarded to `@graph_builder` for the Ray-side
    `.remote(...)` dispatch — useful if a pipeline wants a specific
    `num_cpus` / scheduling strategy. Defaults to `num_cpus=0.1`,
    matching the existing top-level subgraph builders.
    """
    ray_opts.setdefault("num_cpus", 0.1)

    def decorate(fn: Callable) -> Callable:
        wrapped = graph_builder(**ray_opts)(fn)
        _REGISTRY[display_name] = wrapped
        return wrapped

    return decorate


# `@stage` is `@node` under a friendly name. Same call signature, same
# cache behaviour, same `.remote(...)`. Distinct alias so the future
# pipeline-identity refactor can fork @stage without disturbing the
# rest of the codebase.
stage = node
