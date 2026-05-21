"""User-facing pipeline / stage decorators.

* `@pipeline("Display Name")` — registers a function as a user-selectable
  pipeline. The decorated function is a plain Python callable (NOT a
  Ray task): the cart-branch dispatcher in
  `compute_graph.build_main_processing_graph` calls it directly, on
  the same worker that's already running `build_main_processing_graph`
  itself, and splices the recorded `Graph` into the outer graph. So
  pipelines have the same execution shape as `build_compute_graph`:
  graph-building code, runs in-process, returns a `Graph`. There's no
  cloudpickle pass over the pipeline body — important because the
  trace machinery references a `ContextVar` that doesn't survive
  pickling.

* `@stage` — pipeline-local cacheable Ray-task op. Today this is just a
  re-export of `@node`, kept as a distinct name so we can fork the
  implementation later (e.g. pipeline-identity in the cache key)
  without touching every `@node` call site in the existing codebase.
"""
import functools
from typing import Callable

from decorators.decorators import node
from dependency_graph.tracing import trace
from pipelines.registry import _REGISTRY


def pipeline(display_name: str) -> Callable:
    """Decorator: register the wrapped function as a pipeline named
    `display_name`. Re-registration silently overrides (so a user file
    in `~/AstroFlow/pipelines/` can replace a built-in by reusing the
    name). The wrapped function, when called, runs the user's body
    under a fresh trace context and returns the recorded `Graph` (or
    `None` if the body returned `None`).
    """
    def decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def runner(*args, **kwargs):
            """Build and return the pipeline's `Graph` by running the
            user's body under a fresh trace context."""
            return trace(fn, *args, **kwargs)

        runner._function = fn
        runner._is_pipeline = True
        _REGISTRY[display_name] = runner
        return runner

    return decorate


# `@stage` is `@node` under a friendly name. Same call signature, same
# cache behaviour, same `.remote(...)`. Distinct alias so the future
# pipeline-identity refactor can fork @stage without disturbing the
# rest of the codebase.
stage = node
