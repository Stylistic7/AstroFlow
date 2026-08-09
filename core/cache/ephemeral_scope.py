"""Scoped default for the `_ephemeral` kwarg on `@node` calls.

User pipelines write:

    with ephemeral_intermediates():
        row = placed[0]
        for p in placed[1:]:
            row = merge_images_oss(mosaic, filter, row, p)

Every `@node` call inside the `with` block is recorded with
`_ephemeral=True`. The runtime then reclaims the on-disk master + sidecar
after the last in-graph consumer of each ephemeral node completes,
preserving only the tile-pyramid `.tiff` sibling (which the UI serves).

The default propagates through every nested function call — including
`@workflow_group` helpers — by design. Wrapping a pipeline body marks
everything derived inside it as transient.

A per-call override wins over the scope: `foo(..., _ephemeral=False)`
inside the block stays kept.
"""
from contextlib import contextmanager
from contextvars import ContextVar


_EPHEMERAL_SCOPE: ContextVar[bool] = ContextVar(
    "_EPHEMERAL_SCOPE", default=False,
)


@contextmanager
def ephemeral_intermediates():
    """NO-OP since 2026-08-08: the ephemeral-GC decision (see
    project_gc_preflight_removal_decision) — every image persists; the
    entire reclaim machinery is being deleted. This stays as an inert
    context manager so pipeline bodies don't change (op hashes hold).
    Phase 2 replaces it with scope-local dynamic-subgraph semantics
    (skip-if-terminals-cached; self-clean on conclusion)."""
    yield
