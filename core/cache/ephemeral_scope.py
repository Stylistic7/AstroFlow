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
    """Mark every `@node` call inside this block as `_ephemeral=True`.

    See module docstring for full semantics.
    """
    token = _EPHEMERAL_SCOPE.set(True)
    try:
        yield
    finally:
        _EPHEMERAL_SCOPE.reset(token)
