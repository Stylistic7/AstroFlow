"""Delete an ephemeral node's on-disk output after its last consumer
across every pipeline has completed.

Invoked by the global `EphemeralFiles` registry
(`dependency_graph/ephemeral_registry.py`) when the registry's pending
consumer set for a filename empties — i.e. every Node across every
child runtime that declared the producer as a data/ordering dep has
succeeded. Failed/skipped consumers don't decrement (see A2 in the
plan): intermediates linger one extra run on failure for re-dispatch.

An image is an **atom**: the primary (xisf/fits, or a tiff for tiff-primary
ops like `place_image`), its tile-pyramid `.tiff` (the web-openable
co-representation, present only for `display=` ops), and the `.json`/`.context`
sidecars. They live and die together — so reclaim deletes the WHOLE atom. A
`display=` op that is ephemeral still built its TIFF; we delete it here along
with the primary, so the UI can never show an image whose openable primary is
gone. (`final=True` ops are exempt from ephemerality, so this is never called
for them.)

Always invalidate the `@cached.remote` entry afterwards so a future
re-trace re-runs the producer body instead of returning a `WithSidecar`
pointing at a missing file.

Every step is independently try/except-wrapped — cleanup is a side
effect, never a pipeline-failing operation.
"""
from pathlib import Path
from typing import Any

import structlog

from file_io.sidecar import sidecar_path_for
from file_io.types import WithSidecar


log = structlog.get_logger()


def cleanup_ephemeral_artifact(node, result: Any, fn_args: list) -> None:
    """Reclaim disk + invalidate cache for one ephemeral node's output.

    `fn_args` is the bound positional-argument list passed to the
    producer (`node.args + ancestor_results(node)`), needed to compute
    the cache key for invalidation.
    """
    master = _master_path(result)
    if master is None:
        return

    # Delete the whole atom: primary, its sidecar, the tile-pyramid TIFF (if
    # one was built — `display=` ops), and the TIFF's `.context` map. `.tiff`
    # may equal `master` (tiff-primary ops like place_image); unlink is
    # idempotent. Convention: pyramid = master.with_suffix(".tiff"), context
    # = pyramid.with_suffix(".context") (previews/create_tile_pyramid.py).
    tiff = master.with_suffix(".tiff")
    atom = [
        master,
        sidecar_path_for(master),
        tiff,
        tiff.with_suffix(".context"),
    ]
    for path in atom:
        _safe(f"unlink {path.suffix}",
              lambda p=path: p.unlink(missing_ok=True),
              node=node.name, path=path)

    invalidate = getattr(node.task, "invalidate", None)
    if invalidate is not None:
        _safe("cache invalidate", lambda: invalidate(*fn_args), node=node.name)


def _master_path(result: Any) -> Path | None:
    """Extract the on-disk master path from a producer's return value.

    Handles `WithSidecar` wrappers, bare Path subclasses, and any
    result type carrying a `.filename: Path` (e.g. `ImageMetadata`).
    Mirrors `_cached_path_missing` in `cached_decorator.py`.
    """
    target = result
    if isinstance(target, WithSidecar):
        target = target.path
    if isinstance(target, Path):
        return target
    inner = getattr(target, "filename", None)
    if isinstance(inner, Path):
        return inner
    return None


def _safe(label: str, fn, **logctx) -> None:
    """Run `fn()` and log any exception as a warning. Never re-raises —
    ephemeral cleanup is a side effect, never load-bearing."""
    try:
        fn()
    except Exception as e:
        log.warning("ephemeral " + label + " failed",
                    error=str(e), **{k: str(v) for k, v in logctx.items()})
