"""Delete an ephemeral node's on-disk output after its last consumer
across every pipeline has completed.

Invoked by the global `EphemeralFiles` registry
(`dependency_graph/ephemeral_registry.py`) when the registry's pending
consumer set for a filename empties — i.e. every Node across every
child runtime that declared the producer as a data/ordering dep has
succeeded. Failed/skipped consumers don't decrement (see A2 in the
plan): intermediates linger one extra run on failure for re-dispatch.

Cleanup rules — keyed on the master's file extension and the producer's
`_display` attribute:

  - `place_image` (master is `.tiff`, no `_display`)
        → unlink the `.tiff` + `.json`. The TIFF was never UI-relevant
          (no display tap), so reclaiming it is free.

  - `merge_images` / `merge_images_oss` (master is `.xisf`/`.fits`,
    `_display="Merged"`)
        → unlink master + `.json`. The `.tiff` sibling is the tile
          pyramid the UI serves; leave it alone.

  - TIFF master WITH `_display`
        → skip entirely. The TIFF IS both the master AND the tile
          pyramid; deleting it would 404 the UI.

  - `store_mosaic_color_channel_*` (`final=True`)
        → not ephemeral by default, so this function isn't called for
          them. If it ever IS called for one (user mistake), the
          safety guard in `_mark_done` skips terminal nodes anyway.

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

    has_display = hasattr(node.task, "_display")
    is_tiff = master.suffix.lower() in (".tif", ".tiff")

    if is_tiff and has_display:
        # TIFF master IS the tile pyramid the UI serves. Deleting it
        # would 404 the UI; leave it alone.
        return

    _safe("unlink master", lambda: master.unlink(missing_ok=True),
          node=node.name, path=master)
    _safe("unlink sidecar",
          lambda: sidecar_path_for(master).unlink(missing_ok=True),
          node=node.name, path=master)
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
