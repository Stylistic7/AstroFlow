"""Trace-driven discard of cached pipeline results.

Given a `(object_type, object_id, pipeline)` cart item, `group_ops` buckets its
traced output nodes by op (for the dashboard's checkbox list), and `discard_ops`
deletes the selected ops' on-disk atoms + diskcache entries so the pipeline
re-runs them. Pure helpers around `traced_paths_for`; the route layer adds the
snapshot invalidation + `state.drain_and_restart`.

Filenames are content-addressed `call_hash`es, not the entity's `stable_hash`,
so the trace is the only thing that knows which files belong to a given mosaic's
pipeline — hence everything keys off `traced_paths_for`, not a filesystem glob.
"""
from __future__ import annotations

from pathlib import Path

from context import get_cache
from file_io.sidecar import sidecar_path_for
from dependency_graph.trace_paths import traced_paths_for


def _atom_paths(primary: Path) -> list[Path]:
    """The full on-disk atom for one produced file: primary, its `.json`
    sidecar, the display tile-pyramid `_tiles.tiff`, and the pyramid's
    `.context` map. Mirrors `cache/ephemeral_cleanup.reclaim_artifact`."""
    from previews.create_tile_pyramid import pyramid_path_for
    tiff = pyramid_path_for(primary)
    return [primary, sidecar_path_for(primary), tiff, tiff.with_suffix(".context")]


def group_ops(item: tuple[str, str, str]) -> list[dict]:
    """Every op in `item`'s traced pipeline, grouped by `op_name`, with on-disk
    counts + reclaimable bytes (primary + its tile pyramid). Order follows the
    trace (≈topological) so the user reads top-to-bottom dependency order."""
    groups: dict[str, dict] = {}
    for e in traced_paths_for(item):
        op = e["op_name"]
        g = groups.get(op)
        if g is None:
            g = groups[op] = {
                "op_name": op,
                "display": e.get("display"),
                "is_final": bool(e.get("is_final")),
                "count": 0,
                "on_disk": 0,
                "bytes": 0,
            }
        g["count"] += 1
        primary = Path(e["path"])
        if primary.exists():
            g["on_disk"] += 1
            from previews.create_tile_pyramid import pyramid_path_for
            for p in (primary, pyramid_path_for(primary)):
                try:
                    g["bytes"] += p.stat().st_size
                except OSError:
                    pass
    return list(groups.values())


def discard_ops(item: tuple[str, str, str], op_names: set[str]) -> dict:
    """Delete every traced node's atom + diskcache entry for the selected ops
    in `item`'s pipeline. Returns counts. The caller invalidates the tap
    snapshot and triggers the recompute.

    Deleting the file alone already forces a re-run (the cache's stale-miss
    check, `cached_decorator._cached_path_missing`); the `del cache[...]` just
    reclaims the diskcache entry. The cache key is `(op_name, call_hash)` and
    the call_hash IS the file stem, so no op-arg reconstruction is needed."""
    cache = get_cache()
    files_deleted = 0
    cache_deleted = 0
    for e in traced_paths_for(item):
        if e["op_name"] not in op_names:
            continue
        primary = Path(e["path"])
        for p in _atom_paths(primary):
            try:
                p.unlink()
                files_deleted += 1
            except FileNotFoundError:
                pass
            except OSError:
                pass
        try:
            del cache[(e["op_name"], primary.stem)]
            cache_deleted += 1
        except KeyError:
            pass
    return {"files_deleted": files_deleted, "cache_deleted": cache_deleted}
