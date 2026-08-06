"""Global orphan reconciliation for `context.output_dir` — the catch-all GC.

The live set is the union of:
  * every path produced by re-tracing each cart item's pipeline against the
    current context (`trace_paths.paths_for_cart_item`) — deterministic from the
    trace, no file reads, sub-second per item;
  * every path preserved in `context.archive`;
  * **every path any product_versions log references, unconditionally** — a
    log holds the current version plus, mid-rebuild, the last-good the UI is
    still showing. Deleting superseded versions is `version_log.supersede`'s
    job at node completion, never the sweep's.
  * every output of the IMPLICIT preview pipeline: `build_preview_graph`
    dispatches `Subframe` over every mosaic ("every mosaic is implicitly in
    the cart"), so its products — placed panels, merged channels, stored
    previews — are live even though no explicit cart entry names them.
    Without these the sweep deletes the whole subframe preview layer and
    every run deterministically rebuilds it.
  * plus each primary's `.json` sidecar and `<stem>_tiles.tiff` pyramid.

Anything under `output_dir` not in the live set is an orphan.

This is the extracted, callable core: `sweep(ctx, delete=...)` —
`scripts/sweep_orphans.py` is a thin CLI over it, and that CLI is the ONLY
caller. The app never runs a global delete unattended: the auto-run variant
deleted in-flight tail intermediates whenever the live-set trace resolved
PARTIALLY (`resolve_output_paths` skips unresolvable subtrees silently), which
re-churned the color tail every pass. `ctx` supplies `shopping_cart` /
`archive` / `output_dir` (pass `local_context`) — the version-log pin reads
the durable cache directly, so it needs no Ray.

Only a *global* disk-vs-trace scan can find orphans that nothing references —
crash/partial/failed-write leftovers, removed-cart-item trees, non-display
persistent intermediates, foreign cruft — run this by hand when the long tail
is worth reclaiming, ideally while the app is idle.
"""
from __future__ import annotations

import os
from pathlib import Path

from dependency_graph.trace_paths import paths_for_cart_item
from file_io.sidecar import sidecar_path_for


def _siblings_of(primary: str) -> list[Path]:
    """Sidecar + tile-pyramid paths derived from a primary file path (included
    unconditionally so a live primary's siblings are never swept)."""
    from previews.create_tile_pyramid import pyramid_path_for
    p = Path(primary)
    return [sidecar_path_for(p), pyramid_path_for(p)]


def _pinned_versions() -> set[str]:
    """EVERY file any product_versions log references, pinned unconditionally.

    Each log holds at most the current version plus (while a rebuild is in
    flight) the last-good the UI is still showing — deletion of superseded
    versions is `version_log.supersede`'s job at node completion, never the
    sweep's. Unconditional pinning makes the manual sweep safe to run at ANY
    time, including mid-rebuild: it can never take the last-good out from
    under the badge. Reads the durable logs directly (Ray-free)."""
    from context.cache import get_durable_cache
    dc = get_durable_cache()
    out: set[str] = set()
    for k in list(dc):
        if not (isinstance(k, tuple) and k and k[0] == "product_versions"):
            continue
        for e in dc.get(k) or []:
            out.add(e["path"])
            out.update(str(s) for s in _siblings_of(e["path"]))
    return out


def _implicit_preview_items(ctx) -> list[tuple[str, str, str]]:
    """The cart items `build_preview_graph` implies but never writes into
    `ctx.shopping_cart`: every mosaic runs the `Subframe` pipeline ("every
    mosaic is implicitly in the cart"). The live set must trace these too, or
    the sweep deletes the whole subframe preview layer every time it runs."""
    return [("mosaic", mosaic_id, "Subframe")
            for mosaic_id in (getattr(ctx, "mosaics", None) or {})]


def live_set(ctx) -> set[str]:
    """Union of cart-traced + implicit-preview + archive-preserved primaries
    (with siblings) and the versioning-pinned previous versions.

    Traces each item — explicit cart entries AND the implicit Subframe
    preview per mosaic — `strict=True`: if ANY item can't be resolved to its
    outputs (entity absent, pipeline unregistered, trace raised) this raises
    `TraceResolutionError` and `sweep` aborts BEFORE deleting anything — a
    partial live set would misclassify that item's live files as orphans. Fail
    loud rather than delete against an incomplete picture."""
    primaries: set[str] = set()
    for item in (ctx.shopping_cart or []):
        primaries.update(str(p) for p in paths_for_cart_item(tuple(item), strict=True))
    for item in _implicit_preview_items(ctx):
        primaries.update(str(p) for p in paths_for_cart_item(item, strict=True))
    for entry in (ctx.archive or []):
        if len(entry) >= 4:  # (object_type, object_id, pipeline, [paths])
            primaries.update(str(p) for p in entry[3])

    live: set[str] = set()
    for p in primaries:
        live.add(p)
        live.update(str(s) for s in _siblings_of(p))
    live |= _pinned_versions()
    return live


def walk_and_classify(
    output_dir: "str | Path | None", live: set[str],
    include: "set[str] | None" = None,
) -> "tuple[list[tuple[str, int]], dict[str, list[int]]]":
    """Single walk of `output_dir`: classify each file live/orphan and bucket by
    parent dir (rel to `output_dir`). Returns `(orphans, by_dir)` where
    `by_dir[rel] = [live_count, live_bytes, orph_count, orph_bytes]`. When
    `include` is non-empty, dirs not in it are skipped (not classified, not
    swept)."""
    orphans: list[tuple[str, int]] = []
    by_dir: dict[str, list[int]] = {}
    if output_dir is None:
        return orphans, by_dir
    base = str(output_dir).rstrip("/")
    for root, _, files in os.walk(base):
        if not files:
            continue
        rel = os.path.relpath(root, base)
        if include is not None and rel not in include:
            continue
        bucket = by_dir.setdefault(rel, [0, 0, 0, 0])
        for f in files:
            full = os.path.join(root, f)
            try:
                sz = os.path.getsize(full)
            except OSError:
                continue
            if full in live:
                bucket[0] += 1
                bucket[1] += sz
            else:
                bucket[2] += 1
                bucket[3] += sz
                orphans.append((full, sz))
    return orphans, by_dir


def sweep(ctx, delete: bool = False, include: "set[str] | None" = None) -> dict:
    """Compute orphans under `ctx.output_dir` and, when `delete`, unlink them.

    Returns a report `{orphans, by_dir, live_count, reclaimable_bytes,
    deleted, skipped_empty_live}`. The version-log pin keeps every logged
    version out of the orphan set, and `live_set` traces strict so an
    unresolvable cart item raises (never silently drops from the live set).
    MANUAL-ONLY, best run while the app is idle: a partially-resolvable trace
    (mid-rebuild) silently thins the live set, which is exactly why the
    auto-run variant was removed.

    Defensive floor: if the live set resolves EMPTY (empty cart + empty
    archive, or every cart item legitimately traced to no outputs), refuse to
    delete. Every file under `output_dir` would classify as an orphan, and
    "nothing is live" is a misconfiguration signal, never license to wipe the
    output tree. Callers get `skipped_empty_live=True` and zero deletions."""
    live = live_set(ctx)
    if not live:
        return {
            "orphans": [], "by_dir": {}, "live_count": 0,
            "reclaimable_bytes": 0, "deleted": 0, "skipped_empty_live": True,
        }
    orphans, by_dir = walk_and_classify(ctx.output_dir, live, include)
    deleted = 0
    if delete:
        for path, _ in orphans:
            try:
                os.unlink(path)
                deleted += 1
            except OSError:
                pass  # already gone / raced — fine
    return {
        "orphans": orphans,
        "by_dir": by_dir,
        "live_count": len(live),
        "reclaimable_bytes": sum(sz for _, sz in orphans),
        "deleted": deleted,
        "skipped_empty_live": False,
    }
