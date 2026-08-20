"""Global orphan reconciliation for `context.output_dir` — the catch-all GC.

THE TRACE IS THE AUTHORITY (2026-08-17). The live set is the union of:
  * **every path traced for every cart item, including archived ones** — the
    cart is the declaration of what should exist on disk. An archived item is
    traced exactly like a live one; the only difference is that it is not added
    to the compute graph that runs, so nothing recomputes it.
  * **every path any product_versions log references, unconditionally** — the
    log is written only at landings, so its entries are real files.
  * plus each primary's `.json` sidecar and `<stem>_tiles.tiff` pyramid.

Anything under `output_dir` not in the live set is an orphan.

WHY THE RUNTIME CENSUS IS GONE. It recorded what the last COMPLETE pass landed,
and was introduced because early trace predictions "diverged from runtime
landings (phantoms pinned, real files condemned)". But those are two different
failure modes, and only one is dangerous:

  * a phantom pin is BENIGN — the sweep just declines to delete something;
  * condemning a real file is FATAL — and that came from the trace
    UNDER-covering, not from prediction being wrong in principle.

A trace-based live set OVER-pins (it names files that do not exist yet), which
is the safe direction. The census UNDER-pins (it knows only one pass), which is
why it needed a completeness guard — and that guard deadlocked: a failed run
stores no census, so the sweep spared everything and freed nothing, exactly when
space was needed. Worse, an intermediate the scope GC failed to reclaim was
recorded as "landed" and thereby pinned live FOREVER (measured: 458 leaked
`place_image` files, 907 MB, unsweepable by construction).

Measured before the switch, tracing every cart item: 2,717 live paths, 0 trace
failures, and every file on disk accounted for except genuinely superseded
products and the leaked interiors above.

SCOPE INTERIORS. A `delete_intermediates` subgraph is responsible for cleaning
up after itself; its interiors are deliberately absent from the trace. So an
interior appearing as an orphan here is a SCOPE GC BUG, not routine litter, and
`sweep` reports it as such rather than quietly reclaiming it.

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

from dependency_graph.trace_paths import TraceResolutionError
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


#: Dispatched over every mosaic by `compute_graph.build_preview_graph`, whose
#: own comment states the semantics: "every mosaic is implicitly in the cart
#: with the Subframe pipeline". It writes real files (`merge_images_subframe`,
#: `store_mosaic_color_channel_subframe` both publish through `filename_fn`), so
#: the live set must include it or the sweep reclaims ~2 GB of previews that
#: nothing asked to delete.
_IMPLICIT_MOSAIC_PIPELINES = ("Subframe",)


def _cart_items(ctx) -> "list[tuple]":
    """Every `(object_type, object_id, pipeline)` whose outputs must survive.

    The cart is the declaration of what should exist on disk, and it has three
    sources — all traced identically:

      * the shopping cart;
      * the ARCHIVE. Archived means "don't recompute me", never "you may delete
        my images", so an archived item pins its outputs exactly like a live
        one. That is the whole difference: archive entries stay out of the
        compute graph that runs, and stay in the live set.
      * the IMPLICIT per-mosaic pipelines above.
    """
    items: set[tuple] = set()
    for src in (ctx.shopping_cart or [], ctx.archive or []):
        for entry in src:
            if entry is None:
                continue
            t = tuple(entry)
            if len(t) >= 3:
                items.add(tuple(str(x) for x in t[:3]))
    for mosaic_id in (ctx.mosaics or {}):
        for pipeline in _IMPLICIT_MOSAIC_PIPELINES:
            items.add(("mosaic", str(mosaic_id), pipeline))
    return sorted(items)


def live_set(ctx) -> set[str]:
    """Every path the trace says should exist, plus version-log pins.

    Traced STRICTLY: an unresolvable cart item raises `TraceResolutionError`
    rather than silently dropping out of the live set — a dropped item would
    condemn all of its outputs as orphans. That refusal is the one guard that
    matters now the census is gone, and it is the reason this is safe to run at
    any time: the sweep either knows the whole live set or does nothing.
    """
    from dependency_graph.trace_paths import traced_paths_for

    primaries: set[str] = set()
    for item in _cart_items(ctx):
        # strict=True: raises TraceResolutionError rather than returning [].
        for entry in traced_paths_for(item, strict=True):
            p = entry.get("path")
            if p:
                primaries.add(str(p))

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
    swept).

    `include` limits the walk to named op dirs; everything else is untouched."""
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
            is_live = full in live
            if is_live:
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
            "scope_gc_leaks": {},
        }
    orphans, by_dir = walk_and_classify(ctx.output_dir, live, include)
    leaks = scope_gc_leaks(orphans)
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
        "scope_gc_leaks": leaks,
    }


# Ops whose outputs live INSIDE a `delete_intermediates` scope. The scope is
# responsible for reclaiming them when its terminal completes, so one showing up
# as an orphan means that GC did not run — a bug to fix, not litter to sweep.
# (Measured 2026-08-17: 458 leaked `place_image` files from runs predating the
# 2026-08-11 scope GC. `crop_panel` was clean, which is what proves the
# machinery works when it runs.)
SCOPE_INTERIOR_OPS = frozenset({"place_image", "crop_panel", "apply_background_level"})


def scope_gc_leaks(orphans: "list[tuple[str, int]]") -> "dict[str, list[int]]":
    """`{op: [count, bytes]}` for orphans that a scope GC should already have
    reclaimed. Reported separately so a GC regression is visible rather than
    silently absorbed by the sweep."""
    out: dict[str, list[int]] = {}
    for path, sz in orphans:
        op = os.path.basename(os.path.dirname(path))
        if op in SCOPE_INTERIOR_OPS:
            b = out.setdefault(op, [0, 0])
            b[0] += 1
            b[1] += sz
    return out
