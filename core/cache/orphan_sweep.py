"""Global orphan reconciliation for `context.output_dir` — the catch-all GC.

THE TRACE IS THE AUTHORITY (2026-08-17). The live set is the union of:
  * **every path traced for every cart item** — the cart is the declaration
    of what should exist on disk; what is not in it is fair game (Stylistic7,
    2026-11-05). Done with an object: export the picture, take it out.
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

This is the callable core: `sweep(ctx, delete=...)`. The app runs it every
pass (`compute_graph.sweep_orphans`): after every cart entry is traced, so
the live set is those traces' snapshots, and before any pipeline runs.
`scripts/sweep_orphans.py` is the dry-run report and a manual run. The
auto-run variant of 2026-08 deleted in-flight tail intermediates because the
live-set trace resolved PARTIALLY; the trace is strict now, and what a
running node writes is spared by its age (`walk_and_classify`). `ctx`
supplies `shopping_cart` / `mosaics` / `output_dir`.
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

    The cart is the declaration of what should exist on disk, and it has two
    sources — both traced identically:

      * the shopping cart;
      * the IMPLICIT per-mosaic pipelines above.
    """
    items: set[tuple] = set()
    for entry in (ctx.shopping_cart or []):
        if entry is None:
            continue
        t = tuple(entry)
        if len(t) >= 3:
            items.add(tuple(str(x) for x in t[:3]))
    for mosaic_id in (ctx.mosaics or {}):
        for pipeline in _IMPLICIT_MOSAIC_PIPELINES:
            items.add(("mosaic", str(mosaic_id), pipeline))
    return sorted(items)


def item_paths(ctx) -> "dict[tuple, list[str]]":
    """Every cart item's traced output paths, item by item.

    Traced STRICTLY: an unresolvable cart item raises `TraceResolutionError`
    rather than silently dropping out — a dropped item would condemn all of
    its outputs as orphans. That refusal is the one guard that matters now the
    census is gone: the sweep either knows the whole live set or does nothing.
    """
    from dependency_graph.trace_paths import traced_paths_for

    return {item: [str(e["path"]) for e in traced_paths_for(item, recompute=True, strict=True) if e.get("path")]
            for item in _cart_items(ctx)}


def live_set(ctx, paths: "dict[tuple, list[str]] | None" = None) -> set[str]:
    """Every path the trace says should exist (`paths`: `item_paths`, traced
    here when not given), each primary's siblings, plus version-log pins."""
    primaries = {p for ps in (item_paths(ctx) if paths is None else paths).values() for p in ps}
    live: set[str] = set()
    for p in primaries:
        live.add(p)
        live.update(str(s) for s in _siblings_of(p))
    live |= _pinned_versions()
    return live


def footprints(paths: "dict[tuple, list[str]]") -> "dict[str, int]":
    """Bytes on disk per cart item, keyed "type/id/pipeline": what removing
    the entry frees on the next pass. A path two entries share counts under
    each — the number is an estimate for a person, not a ledger."""
    out = {}
    for item, ps in paths.items():
        total = 0
        for p in ps:
            for q in (p, *(str(s) for s in _siblings_of(p))):
                try:
                    total += os.path.getsize(q)
                except OSError:
                    pass
        out["/".join(item)] = total
    return out


# Op directories the IDENTIFICATION graph fills, which no pipeline trace names:
# per-run dust measurements (`stacks/dust_runs.measure_dust_run`, the input of
# `dust_epochs`) and star flats (`stacks/star_flat.build_star_flat`). The live
# set is the pipeline traces, so these classified as 100% orphan, were swept,
# and the next pass re-measured every run — 5,290 of them, over an hour, twice
# in one afternoon (2026-10-28). Never walked until the identification graph
# is traced for its paths the way the cart is.
IDENTIFICATION_OPS = frozenset({"measure_dust_run", "build_star_flat"})


CLOCK_MARGIN_S = 300.0     # the worker's files arrive over SMB with the worker's clock


def session_start(session_dir: "str | None" = None) -> float:
    """When the Ray session this process belongs to started, as an epoch, from
    the session directory's name (`session_YYYY-MM-DD_HH-MM-SS_<us>_<pid>`,
    local time): the app's boot, which a sweep spares everything newer than."""
    import datetime
    import re

    if session_dir is None:
        import ray._private.worker as worker
        session_dir = worker.global_worker.node.get_session_dir_path()
    m = re.match(r"session_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_", os.path.basename(session_dir))
    if not m:
        raise RuntimeError(f"cannot tell when the app booted from {session_dir!r}")
    return datetime.datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S").timestamp()


def walk_and_classify(
    output_dir: "str | Path | None", live: set[str],
    include: "set[str] | None" = None, spare_since: "float | None" = None,
) -> "tuple[list[tuple[str, int]], dict[str, list[int]], list[tuple[str, int]]]":
    """Single walk of `output_dir`: classify each file live/orphan and bucket by
    parent dir (rel to `output_dir`). Returns `(orphans, by_dir, spared)` where
    `by_dir[rel] = [live_count, live_bytes, orph_count, orph_bytes]`. When
    `include` is non-empty, dirs not in it are skipped (not classified, not
    swept); `IDENTIFICATION_OPS` dirs are always skipped.

    An untraced file modified at or after `spare_since` (an epoch) is in
    flight, not an orphan: a running node's scratch and half-written outputs
    are in no trace, and a crash leftover is older than the app that is
    running now. Such files go to `spared`, counted nowhere else."""
    orphans: list[tuple[str, int]] = []
    spared: list[tuple[str, int]] = []
    by_dir: dict[str, list[int]] = {}
    if output_dir is None:
        return orphans, by_dir, spared
    base = str(output_dir).rstrip("/")
    for root, _, files in os.walk(base):
        if not files:
            continue
        rel = os.path.relpath(root, base)
        if rel in IDENTIFICATION_OPS or (include is not None and rel not in include):
            continue
        bucket = by_dir.setdefault(rel, [0, 0, 0, 0])
        for f in files:
            full = os.path.join(root, f)
            try:
                st = os.stat(full)
            except OSError:
                continue
            if full in live:
                bucket[0] += 1
                bucket[1] += st.st_size
            elif spare_since is not None and st.st_mtime >= spare_since:
                spared.append((full, st.st_size))
            else:
                bucket[2] += 1
                bucket[3] += st.st_size
                orphans.append((full, st.st_size))
    return orphans, by_dir, spared


def sweep(ctx, delete: bool = False, include: "set[str] | None" = None,
          spare_since: "float | None" = None) -> dict:
    """Compute orphans under `ctx.output_dir` and, when `delete`, unlink them.

    Returns a report `{orphans, by_dir, live_count, reclaimable_bytes,
    deleted, skipped_empty_live, spared, scope_gc_leaks}`. The version-log pin
    keeps every logged version out of the orphan set, and `live_set` traces
    strict so an unresolvable cart item raises (never silently drops from the
    live set). Safe while the graph runs when `spare_since` is the app's boot:
    every traced output is live before it exists, and what a running node
    writes untraced — scratch, a partial file — is newer than the boot and
    spared (`walk_and_classify`); a later sweep takes it if it is still an
    orphan then.

    Defensive floor: if the live set resolves EMPTY (empty cart + empty
    archive, or every cart item legitimately traced to no outputs), refuse to
    delete. Every file under `output_dir` would classify as an orphan, and
    "nothing is live" is a misconfiguration signal, never license to wipe the
    output tree. Callers get `skipped_empty_live=True` and zero deletions."""
    paths = item_paths(ctx)
    live = live_set(ctx, paths)
    if not live:
        return {
            "orphans": [], "by_dir": {}, "live_count": 0,
            "reclaimable_bytes": 0, "deleted": 0, "skipped_empty_live": True,
            "spared": [], "scope_gc_leaks": {}, "footprints": footprints(paths),
        }
    orphans, by_dir, spared = walk_and_classify(ctx.output_dir, live, include, spare_since)
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
        "spared": spared,
        "scope_gc_leaks": leaks,
        "footprints": footprints(paths),
    }


# Ops whose outputs live INSIDE a `delete_intermediates` scope. The scope is
# responsible for reclaiming them when its terminal completes, so one showing up
# as an orphan means that GC did not run — a bug to fix, not litter to sweep.
# (Measured 2026-08-17: 458 leaked `place_image` files from runs predating the
# 2026-08-11 scope GC. `crop_panel` was clean, which is what proves the
# machinery works when it runs.)
SCOPE_INTERIOR_OPS = frozenset({"place_image", "crop_panel", "autostretch_panel"})


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
