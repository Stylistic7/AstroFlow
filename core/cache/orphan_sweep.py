"""Global orphan reconciliation for `context.output_dir` — the catch-all GC.

The live set is the union of:
  * every path produced by re-tracing each cart item's pipeline against the
    current context (`trace_paths.paths_for_cart_item`) — deterministic from the
    trace, no file reads, sub-second per item;
  * every path preserved in `context.archive`;
  * **each display product's last-good (N-1) version, but ONLY while its newest
    version isn't built yet** — so a sweep mid-recompute can never delete a
    still-shown N-1 (graceful N/N-1 versioning). Once the newest lands, N-1 is
    collectible; the per-build prune deletes it proactively, this just prevents
    the race.
  * plus each primary's `.json` sidecar and `<stem>_tiles.tiff` pyramid.

Anything under `output_dir` not in the live set is an orphan.

This is the extracted, callable core: `sweep(ctx, delete=...)`. The app invokes
it automatically at pipeline-completion (`state_machine`); `scripts/sweep_orphans.py`
is a thin CLI over it. `ctx` supplies `shopping_cart` / `archive` / `output_dir`
(pass `local_context` from the CLI, the app's `context` from the app) — the
version-log pin reads the durable cache directly, so it needs no Ray.

Only a *global* disk-vs-trace scan can find orphans that nothing references —
crash/partial/failed-write leftovers, removed-cart-item trees, non-display
persistent intermediates, foreign cruft — so this reconciliation can't be
eliminated, only automated.
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


def _pinned_previous_versions() -> set[str]:
    """Each product's previous (last-good) version files, pinned ONLY while its
    newest recorded version isn't on disk yet. Reads the durable
    `product_versions` logs directly (Ray-free); see products/version_log.py."""
    from context.cache import get_durable_cache
    from catalog.queries import viewable_from_master
    dc = get_durable_cache()
    out: set[str] = set()
    for k in list(dc):
        if not (isinstance(k, tuple) and k and k[0] == "product_versions"):
            continue
        log = dc.get(k) or []
        if len(log) < 2:
            continue
        # log[-1] is the newest recorded (the current) version; while its file
        # isn't viewable yet, spare every older entry (the last-good the UI is
        # still showing). Once it's built, none of these are pinned → collectible.
        if viewable_from_master(log[-1].get("path")) is None:
            for e in log[:-1]:
                out.add(e["path"])
                out.update(str(s) for s in _siblings_of(e["path"]))
    return out


def prune_superseded_versions() -> int:
    """Cheap continuous cleanup (NO os.walk): for each display product whose
    CURRENT (newest logged) version is built, delete its older logged versions'
    files (+ siblings) and trim the log to just the current. Iterates the durable
    `product_versions` logs only — so it's fast enough to run on every build
    completion, keeping disk ~1× throughout a long run instead of accumulating
    N-1's until the end. The heavier global `sweep` (long-tail orphans) can then
    run less often.

    Safe by construction: a non-current entry is deleted ONLY once the current's
    file is confirmed viewable, so a still-shown N-1 is never touched. Returns
    `(files_unlinked, bytes_freed)` and logs one line per pruned product.
    Head-local (reads/writes the durable cache directly); intended to run from
    the head-pinned StateActor."""
    from context.cache import get_durable_cache
    from catalog.queries import viewable_from_master
    from products import version_log
    dc = get_durable_cache()
    deleted = 0
    freed = 0
    for k in list(dc):
        if not (isinstance(k, tuple) and k and k[0] == "product_versions"):
            continue
        log = dc.get(k) or []
        if len(log) < 2:
            continue
        current = log[-1]
        if viewable_from_master(current.get("path")) is None:
            continue  # current not built yet → keep the last-good
        n0, f0 = deleted, freed
        for e in log[:-1]:
            for p in (e["path"], *(str(s) for s in _siblings_of(e["path"]))):
                try:
                    sz = os.path.getsize(p)
                    os.unlink(p)
                    deleted += 1
                    freed += sz
                except OSError:
                    pass  # already gone — fine
        dc.set(k, [current])  # keep only the current version
        if deleted > n0:
            print(f"### version prune {version_log.describe(k)}: dropped "
                  f"{deleted - n0} file(s) ({(freed - f0) / 1e6:.0f} MB) — N-1 "
                  f"{log[-2]['call_hash'][:8]} superseded by {current['call_hash'][:8]}")
    return deleted, freed


def live_set(ctx) -> set[str]:
    """Union of cart-traced + archive-preserved primaries (with siblings) and
    the versioning-pinned previous versions.

    Traces each cart item `strict=True`: if ANY item can't be resolved to its
    outputs (entity absent, pipeline unregistered, trace raised) this raises
    `TraceResolutionError` and `sweep` aborts BEFORE deleting anything — a
    partial live set would misclassify that item's live files as orphans. Fail
    loud rather than delete against an incomplete picture."""
    primaries: set[str] = set()
    for item in (ctx.shopping_cart or []):
        primaries.update(str(p) for p in paths_for_cart_item(tuple(item), strict=True))
    for entry in (ctx.archive or []):
        if len(entry) >= 4:  # (object_type, object_id, pipeline, [paths])
            primaries.update(str(p) for p in entry[3])

    live: set[str] = set()
    for p in primaries:
        live.add(p)
        live.update(str(s) for s in _siblings_of(p))
    live |= _pinned_previous_versions()
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
    deleted, skipped_empty_live}`. Safe to run anytime: the version-log pin
    keeps a still-shown N-1 out of the orphan set, and `live_set` traces
    strict so an unresolvable cart item raises (never silently drops from the
    live set). Intended to be called at pipeline-completion/idle (nothing
    building) so the current trace is authoritative.

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
