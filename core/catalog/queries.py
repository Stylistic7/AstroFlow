"""Queries over the catalog — the lazy product `(object × pipeline × tap × filter)`.

Every function here is a PURE READ derived from three sources the catalog does
not own: the pipeline registry (the pipeline names), the build-time trace (where
a product's file lands — a static function of content + config, filed per
ROUTED entity by `compute_graph._store_product_taps`), and disk (what's
actually there). It never writes; materializing products is the compute
graph's job, and the cart merely selects which products to materialize.

The whole point: "what can I show for this object?" is ONE lookup keyed by the
object's own `(type, stable_hash)` — a stack inside a mosaic, a target, a
calibration stack whose master was built inside some light stack's trace —
followed by asking disk what exists. No URL ancestry, no scan of other
objects' traces. See `products.product.Product`.
"""
import os
from pathlib import Path
from typing import Iterator

from context import local_context as context
from dependency_graph.trace_paths import product_taps_key
from pipelines.registry import registered_pipelines
from previews.create_tile_pyramid import needs_display_pyramid, pyramid_path_for
from products.product import Product, resolve_entity  # noqa: F401 — re-exported


# ── canonical display per object type (was duplicated in urls.js + useVersions.js
#    AND hardcoded in files.py) ────────────────────────────────────────────────
_DEFAULT_DISPLAY_BY_TYPE = {
    "mosaic": "Mosaic",
    "target": "Channel",
    "stack": "Master",
    "calibration_stack": "Master",
}


def default_display_for(object_type: str) -> str:
    return _DEFAULT_DISPLAY_BY_TYPE.get(object_type, "Master")


# ── pipeline preference for the default view (was the frontend PIPELINE_PREFERENCE).
#    App-supplied pipelines first — the reference baseline A/B variants compare
#    against; a future per-object pinned "winner" would front-run this. ─────────
_PIPELINE_PREFERENCE = (
    "PixInsight Master",
    "Hybrid: PI Master + OSS Mosaic",
    "PixInsight Downscaled",
    "Hybrid: PI Downscaled + OSS Mosaic",
    "Hybrid: OSS Stack + OSS Mosaic",
)


def _pipeline_rank(name: str) -> int:
    try:
        return _PIPELINE_PREFERENCE.index(name)
    except ValueError:
        return len(_PIPELINE_PREFERENCE)


_FILTER_ORDER = ("L", "H", "S", "O", "R", "G", "B")


def _filter_sort_key(f: str):
    return (_FILTER_ORDER.index(f) if f in _FILTER_ORDER else len(_FILTER_ORDER), f)


_MOSAIC_FILTERS_MEMO: dict[str, list[str]] = {}


def _mosaic_filters(mosaic) -> list[str]:
    cached = _MOSAIC_FILTERS_MEMO.get(mosaic.stable_hash)
    if cached is not None:
        return cached
    stacks = context.stacks or {}
    targets = context.targets or {}
    filters = set()
    for target_id in mosaic.target_ids:
        target = targets[target_id].get()
        for stack_id in target.stack_ids:
            stack_ref = stacks.get(stack_id)
            if stack_ref is None:
                continue
            filters.add(stack_ref.get().filter)
    out = sorted(filters, key=_filter_sort_key)
    _MOSAIC_FILTERS_MEMO[mosaic.stable_hash] = out
    return out


def filters_for(object_type: str, object_id: str) -> list[str]:
    """The filters this object's products span — the ONE ordered list the whole
    UI iterates (columns, channel selector, pills). Mosaic = union across its
    targets' stacks; target = its stacks' filters; stack/calibration = its own
    filter if any. One order (L,H,S,O,R,G,B, then the rest) everywhere, which
    kills the frontend/backend sort-order divergence."""
    entity = resolve_entity(object_type, object_id)
    if entity is None:
        return []
    if object_type == "mosaic":
        return _mosaic_filters(entity)
    if object_type == "target":
        stacks = context.stacks or {}
        return sorted(
            {stacks[sid].get().filter for sid in entity.stack_ids
             if sid in stacks and stacks[sid].get().filter},
            key=_filter_sort_key,
        )
    if object_type in ("stack", "calibration_stack"):
        return [entity.filter] if getattr(entity, "filter", None) else []
    return []


# ── the queries ──────────────────────────────────────────────────────────────
def product_taps(object_type: str, object_id: str, pipeline: str) -> list[dict]:
    """The display taps `pipeline`'s build-time trace routed to this object —
    `[{display, filter, path}, ...]`, existence-agnostic — or `[]` when the
    object is unknown or that pipeline never traced it. ONE durable-store read
    (`product_taps_key`, filed per routed entity by
    `compute_graph._store_product_taps`); never a trace on a request."""
    entity = resolve_entity(object_type, object_id)
    if entity is None:
        return []
    return context.get(product_taps_key(
        object_type, object_id, entity.content_hash, pipeline)) or []


def pipelines_for(object_type: str, object_id: str) -> list[str]:
    """Every registered pipeline whose trace routed a display tap to this
    object — a mosaic pipeline counts for the stacks, targets and calibration
    stacks inside it. Whether a product is actually on disk is `exists`."""
    names = sorted({name for (name, _cls) in registered_pipelines()})
    return [n for n in names if product_taps(object_type, object_id, n)]


def predict_path(product: Product) -> "Path | None":
    """The product's predicted on-disk primary path from the (cached) trace —
    existence-AGNOSTIC. None when the trace has no such tap for the object."""
    f = str(product.filter)
    for e in product_taps(product.object_type, product.object_id, product.pipeline):
        if e["display"] == product.display and str(e["filter"]) == f:
            return Path(e["path"])
    return None


def master_path(product: Product) -> "Path | None":
    """The product's PRIMARY file if it is on disk, else None — for downloads
    and opening in PixInsight, which want the deliverable itself rather than
    whatever serves tiles (`viewable_path`)."""
    p = predict_path(product)
    return p if p is not None and os.path.exists(p) else None


def viewable_from_master(master: "str | Path | None") -> "Path | None":
    """The path that will actually SERVE, given an already-resolved master path
    — no trace. Prefers the `<stem>_tiles.tiff` display pyramid (so it still
    renders after ephemeral cleanup reclaimed the primary master), falling back
    to the master only when it is itself a servable tiff. This is the ONE
    "exists = viewable" definition — the tile server gates on it, and now the
    dashboard/pill/default view do too (the master-vs-pyramid divergence).
    Callers holding the path already (e.g. from `product_taps`) use this so
    they don't re-trace per product."""
    if master is None:
        return None
    master = Path(str(master))
    tiff = pyramid_path_for(master)
    if tiff.exists():
        return tiff
    # THE PAYLOAD MAY BE ITS OWN PYRAMID (2026-08-17): display-tap TIFFs are
    # now written pyramidal, so there is no `_tiles.tiff` sibling to find. A
    # tiled multi-page TIFF serves exactly like one — same pages, same tiles —
    # because page 0 IS the master. See `products.registry._pyramid_is_the_payload`.
    from file_io.planar import is_tiled_tiff
    if is_tiled_tiff(master):
        return master
    # Fall back to the master ONLY when it is its own display artifact — an
    # already-display-ready uint8 tiff (the Subframe previews), which the tile
    # route's `uchar` branch rescales /255. A float master must NEVER be
    # served: it is one page, so the tile route's `page = maxZoom - z` goes
    # negative and pyvips (which warns "value -8 out of range for property
    # 'page'" instead of raising) silently serves FULL RESOLUTION at every
    # zoom; and it is written strip-based, so random tile access decodes
    # ~74 MB strips. That fallback is how a 19 GB `Linear RGB` master with a
    # half-built pyramid served 36,150 broken tiles on 2026-08-14, and how
    # `channelStats` once took 822 s. No pyramid yet => not viewable yet.
    if (str(master).endswith((".tiff", ".tif")) and os.path.exists(master)
            and not needs_display_pyramid(master)):
        return master
    # XISF masters serve DIRECTLY (2026-08-17): the tile route memory-maps the
    # pixel block and shrinks on the fly, so there is no pyramid to wait for and
    # none is written. That escapes both hazards in the comment above — the
    # mapped image is shrunk per zoom level rather than served full-res at every
    # zoom, and reads come from mapped pages rather than full-width strips.
    # 273.8 GB of XISF pyramids deleted on that basis; see file_io/xisf_mmap.
    if str(master).lower().endswith(".xisf") and os.path.exists(master):
        return master
    return None


def resolve_product(product: Product) -> "tuple[Path | None, str]":
    """The servable path for a product AND whether it's current — the version-
    aware resolver behind graceful N/N-1 versioning. Returns `(path, status)`:

      - `(path, "live")`        — the CURRENT trace's output is on disk;
      - `(path, "out_of_date")` — the current isn't built yet, but a prior
                                  version (from the version log) still is, so we
                                  serve that last-good and the UI badges it;
      - `(None, "absent")`      — nothing viewable (no current tap, or truly
                                  gone / never built).

    Fallback only kicks in when the current tap EXISTS in the trace but its file
    isn't built yet (a recompute in flight) — never when the object has no tap
    at all (e.g. removed from the cart), so a removed object never shows a stale
    image. Reuses `viewable_from_master` for the serve logic (pyramid-vs-master).
    """
    master = predict_path(product)
    cur = viewable_from_master(master)
    if cur is not None:
        # SNAPSHOT-LAG GUARD (2026-08-31): `predict_path` reads the build-time
        # tap snapshot; when a replan re-keys a product, the snapshot can lag
        # and this "live" is an OLD generation whose file happens to exist —
        # the boot-16 "Linear SHO current while the current file is absent"
        # confusion. The version log is append-ordered by COMPLETION, so a
        # newest entry whose call_hash differs from the predicted one means a
        # newer generation already finished: serve THAT, badged out_of_date
        # (the trace, not the log, stays the authority on "live").
        from products import version_log
        entries = version_log.versions(version_log.key_for(product))
        if entries:
            newest = entries[-1]
            if newest.get("call_hash") != version_log.call_hash_of(master):
                prev = viewable_from_master(newest.get("path"))
                if prev is not None:
                    return prev, "out_of_date"
        return cur, "live"
    if master is None:
        return None, "absent"  # no current tap in the trace — don't serve stale
    # Current tap exists but its output isn't built yet → serve the newest OTHER
    # logged version still on disk (the last-good).
    from products import version_log
    cur_hash = version_log.call_hash_of(master)
    for entry in reversed(version_log.versions(version_log.key_for(product))):
        if entry.get("call_hash") == cur_hash:
            continue  # this is the not-yet-built current
        prev = viewable_from_master(entry.get("path"))
        if prev is not None:
            return prev, "out_of_date"
    return None, "absent"


def viewable_path(product: Product) -> "Path | None":
    """The servable path for a product (path-only; last-good fallback included).
    Use `resolve_product` when you also need the live/out-of-date status."""
    return resolve_product(product)[0]


def exists(product: Product) -> bool:
    """Whether the product is on disk and viewable."""
    return viewable_path(product) is not None


def products_for(object_type: str, object_id: str) -> Iterator[Product]:
    """Every product in the catalog for this object: each pipeline that traced
    it × each of its display taps × filter. Lazy; existence is NOT checked here
    (use `existing_for`). Snapshot-only reads — never a trace on a request."""
    for pipeline in pipelines_for(object_type, object_id):
        for e in product_taps(object_type, object_id, pipeline):
            yield Product(object_type, object_id, pipeline,
                          e["display"], e["filter"])


def existing_for(object_type: str, object_id: str) -> Iterator[Product]:
    """The products for this object that are on disk and viewable — the viewer's
    slice of the catalog."""
    for product in products_for(object_type, object_id):
        if exists(product):
            yield product


def default_product_for(
    object_type: str, object_id: str, filter: "str | None" = None,
) -> "Product | None":
    """The product to show for a bare object URL: the highest-preference pipeline
    that actually produced a viewable tap, on the object type's default display
    where available, else any existing tap in preference order. Optionally pinned
    to one `filter`. None when nothing is on disk yet."""
    want = default_display_for(object_type)
    existing = [p for p in existing_for(object_type, object_id)
                if filter is None or str(p.filter) == str(filter)]
    if not existing:
        return None
    existing.sort(key=lambda p: (_pipeline_rank(p.pipeline),
                                 0 if p.display == want else 1))
    return existing[0]


def default_pipeline_for(object_type: str, object_id: str) -> str:
    """The pipeline name to open / report by default for a bare object URL: the
    highest-preference pipeline that produced a viewable tap, else the top
    preference. The fallback only bites when nothing is on disk yet — the caller
    then finds no file and reports empty — so this never invents a wrong answer,
    and `_PIPELINE_PREFERENCE` stays the single home for the ordering (no more
    `"PixInsight Master"` string literals scattered through the web backend)."""
    product = default_product_for(object_type, object_id)
    return product.pipeline if product else _PIPELINE_PREFERENCE[0]


def product_matrix(object_type: str, object_id: str,
                   with_paths: bool = False) -> dict | None:
    """Every pipeline whose trace routed a tap to this object, its display
    taps in order of computation, and per `(display, filter)` whether the
    output is on disk — `{filters, pipelines: [{name, taps: [{display,
    outputs: [{filter, exists, mtime, status}]}]}], default}`, plus `path`
    and `predicted_path` per output when `with_paths`. None for an unknown
    object. Snapshot-only: never traces.
    """
    if resolve_entity(object_type, object_id) is None:
        return None
    iter_filters = filters_for(object_type, object_id) or [None]
    entity_filter_strs = {str(f) for f in iter_filters}
    traces: dict[str, dict[str, dict[str, str]]] = {}
    extras: set[str] = set()
    for name in pipelines_for(object_type, object_id):
        by_display: dict[str, dict[str, str]] = {}
        for e in product_taps(object_type, object_id, name):
            fk = str(e["filter"])
            by_display.setdefault(e["display"], {})[fk] = e["path"]
            if fk not in entity_filter_strs:
                extras.add(fk)
        traces[name] = by_display
    columns = list(iter_filters) + sorted(extras)
    pipelines = []
    for name, by_display in traces.items():
        taps = []
        for display, paths_by_filter in by_display.items():
            outputs = []
            # Only the filters that actually route through this tap; a
            # filter that passes through but is not on disk yet is kept.
            for f in columns:
                if str(f) not in paths_by_filter:
                    continue
                product = Product(object_type, object_id, name, display, f)
                path, status = resolve_product(product)
                row = {
                    "filter": f,
                    "exists": path is not None,
                    "mtime": os.path.getmtime(str(path)) if path is not None else None,
                    "status": status,
                }
                if with_paths:
                    predicted = predict_path(product)
                    row["path"] = str(path) if path is not None else None
                    row["predicted_path"] = str(predicted) if predicted else None
                outputs.append(row)
            taps.append({"display": display, "outputs": outputs})
        pipelines.append({"name": name, "taps": taps})
    default = default_product_for(object_type, object_id)
    return {
        "filters": columns,
        "pipelines": pipelines,
        "default": default.as_view() if default else None,
    }
