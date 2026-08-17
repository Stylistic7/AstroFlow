"""Queries over the catalog — the lazy product `(object × pipeline × tap × filter)`.

Every function here is a PURE READ derived from three sources the catalog does
not own: the pipeline registry (which pipelines process which object class), the
trace (where a product's file lands — a static function of content + config),
and disk (what's actually there). It never writes; materializing products is the
compute graph's job, and the cart merely selects which products to materialize.

The whole point: "what can I show for this object?" is answered by *slicing the
product by the object and asking disk what exists* — not by reconstructing
engagement from the cart or snapshots. See `products.product.Product`.
"""
import os
from pathlib import Path
from typing import Iterator

from context import local_context as context
from dependency_graph.trace_paths import tap_outputs_for, tap_path_for
from pipelines.registry import registered_pipelines
from previews.create_tile_pyramid import needs_display_pyramid, pyramid_path_for
from products.product import Product, object_class

# A child's cart-item ancestor chain (root→leaf `(object_type, object_id)`),
# passed down from the URL hierarchy so a leaf without its own tap snapshot
# resolves from an ancestor's. `()` for a cart item viewed directly.
Ancestors = tuple[tuple[str, str], ...]


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


# ── entity + filters (the canonical home; web/backend re-points here in step 2) ─
def resolve_entity(object_type: str, object_id: str):
    """The in-context entity (Stack / CalibrationStack / Target / Mosaic) for an
    `(object_type, object_id)`, or None if not present."""
    if object_type == "stack":
        s = (context.stacks or {}).get(object_id)
        return s.get() if s else None
    if object_type == "calibration_stack":
        return (context.calibration_stacks or {}).get(object_id)
    if object_type == "target":
        t = (context.targets or {}).get(object_id)
        return t.get() if t else None
    if object_type == "mosaic":
        m = (context.mosaics or {}).get(object_id)
        return m.get() if m else None
    return None


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
def pipelines_for(object_type: str, object_id: str) -> list[str]:
    """Every registered pipeline that PROCESSES this object's class (has a
    `for_<type>` entry point) — the class-slice of the product. Deliberately NOT
    scoped to the cart/snapshots: whether a product was actually computed is
    answered by `exists`, never by engagement."""
    cls = object_class(object_type)
    if cls is None:
        return []
    return sorted({name for (name, c) in registered_pipelines() if c == cls})


def predict_path(product: Product, ancestors: Ancestors = ()) -> "Path | None":
    """The product's predicted on-disk primary path from the (cached) trace —
    existence-AGNOSTIC. None when the trace has no such tap for the object.
    `ancestors` resolves a child leaf from its cart-item ancestor's snapshot."""
    return tap_path_for(
        (product.object_type, product.object_id, product.pipeline),
        product.display, product.filter, ancestors=ancestors,
    )


def viewable_from_master(master: "str | Path | None") -> "Path | None":
    """The path that will actually SERVE, given an already-resolved master path
    — no trace. Prefers the `<stem>_tiles.tiff` display pyramid (so it still
    renders after ephemeral cleanup reclaimed the primary master), falling back
    to the master only when it is itself a servable tiff. This is the ONE
    "exists = viewable" definition — the tile server gates on it, and now the
    dashboard/pill/default view do too (the master-vs-pyramid divergence).
    Callers holding the path already (e.g. from `tap_outputs_for`) use this so
    they don't re-trace per product."""
    if master is None:
        return None
    master = Path(str(master))
    tiff = pyramid_path_for(master)
    if tiff.exists():
        return tiff
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


def resolve_product(
    product: Product, ancestors: Ancestors = (),
) -> "tuple[Path | None, str]":
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
    master = predict_path(product, ancestors)
    cur = viewable_from_master(master)
    if cur is not None:
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


def viewable_path(product: Product, ancestors: Ancestors = ()) -> "Path | None":
    """The servable path for a product (path-only; last-good fallback included).
    Use `resolve_product` when you also need the live/out-of-date status."""
    return resolve_product(product, ancestors)[0]


def exists(product: Product, ancestors: Ancestors = ()) -> bool:
    """Whether the product is on disk and viewable."""
    return viewable_path(product, ancestors) is not None


def products_for(
    object_type: str, object_id: str, ancestors: Ancestors = (),
) -> Iterator[Product]:
    """Every product in the catalog for this object — the class-slice: each
    applicable pipeline × each of its display taps × filter. Lazy; existence is
    NOT checked here (use `existing_for`).

    Reads are `recompute=False` (snapshot-only): a pipeline the build never ran
    for this object yields nothing (a 0.1ms miss), so iterating the whole
    class-slice never triggers a live trace on a request — produced pipelines
    hit their per-object snapshot, un-run ones skip. `ancestors` lets a child
    leaf resolve from its cart-item ancestor's snapshot."""
    for pipeline in pipelines_for(object_type, object_id):
        for e in tap_outputs_for((object_type, object_id, pipeline),
                                 recompute=False, ancestors=ancestors):
            yield Product(object_type, object_id, pipeline,
                          e["display"], e["filter"])


def existing_for(
    object_type: str, object_id: str, ancestors: Ancestors = (),
) -> Iterator[Product]:
    """The products for this object that are on disk and viewable — the viewer's
    slice of the catalog."""
    for product in products_for(object_type, object_id, ancestors):
        if exists(product, ancestors):
            yield product


def default_product_for(
    object_type: str, object_id: str, filter: "str | None" = None,
    ancestors: Ancestors = (),
) -> "Product | None":
    """The product to show for a bare object URL: the highest-preference pipeline
    that actually produced a viewable tap, on the object type's default display
    where available, else any existing tap in preference order. Optionally pinned
    to one `filter`. None when nothing is on disk yet."""
    want = default_display_for(object_type)
    existing = [p for p in existing_for(object_type, object_id, ancestors)
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
