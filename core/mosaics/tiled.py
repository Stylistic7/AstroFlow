"""Graph-level tiling idiom — run any single-image `@node` op over big
overlapping tiles as distributable graph nodes, then feather them back.

`process_in_tiles(op, *op_args, dims=..., blend=...)` is the user-facing pipeline
helper. `op` is an ordinary `@node` shaped `op(..., image) -> image` (image LAST).
Call it directly to run whole; wrap it in `process_in_tiles` to tile — SAME node,
same args, the caller decides per use. process_in_tiles crops the source (the
last of `op_args`) into overlapping tiles via `crop_tile` nodes, runs `op` on
each tile (op is unchanged — it just sees a smaller image), and reassembles with
a streaming feather `blend`. The pure grid/crop/blend math + the real-data
equivalence proof (whole ≈ tiled, PSNR ~76 dB, no seams) live in
`file_io/tiling.py`.

Why tiling is safe here: rc-astro (BXT/NXT/SXT) already tile internally (512 px,
semi-independent for non-stationary PSF) — they're local, not global-statistics
tools — so a coarser external tiling with a feathered overlap is the same idea.

Incremental cache scoping (the panel_scope machinery below)
-----------------------------------------------------------
The color tail chains tiled ops over a MOSAIC source (channel → BXT → SXT →
NXT …). Each stage's derived source gets a fresh whole-image `content_hash`
whenever ANY tile of the previous stage moved, so keying `crop_tile` on `source`
(the pre-scoping behaviour) meant one changed panel re-keyed EVERY crop → the
whole gigapixel tail re-derived. Instead, when the source decomposes into panels
(`op_args[0]` is a `Mosaic`) we key each `crop_tile` on the `panel_scope` — the
mosaic panels whose footprint reaches that tile — NOT on `source`. A panel's
`content_hash` only moves when that panel's data/code changes, so a tile whose
panels are unchanged is a cache HIT regardless of the derived source's churn:
changing a few panels re-runs only the tiles they touch.

Correctness of the scope is load-bearing — a too-narrow scope silently produces a
WRONG tile (no error). See `_panel_scopes` / `_scope_for_rect` for the margin
argument. For a non-decomposable source (a single big stack/image) `panel_scope`
is None and the key is byte-for-byte the pre-scoping `(mosaic, rect, source)`.
"""
import os
from pathlib import Path

import numpy as np

from astroflow.pipeline import ephemeral_intermediates
from decisions.settings import ProcessingConfig
from decorators import node
from file_io.tiff_file_io import write_tiff_image
from file_io.tiling import tile_grid, crop_source_region, feather_blend_to_bigtiff
from file_io.types import TiffFilePath, WithSidecar
from metadata.classes import ImageMetadata
from mosaics.types import Mosaic


# How many chained tiled-stage overlap-distances to pad a tile by when deciding
# which panels reach it. Each tiled stage is LOCAL, but its feathered reassembly
# lets a tile's pixels depend on its source up to ONE overlap distance
# (`round(tile_px*overlap)`) away — the op support the overlap is sized to hide.
# So the crop at stage k depends on the ORIGINAL panels within the tile expanded
# by (number of chained tiled stages upstream of this crop) × overlap-distance
# (the mosaic-merge stage adds no extra reach: the merge only shrinks panels via
# edge erosion, and the UN-eroded footprint bbox we test is already a superset of
# a panel's contribution to the channel). The deepest crop the OSS tail builds
# feeds NXT, whose source (the SHO stack) is two chained tiled stages downstream
# of the merge (channel → BXT → SXT → pointwise-stack → NXT). We pad by THREE
# overlap-distances: covers that 2-stage reach with a full extra stage of slack
# for sub-pixel projection disagreement and any future deepening. Too broad only
# costs a few extra tile re-runs; too narrow corrupts — when in doubt, include.
_SCOPE_MARGIN_OVERLAPS = 3


def _crop_narrow(args, kwargs):
    # A crop's identity is (rect, source content); `mosaic` only routes the path.
    #
    # Mosaic source (`panel_scope` present) — the panels ARE the identity. Key on
    # the STABLE mosaic id (mosaic.content_hash composes ALL panels, so keying on
    # the Mosaic object would itself churn on any panel change), the rect, the
    # overlapping panels' content_hashes, and the pipeline `config`. `source` is
    # DELIBERATELY excluded: its whole-image content_hash churns whenever any tile
    # of a prior stage moved — exactly the whole-tail re-run this scoping kills.
    # `config` IS kept so tuning a tiled op (bxt/sxt/nxt/tiling) still invalidates
    # every crop of its downstream source (config is constant across panel
    # changes, so folding it in costs nothing incrementally). NOTE: editing the
    # BODY of a tiled op does NOT auto-invalidate crops of its output — evict
    # like any other callee edit (see project memory).
    #
    # Generic source (`panel_scope` None) — fall back to the pre-scoping key,
    # byte-for-byte, so a single-image source is a zero-regression no-op.
    mosaic, rect, source, panel_scope, config = args
    if panel_scope is None:
        return (mosaic, rect, source), kwargs
    return (mosaic.stable_hash, rect, panel_scope, config), kwargs


@node(cache_key_narrow=_crop_narrow, auto_memory=True, plate_solve=False)
def crop_tile(mosaic: Mosaic, rect: tuple, source: ImageMetadata,
              panel_scope: tuple | None, config: ProcessingConfig) -> WithSidecar[TiffFilePath]:
    """One overlapping tile cropped out of the (tiled BigTIFF) source — streaming,
    so only the tile's pixels are read. Written STRIP-based so rc-astro ops can
    read it directly (rc-astro rejects tiled TIFF). The unit `process_in_tiles`
    feeds to `op`; ephemeral — reclaimed after `op` consumes it.

    `panel_scope` (from `process_in_tiles`) is the set of mosaic panels reaching
    this tile; when set it — not `source` — keys the tile, so a panel change
    re-runs only the tiles it touches. The body still reads `source.filename` on a
    miss (the merged source on disk is always fresh); `config` is unused here, it
    only tightens the cache key."""
    out = crop_tile.filename_fn(mosaic, rect, source, panel_scope, config)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_tiff_image(Path(out), crop_source_region(source.filename, rect),
                     {"PIPELINE": "AstroFlow-OSS"})
    return WithSidecar(TiffFilePath(out))


def _blend_narrow(args, kwargs):
    mosaic, config, rects, *tiles = args
    return (mosaic, config.tiling, rects, *tiles), kwargs


@node(cache_key_narrow=_blend_narrow, auto_memory=True, plate_solve=False)
def blend_tiles(mosaic: Mosaic, config: ProcessingConfig, rects: tuple,
                *tiles: ImageMetadata) -> WithSidecar[TiffFilePath]:
    """Feather the processed tiles back into one BigTIFF — streaming (never
    materializes the canvas). The generic INTERMEDIATE blend (no `display=`); a
    deliverable passes its own display-carrying blend to `process_in_tiles`."""
    out = blend_tiles.filename_fn(mosaic, config, rects, *tiles)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    feather_blend_to_bigtiff(
        [tuple(r) for r in rects], [t.filename for t in tiles], out,
        overlap=config.tiling.overlap, blend=config.tiling.blend,
        metadata={"PIPELINE": "AstroFlow-OSS"})
    return WithSidecar(TiffFilePath(out))


def _target_pixel_bboxes(mosaic: Mosaic, viewbox) -> list[tuple]:
    """`[(content_hash, (x0, y0, x1, y1) | None), ...]` for every mosaic panel:
    the target footprint projected into `viewbox`'s pixel frame and rounded
    OUTWARD (floor mins / ceil maxs) so each box is a SUPERSET of the true
    projected footprint. Reuses `mosaics.trimming.project_to_viewbox` — the SAME
    `(x - x_min)/resolution` mapping `place_image` projects panels with — so these
    boxes live in the exact pixel coordinates `tile_grid` builds its rects in.

    Target-level (all-filters) `content_hash` is deliberate: a change to a target's
    O stack also re-runs its H tiles — slightly over-conservative, never wrong. A
    per-filter refinement can come later. A box is None when the footprint won't
    project finitely (degenerate / off-canvas); `_panel_scopes` then treats that
    panel as present in EVERY tile (include-when-in-doubt)."""
    from mosaics.trimming import project_to_viewbox

    boxes = []
    for target in mosaic.targets:
        bounds = project_to_viewbox(target.footprint, viewbox).bounds
        if len(bounds) == 4 and all(np.isfinite(bounds)):
            minx, miny, maxx, maxy = bounds
            box = (int(np.floor(minx)), int(np.floor(miny)),
                   int(np.ceil(maxx)), int(np.ceil(maxy)))
        else:
            box = None
        boxes.append((target.content_hash, box))
    return boxes


def _scope_for_rect(rect: tuple, boxes: list[tuple], always: tuple,
                    margin: int) -> tuple:
    """Sorted tuple of the panel content_hashes whose (outward-rounded) pixel
    bbox — grown by `margin` px — intersects tile `rect`, unioned with `always`
    (panels that had no finite bbox). PURE integer geometry: `rect` and `boxes`
    are plain pixels, so this IS the whole scoping decision and the unit the tests
    exercise. Standard axis-aligned overlap; a panel touching an edge counts as
    overlapping (every boundary rounds toward INCLUDING the panel — too broad is
    safe, too narrow corrupts)."""
    x0, y0, tw, th = rect
    rx0, ry0, rx1, ry1 = x0, y0, x0 + tw, y0 + th
    hits = set(always)
    for content_hash, box in boxes:
        bx0, by0, bx1, by1 = box
        separated = (bx1 + margin < rx0 or bx0 - margin > rx1
                     or by1 + margin < ry0 or by0 - margin > ry1)
        if not separated:
            hits.add(content_hash)
    return tuple(sorted(hits))


def _viewbox_for_dims(mosaic: Mosaic, dims: tuple):
    """The mosaic viewbox whose pixel size equals `dims`. The caller passed `dims`
    straight off one of the built-in quality viewboxes but not WHICH one
    (`process_in_tiles`' signature is fixed and takes only `dims`), so recover it
    by matching each quality's `(num_pixels_x, num_pixels_y)` against `dims`. The
    quality sizes are far apart, so the match is unambiguous. Reuses
    `mosaic.viewbox` / `compute_viewbox` rather than reconstructing the
    projection. Returns None if none matches (then scoping is skipped and the crop
    stays on the conservative `source` key)."""
    from previews.types import DOWNSCALE_FACTORS

    w, h = dims
    for mode in DOWNSCALE_FACTORS:
        vb = mosaic.viewbox(mode)
        if (vb.num_pixels_x, vb.num_pixels_y) == (w, h):
            return vb
    return None


def _panel_scopes(source_router, dims: tuple, rects: tuple,
                  config: ProcessingConfig) -> list:
    """Per-rect `panel_scope` aligned with `rects`, or all-None when the source
    does not decompose into panels (generic use — `op_args[0]` is not a `Mosaic`).

    A tile's scope is EVERY panel whose footprint reaches it, padded by
    `_SCOPE_MARGIN_OVERLAPS` overlap-distances so a panel that only feathers into
    the tile through a prior tiled stage is still included (see the module
    constant). Falls back to all-None (the conservative `source` key) if the
    viewbox can't be resolved — never narrows on uncertainty."""
    if not isinstance(source_router, Mosaic):
        return [None] * len(rects)
    viewbox = _viewbox_for_dims(source_router, dims)
    if viewbox is None:
        return [None] * len(rects)
    boxes = _target_pixel_bboxes(source_router, viewbox)
    always = tuple(h for h, box in boxes if box is None)
    boxed = [(h, box) for h, box in boxes if box is not None]
    margin = _SCOPE_MARGIN_OVERLAPS * round(
        config.tiling.tile_px * config.tiling.overlap)
    return [_scope_for_rect(r, boxed, always, margin) for r in rects]


def process_in_tiles(op, *op_args, dims, blend=blend_tiles):
    """Run `op` over big overlapping tiles of the source (the LAST of `op_args`),
    fanned out as distributable graph nodes, then feather them in `blend`.

    `op_args` are exactly what you'd pass `op` directly (source last); `dims` =
    (width, height) of the source, from the mosaic viewbox — the grid is built at
    TRACE time, before any node runs. `op` is unchanged: it just receives a tile
    image instead of the whole one. With `config.tiling.enabled` False it's a
    single full-image tile (i.e. `op` whole). The `ProcessingConfig` is found in
    `op_args`; tiles are ephemeral (reclaimed after `blend`); the deliverable
    identity lives on `blend`.

    When `op_args[0]` is a `Mosaic`, each `crop_tile` is keyed on the panels
    reaching its tile (its `panel_scope`) instead of on `source`, so changing a
    few panels re-runs only the overlapping tiles (see the module docstring);
    otherwise `panel_scope` is None and the tile keeps the pre-scoping key."""
    mosaic, source, lead = op_args[0], op_args[-1], op_args[:-1]
    config = next(a for a in op_args if isinstance(a, ProcessingConfig))
    w, h = dims
    if config.tiling.enabled:
        rects = tuple(tile_grid(w, h, config.tiling.tile_px, config.tiling.overlap))
    else:
        rects = ((0, 0, w, h),)   # single full-image tile == run `op` whole
    scopes = _panel_scopes(mosaic, dims, rects, config)
    with ephemeral_intermediates():
        tiles = [op(*lead, crop_tile(mosaic, r, source, scope, config))
                 for r, scope in zip(rects, scopes)]
    return blend(mosaic, config, rects, *tiles)
