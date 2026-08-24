"""Panels-as-tiles: run rc-astro ops over a merged mosaic one PANEL at a time.

Replaces the old arbitrary-grid tiling (`tile_grid` + `process_in_tiles`). The
natural decomposition of a gigapixel merged mosaic is its PANELS — each panel is
a `Target` with its own churn-stable `content_hash` and a footprint bbox in the
viewbox. Tiling on panels instead of an anonymous grid buys three things:

  * **No collisions by construction.** Each tile carries strong identity (the
    panel's hash, threaded through every op via the manifest below), so two
    derived products that share a mosaic/rect — BXT vs SXT, S vs O, stars vs
    nebula — never resolve to one crop. The mono-stars bug class is gone.
  * **O(change) re-runs.** A panel that re-stacks re-runs only its own tile plus
    the tiles it overlaps (the feather "ring"), never the whole tail — the
    re-run cost scales with what changed, not with mosaic size.
  * **Best-practice quality.** Panels are ≤100 Mpx (smaller than the old 100 Mpx
    grid tiles), so rc-astro never OOMs, and each tool sees a PSF/star-field
    coherent unit (RC-Astro's mosaic guidance — see the Croman two-pass note in
    the pipeline builder; the pre-placement `bxt_correct_only` is Round 1, this
    is Round 2 on the assembled mosaic).

A `PanelMosaic` pairs the merged image (pixels) with a per-panel MANIFEST — a
`{panel_id: derived_hash}` map where each hash folds in every op applied so far
but ONLY the local panels, so a distant panel's re-stack never moves it. The
derived hash is literally the processing op's own output identity
(`op.filename_fn(...).stem`), so it drifts from the real cache key by
construction-impossibility. `process_in_panels` crops each panel's padded bbox
from the merged image (neighbours come free from the already-merged raster), runs
the op, feathers the results back, and threads an updated manifest. Crops key on
the manifest — never on the merged image's whole-image `content_hash` (which
would churn on any panel and re-derive the entire tail).
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import ray

from decisions.settings import ProcessingConfig
from decorators import node
from dependency_graph.classes import Graph
from dependency_graph.tracing import IdentityRef, present_terminal, record
from file_io.tiff_file_io import write_tiff_image
from file_io.tiling import crop_source_region, feather_blend_to_bigtiff
from file_io.types import TiffFilePath, WithSidecar
from hashing import hash_list, hash_source_code
from metadata.classes import ImageMetadata
from mosaics.trimming import project_to_viewbox
# MosaicLike bound as `Mosaic`: widens beartype to accept a Target acting as
# its own single-panel mosaic WITHOUT touching the `mosaic: Mosaic`
# annotations — those serialize into the op code hash, so editing them would
# re-key every op in this module (see mosaics/types.py).
from mosaics.types import MosaicLike as Mosaic, Viewbox
from targets.types import Target


@dataclass(frozen=True)
class PanelMosaic:
    """A merged mosaic image (`merged`, the pixels) plus its per-panel identity
    `manifest` = `{panel_id: derived_hash}`. The panel_id is `target.id`
    (`stable_hash`); the derived hash folds in every op applied so far but only
    the LOCAL panels (see the module docstring). Trace-time only — never hashed
    into a node's key, never persisted; it just carries identity between stages."""
    merged: object                       # merged-image node handle (WithSidecar/ImageMetadata)
    manifest: dict                       # {panel_id: derived_hash_str}
    mosaic: Mosaic = field(compare=False)


def base_panel_mosaic(mosaic: Mosaic, filter: str, merged,
                      panel_sources: dict | None = None) -> PanelMosaic:
    """Wrap a freshly-merged per-filter channel as the stage-0 `PanelMosaic`.

    Base manifest = `hash(panel_source, filter)` per panel, where `panel_source`
    is that panel's PRE-MERGE product identity (`bxt_correct_only`'s output stem,
    built by `panel_source_manifest`). FILTER-specific so the H and O channels
    never collide, which raw `target.content_hash` — being all-filters — would
    allow.

    WHY the pre-merge identity and not `target.content_hash` (2026-08-15): the
    content hash describes the target's INPUT SUBS, so two pipelines processing
    the same target produce byte-identical manifests and the whole colour tail
    collides onto one cached product. That is exactly what happened to the
    GraXpert-vs-MGC A/B on the Heart nebula: divergence propagated correctly
    through Post-Processed → Channel → Mosaic (18 taps differed), then every
    scene format — Linear SHO/RGB/Narrowband/RGB Stars — resolved to ONE file,
    so the two pipelines rendered pixel-identical images. The pre-merge stem
    folds in whatever produced those pixels, so pipelines separate.

    It stays PER-PANEL, which is the property that matters: each entry depends
    only on its own panel, so a panel re-stack still moves only its own tile
    plus the feather ring. Folding in the MERGED channel's whole-image hash
    would also separate the pipelines, but every panel would churn whenever any
    panel moved — the exact anti-pattern this module exists to avoid (see the
    module docstring and `crop_panel`).

    `panel_sources` is `{panel_id: identity}`; a panel absent from it falls back
    to `target.content_hash` (the pre-2026-08-15 value) so callers that never
    build panels — unit tests exercising manifest algebra — still work. Every
    production pipeline passes it; a pipeline that forgets will silently share
    the tail with any other pipeline on the same target."""
    sources = panel_sources or {}
    manifest = {t.id: hash_list([sources.get(t.id, t.content_hash), filter])
                for t in mosaic.targets}
    return PanelMosaic(merged=merged, manifest=manifest, mosaic=mosaic)


# --------------------------------------------------------------------------- #
# Panel geometry — the panels ARE the tiling (no arbitrary grid)              #
# --------------------------------------------------------------------------- #

def _panel_pixel_bboxes(mosaic: Mosaic, viewbox) -> list[tuple]:
    """`[(panel_id, (x0, y0, x1, y1) | None), ...]` for every panel: the target
    footprint projected into `viewbox` pixel space and rounded OUTWARD (floor
    mins / ceil maxs), a superset of the true footprint. `None` when the footprint
    won't project finitely (degenerate / off-canvas) — such a panel gets no tile
    but is treated as present in every other tile's scope (include-when-in-doubt).
    Reuses `project_to_viewbox` — the same world→pixel map `place_image` uses — so
    these boxes live in the exact coordinates the crops are taken in."""
    boxes = []
    for t in mosaic.targets:
        bounds = project_to_viewbox(t.footprint, viewbox).bounds
        if len(bounds) == 4 and all(np.isfinite(bounds)):
            minx, miny, maxx, maxy = bounds
            box = (int(np.floor(minx)), int(np.floor(miny)),
                   int(np.ceil(maxx)), int(np.ceil(maxy)))
        else:
            box = None
        boxes.append((t.id, box))
    return boxes


def _rects_overlap(a: tuple, b: tuple) -> bool:
    """Axis-aligned overlap of two (x0, y0, x1, y1) boxes; edge-touch counts as
    overlapping (every boundary rounds toward INCLUDING — too broad re-runs a few
    extra tiles, too narrow silently corrupts)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    # Separated iff one box is fully left/right/above/below the other.
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def panel_rect(panel, viewbox, pad: int) -> "tuple | None":
    """One panel's padded crop rect `(x0, y0, w, h)` in `viewbox` pixel space,
    clamped to the canvas — or None when the panel doesn't project finitely.

    Computed at RUNTIME from whichever viewbox the op actually received, so
    the rect always lives on the same canvas as the merged image it crops —
    the viewbox may be a measured-geometry node result the trace only knew as
    a handle. The panel's DECLARED footprint bounds the box; the declared vs
    measured drift observed on real data (3.38 deg of rotation, 25-90 px of
    translation, worst-case ~180 px corner displacement) is well inside the
    512 px default pad, and the feather design already treats an over-large
    crop as free (crops overlap by construction; the taper eats the excess).
    """
    bounds = project_to_viewbox(panel.footprint, viewbox).bounds
    if len(bounds) != 4 or not all(np.isfinite(bounds)):
        return None
    minx, miny, maxx, maxy = bounds
    W, H = viewbox.num_pixels_x, viewbox.num_pixels_y
    x0 = max(0, int(np.floor(minx)) - pad)
    y0 = max(0, int(np.floor(miny)) - pad)
    x1 = min(W, int(np.ceil(maxx)) + pad)
    y1 = min(H, int(np.ceil(maxy)) + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def panel_tiles(mosaic: Mosaic, viewbox, config: ProcessingConfig) -> list[tuple]:
    """`[(panel_id, padded_rect(x0,y0,w,h), scope_ids), ...]` — one tile per panel
    with a finite bbox.

    Each panel's bbox is grown by `config.tiling.overlap_px` (an ABSOLUTE margin,
    px) on every side into its neighbours (clamped to the canvas) so the crop
    carries the edge context BXT/SXT need; it's feathered away at reassembly. The
    margin is absolute, not a fraction, because the tools' edge-support radius R is
    absolute — the seam is invisible when the crop overlap O ≥ 2R (see
    TilingSettings.overlap_px). `scope_ids` = the panels whose (un-padded) bbox
    intersects this padded rect,
    unioned with `always` (panels that don't project) — the sorted set of panel
    ids whose change must re-run this tile. Pure integer geometry: trace-time,
    reads no pixels."""
    W, H = viewbox.num_pixels_x, viewbox.num_pixels_y
    geom = _panel_pixel_bboxes(mosaic, viewbox)
    always = tuple(pid for pid, box in geom if box is None)
    boxed = [(pid, box) for pid, box in geom if box is not None]
    pad = config.tiling.overlap_px   # ABSOLUTE margin per side (px); see TilingSettings
    tiles = []
    for pid, (bx0, by0, bx1, by1) in boxed:
        px0, py0 = max(0, bx0 - pad), max(0, by0 - pad)
        px1, py1 = min(W, bx1 + pad), min(H, by1 + pad)
        if px1 <= px0 or py1 <= py0:
            continue   # panel projects entirely off-canvas → contributes no tile
        padded = (px0, py0, px1, py1)
        scope = set(always)
        scope.update(oid for oid, obox in boxed if _rects_overlap(padded, obox))
        rect = (px0, py0, px1 - px0, py1 - py0)
        tiles.append((pid, rect, tuple(sorted(scope))))
    return tiles


# --------------------------------------------------------------------------- #
# The two @node primitives: crop one panel, feather them back                 #
# --------------------------------------------------------------------------- #

def _crop_panel_narrow(args, kwargs):
    """Cache key = (mosaic.stable_hash, panel.stable_hash, viewbox, pad,
    scope_manifest_hashes).

    The rect used to sit here directly; it is now DERIVED at runtime from
    `(panel, viewbox, pad)`, so those three stand in for it — same
    information, but computable at trace time when `viewbox` is a handle
    (its content_hash is what the hash machinery reduces it to, and the
    runtime-stamped Viewbox carries the identical hash).

    `source` (the merged image handle) is DELIBERATELY excluded: its whole-image
    content_hash churns whenever any panel moved. `scope_manifest_hashes` — the
    derived hashes of the panels overlapping this tile, from the SOURCE stage's
    manifest — is what identifies the tile: it moves iff a panel in scope changed
    OR a prior op changed (the manifest folds both), and differs between stages
    (BXT-source manifest ≠ SXT-source manifest), so no two products collide. No
    `config`: config-sensitivity already rides inside `scope_manifest_hashes`
    (each derived hash is an op's `filename_fn`, which narrows on config) — and
    `pad` (the one tiling field the rect depends on) is an explicit arg."""
    mosaic, panel, viewbox, pad, source, scope_hashes = args
    return (mosaic.stable_hash, panel.stable_hash, viewbox, pad,
            scope_hashes), kwargs


# ephemeral: a materialized VIEW of the merged raster, written only because
# rc-astro needs a file path. Re-cropping is seconds; keeping it cost 235 GB.
@node(cache_key_narrow=_crop_panel_narrow, auto_memory=True, plate_solve=False,
      ephemeral=True)
def crop_panel(mosaic: Mosaic, panel: Target, viewbox: Viewbox, pad: int,
               source: ImageMetadata,
               scope_hashes: tuple) -> WithSidecar[TiffFilePath]:
    """One panel's padded bbox cropped out of the merged mosaic — streaming, so
    only the tile's pixels are read. Written STRIP-based so rc-astro reads it
    directly (it rejects tiled TIFF). Ephemeral: reclaimed after its op consumes
    it. The rect is derived HERE from `(panel, viewbox, pad)` — see
    `panel_rect` — so it always lives on the canvas the merged image was
    actually built on. The body reads `source.filename` (the merged image is
    always fresh on a miss); `scope_hashes` only tightens the key (see
    `_crop_panel_narrow`)."""
    rect = panel_rect(panel, viewbox, pad)
    assert rect is not None, (
        f"panel {panel.stable_hash} projects off the canvas it was placed "
        f"on — viewbox/placement disagreement")
    out = crop_panel.filename_fn(mosaic, panel, viewbox, pad, source,
                                 scope_hashes)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_tiff_image(Path(out), crop_source_region(source.filename, rect),
                     {"PIPELINE": "AstroFlow-OSS"})
    return WithSidecar(TiffFilePath(out))


def _blend_panels_narrow(args, kwargs):
    """Beyond the tile handles + the rect inputs (panels + viewbox + the
    feather params in `config.tiling`), nothing matters; route on
    `mosaic.stable_hash` (not the churning `content_hash`). Panels reduce to
    their stable ids — their CONTENT is already folded into each tile."""
    mosaic, config, viewbox, panels, *tiles = args
    return (mosaic.stable_hash, config.tiling, viewbox,
            tuple(p.stable_hash for p in panels), *tiles), kwargs


# ephemeral: a feather of tiles that are THEMSELVES retained (the GPU stages
# persist), so re-deriving this is one blend — ~58 s median — not GPU minutes.
# It is consumed only by the next stage's crops inside the same pipeline graph,
# so it is reclaimed when that graph concludes. Measured 2026-08-17: 181 GB,
# 72 GB per compute-hour — the worst disk-for-compute trade in the tail.
@node(cache_key_narrow=_blend_panels_narrow, auto_memory=True, plate_solve=False,
      ephemeral=True)
def blend_panels(mosaic: Mosaic, config: ProcessingConfig, viewbox: Viewbox,
                 panels: tuple,
                 *tiles: ImageMetadata) -> WithSidecar[TiffFilePath]:
    """Feather the processed panel tiles back into one BigTIFF — streaming (never
    materializes the canvas). The generic INTERMEDIATE blend (no `display=`); a
    deliverable passes its own display-carrying blend to `process_in_panels`.
    Rects are re-derived from `(panels, viewbox, pad)` — the identical
    arithmetic the crops used, so tile and placement can't disagree."""
    pad = config.tiling.overlap_px
    rects = [panel_rect(panel, viewbox, pad) for panel in panels]
    assert all(r is not None for r in rects), "panel projects off-canvas"
    out = blend_panels.filename_fn(mosaic, config, viewbox, panels, *tiles)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    feather_blend_to_bigtiff(
        [tuple(r) for r in rects], [t.filename for t in tiles], out,
        taper_px=config.tiling.overlap_px, blend=config.tiling.blend,
        metadata={"PIPELINE": "AstroFlow-OSS"})
    return WithSidecar(TiffFilePath(out))


# --------------------------------------------------------------------------- #
# The user-facing helpers                                                     #
# --------------------------------------------------------------------------- #

@ray.remote(num_cpus=0.1)
def _panel_stage(op_ref, mosaic, panel, viewbox, pad, scope_hashes, lead,
                 source_merged, dry_run=False):
    """One panel: crop its padded bbox out of the merged source, run `op` on
    that tile, and hand the tile back. Returned as a dynamic subgraph with
    `delete_intermediates=True`, so the CROP — a view of `source_merged` that
    exists only because rc-astro needs a file path — is deleted when this
    subgraph concludes, while the op's output (GPU minutes) escapes and
    persists.

    `op_ref` is `(module, name)` rather than the op wrapper itself: a wrapper
    in a Node's args would be hashed by its address-based repr, which the
    determinism guard rejects (and rightly — it would flap every process)."""
    del dry_run
    import importlib
    from dependency_graph.tracing import trace

    op = getattr(importlib.import_module(op_ref[0]), op_ref[1])

    def _body():
        crop = crop_panel(mosaic, panel, viewbox, pad, source_merged,
                          scope_hashes)
        return op(*lead, crop)

    return Graph(trace(_body))


def process_in_panels(op, *op_args, viewbox, blend=blend_panels) -> PanelMosaic:
    """Run `op` over each panel of a merged mosaic, then feather the results back.

    `op_args` are exactly what you'd pass `op` directly, with the SOURCE (a
    `PanelMosaic`) last — e.g. `process_in_panels(bxt_correct, mosaic, cf, config,
    pm, viewbox=vb)`. For each panel: crop its padded bbox out of `pm.merged`
    (neighbours free), run `op` on the tile, and record the tile's new identity =
    `op.filename_fn(...).stem` into the output manifest. `blend` feathers the
    tiles into a new merged image (pass a display-carrying blend for a
    deliverable). Returns a new `PanelMosaic` (merged image + threaded manifest).

    The crop keys on the SOURCE manifest's scoped hashes, never on `pm.merged`'s
    whole-image hash, so a distant panel's re-stack re-runs nothing here."""
    mosaic, source, lead = op_args[0], op_args[-1], op_args[:-1]
    config = next(a for a in op_args if isinstance(a, ProcessingConfig))
    if not isinstance(source, PanelMosaic):
        raise TypeError(
            f"process_in_panels source (last arg) must be a PanelMosaic, "
            f"got {type(source).__name__}")

    manifest = dict(source.manifest)   # carry non-tiled panels forward unchanged
    # Each panel becomes a two-node DYNAMIC SUBGRAPH — `crop_panel` then `op`,
    # with `op` as the terminal — declared `delete_intermediates=True`. That is
    # the canonical shape for this mechanism: a cheap op feeding an expensive
    # one. The crop (a view of the merged raster, materialized only because
    # rc-astro needs a file path) is deleted when its panel's subgraph
    # concludes; the op's output — GPU minutes — escapes and persists, so
    # incremental runs still cache-hit it. Before this, 24 panels x several
    # stages left 235 GB of crops on disk forever.
    op_ref = (op._function.__module__, op._function.__name__)
    # SCOPE (which panels invalidate which tiles) is decided on the DECLARED
    # Master canvas, deliberately: the passed `viewbox` may be a runtime
    # handle (measured geometry), unreadable at trace time, and scope only
    # needs a conservative overlap answer, not the exact canvas. The pad is
    # inflated 8x so the answer stays conservative for downscaled canvases
    # too (their pad covers 8x the sky per pixel) — "too broad re-runs a few
    # extra tiles, too narrow silently corrupts". Crop RECTS, by contrast,
    # are computed at runtime inside the ops from the viewbox they actually
    # received.
    import dataclasses as _dc
    from mosaics.viewbox import compute_viewbox as _declared_viewbox
    scope_config = _dc.replace(
        config, tiling=_dc.replace(config.tiling,
                                   overlap_px=config.tiling.overlap_px * 8))
    scope_tiles = panel_tiles(
        mosaic, _declared_viewbox("Master", mosaic), scope_config)
    pad = config.tiling.overlap_px
    targets_by_id = {t.id: t for t in mosaic.targets}
    panels, handles = [], []
    for pid, _scope_rect, scope in scope_tiles:
        panel = targets_by_id[pid]
        scope_hashes = tuple(sorted(
            source.manifest[j] for j in scope if j in source.manifest))
        # Identities are pure `filename_fn` arithmetic — no node in THIS trace.
        # The crop's stem is its call hash, which is exactly the content_hash
        # its result will carry, so the op's identity is unchanged from when
        # the crop handle sat here. A handle viewbox reduces to its stamped
        # content_hash; the resolved Viewbox carries the same one.
        crop_id = Path(crop_panel.filename_fn(
            mosaic, panel, viewbox, pad, source.merged, scope_hashes)).stem
        # The tile's new identity IS its op's output identity — folds op code,
        # config (via the op's narrow), and the scoped source, drift-free.
        tile_path = op.filename_fn(*lead, IdentityRef(crop_id))
        handle = record(_panel_stage, op_ref, mosaic, panel, viewbox, pad,
                        scope_hashes, lead, source.merged)
        # Builder ≡ terminal: the handle presents the op's output — identity
        # (so `blend`'s key and every downstream key are bit-identical to the
        # pre-subgraph version), expected path (so the workflow UI's on-disk
        # checks see the tile, not an opaque stub), and op name. The GPU ops
        # carry no display=, so nothing UI-facing is stamped here.
        present_terminal(handle, op, path=tile_path)
        handles.append(handle)
        panels.append(panel)
        manifest[pid] = Path(tile_path).stem
    merged = blend(mosaic, config, viewbox, tuple(panels), *handles)
    return PanelMosaic(merged=merged, manifest=manifest, mosaic=mosaic)


def combine_panels(combine_op, mosaic: Mosaic, *inputs: PanelMosaic) -> PanelMosaic:
    """Point-wise combine (no spatial support — e.g. RGB band-join, SHO stack) of
    several `PanelMosaic`s. Runs the `@node` combine on the merged images, and
    combines the manifests PANEL-WISE (panel i's output depends only on panel i's
    inputs, so no scope): `manifest[i] = hash(combine_code, in₀[i], in₁[i], …)`.
    Filter-specificity/identity flows straight through from the inputs."""
    merged = combine_op(mosaic, *[pm.merged for pm in inputs])
    tag = hash_source_code(combine_op._original)
    ids = set().union(*(set(pm.manifest) for pm in inputs))
    manifest = {
        pid: hash_list([tag, *[pm.manifest.get(pid, "") for pm in inputs]])
        for pid in ids
    }
    return PanelMosaic(merged=merged, manifest=manifest, mosaic=mosaic)
