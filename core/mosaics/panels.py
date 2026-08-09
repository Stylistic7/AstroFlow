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
from dependency_graph.tracing import record
from file_io.tiff_file_io import write_tiff_image
from file_io.tiling import crop_source_region, feather_blend_to_bigtiff
from file_io.types import TiffFilePath, WithSidecar
from hashing import hash_list, hash_source_code
from metadata.classes import ImageMetadata
from mosaics.trimming import project_to_viewbox
from mosaics.types import Mosaic


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


def base_panel_mosaic(mosaic: Mosaic, filter: str, merged) -> PanelMosaic:
    """Wrap a freshly-merged per-filter channel as the stage-0 `PanelMosaic`.

    Base manifest = `hash(target.content_hash, filter)` per panel: DATA-sensitive
    (a panel's re-stack moves its `content_hash`) and FILTER-specific (so the H
    and O channels never collide, which raw `target.content_hash` — being
    all-filters — would allow). Deliberately does NOT fold in the pre-merge op
    code (`bxt_correct_only`/`place_image`/`merge_images_oss`); a change to those
    needs an eviction like any callee edit, exactly as the old tiling required."""
    manifest = {t.id: hash_list([t.content_hash, filter]) for t in mosaic.targets}
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
    """Cache key = (mosaic.stable_hash, padded_rect, scope_manifest_hashes).

    `source` (the merged image handle) is DELIBERATELY excluded: its whole-image
    content_hash churns whenever any panel moved. `scope_manifest_hashes` — the
    derived hashes of the panels overlapping this tile, from the SOURCE stage's
    manifest — is what identifies the tile: it moves iff a panel in scope changed
    OR a prior op changed (the manifest folds both), and differs between stages
    (BXT-source manifest ≠ SXT-source manifest), so no two products collide. No
    `config`: config-sensitivity already rides inside `scope_manifest_hashes`
    (each derived hash is an op's `filename_fn`, which narrows on config)."""
    mosaic, rect, source, scope_hashes = args
    return (mosaic.stable_hash, rect, scope_hashes), kwargs


@node(cache_key_narrow=_crop_panel_narrow, auto_memory=True, plate_solve=False)
def crop_panel(mosaic: Mosaic, rect: tuple, source: ImageMetadata,
               scope_hashes: tuple) -> WithSidecar[TiffFilePath]:
    """One panel's padded bbox cropped out of the merged mosaic — streaming, so
    only the tile's pixels are read. Written STRIP-based so rc-astro reads it
    directly (it rejects tiled TIFF). Ephemeral: reclaimed after its op consumes
    it. The body reads `source.filename` (the merged image is always fresh on a
    miss); `scope_hashes` only tightens the key (see `_crop_panel_narrow`)."""
    out = crop_panel.filename_fn(mosaic, rect, source, scope_hashes)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_tiff_image(Path(out), crop_source_region(source.filename, rect),
                     {"PIPELINE": "AstroFlow-OSS"})
    return WithSidecar(TiffFilePath(out))


def _blend_panels_narrow(args, kwargs):
    """Beyond the tile handles + their rects, only the feather params
    (`config.tiling`) matter; route on `mosaic.stable_hash` (not the churning
    `content_hash`)."""
    mosaic, config, rects, *tiles = args
    return (mosaic.stable_hash, config.tiling, rects, *tiles), kwargs


@node(cache_key_narrow=_blend_panels_narrow, auto_memory=True, plate_solve=False)
def blend_panels(mosaic: Mosaic, config: ProcessingConfig, rects: tuple,
                 *tiles: ImageMetadata) -> WithSidecar[TiffFilePath]:
    """Feather the processed panel tiles back into one BigTIFF — streaming (never
    materializes the canvas). The generic INTERMEDIATE blend (no `display=`); a
    deliverable passes its own display-carrying blend to `process_in_panels`."""
    out = blend_panels.filename_fn(mosaic, config, rects, *tiles)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    feather_blend_to_bigtiff(
        [tuple(r) for r in rects], [t.filename for t in tiles], out,
        taper_px=config.tiling.overlap_px, blend=config.tiling.blend,
        metadata={"PIPELINE": "AstroFlow-OSS"})
    return WithSidecar(TiffFilePath(out))


# --------------------------------------------------------------------------- #
# The user-facing helpers                                                     #
# --------------------------------------------------------------------------- #

class _CropRef:
    """A stand-in for a crop handle carrying only its identity.

    `process_in_panels` derives each tile's manifest entry from the op's own
    output identity (`op.filename_fn(*lead, crop)`), which needs the crop's
    `content_hash`. The crop node now lives inside a per-panel subgraph, so
    the outer trace never sees its handle — but the crop's identity is pure
    arithmetic (`crop_panel.filename_fn(...)`), and cache keys resolve any
    object with a `content_hash` (`hashing._maybe_use_content_hash`). Same
    value the handle produced; no node required."""
    __slots__ = ("content_hash",)

    def __init__(self, content_hash: str) -> None:
        self.content_hash = content_hash


@ray.remote(num_cpus=0.1)
def _panel_stage(op_ref, mosaic, rect, scope_hashes, lead, source_merged,
                 dry_run=False):
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
        crop = crop_panel(mosaic, rect, source_merged, scope_hashes)
        return op(*lead, crop)

    return Graph(trace(_body), delete_intermediates=True)


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
    rects, handles = [], []
    for pid, rect, scope in panel_tiles(mosaic, viewbox, config):
        scope_hashes = tuple(sorted(
            source.manifest[j] for j in scope if j in source.manifest))
        # Identities are pure `filename_fn` arithmetic — no node in THIS trace.
        # The crop's stem is its call hash, which is exactly the content_hash
        # its result will carry, so the op's identity is unchanged from when
        # the crop handle sat here.
        crop_id = Path(crop_panel.filename_fn(
            mosaic, rect, source.merged, scope_hashes)).stem
        # The tile's new identity IS its op's output identity — folds op code,
        # config (via the op's narrow), and the scoped source, drift-free.
        tile_id = Path(op.filename_fn(*lead, _CropRef(crop_id))).stem
        handle = record(_panel_stage, op_ref, mosaic, rect, scope_hashes,
                        lead, source.merged)
        # Stamp the subgraph handle with the identity of what it RESOLVES to
        # (the op's output), so `blend`'s key — and every downstream key — is
        # bit-identical to the pre-subgraph version.
        handle.content_hash = tile_id
        handles.append(handle)
        rects.append(rect)
        manifest[pid] = tile_id
    merged = blend(mosaic, config, tuple(rects), *handles)
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
