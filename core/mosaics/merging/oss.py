"""OSS mosaic merging: distance- and inverse-variance-weighted N-way blend
with sky equalization.

Replaces PhotometricMosaic. Placed images are already in the same
coordinate system (the mosaic viewbox), so no WCS reprojection needed —
just feather across all seams at once and equalize brightness across
every overlapping pair.

Variadic (one call per mosaic-per-filter, blending every panel together
in a single global pass) rather than pairwise. Pairwise composition
gives order-dependent results because each `distance_transform_edt` is
recomputed on the running merged region, the flux equalization drifts
panel-by-panel, and N-1 stacked feather seams accumulate. The math
below sees every panel at once and is order-independent.
"""
import numpy as np
import ray
import structlog
from pathlib import Path

import edt

from decorators import node
from mosaics.placement.panel_io import read_placed_panel
from file_io.tiff_file_io import write_tiff_image
from file_io.types import Produced, TiffFilePath
from hashing import hash_list
from metadata.classes import ImageMetadata
from mosaics.merging.pixinsight import PANEL_COUNT, _panel_count
# MosaicLike bound as `Mosaic`: widens beartype to accept a Target acting as
# its own single-panel mosaic WITHOUT touching the `mosaic: Mosaic`
# annotations — those serialize into the op code hash, so editing them would
# re-key every op in this module (see mosaics/types.py).
from mosaics.types import MosaicLike as Mosaic

log = structlog.get_logger()


# The overlap-based flux solve (`_solve_flux_scales` + `_signal_means`)
# was DELETED 2026-09-01: gains are now calibrated absolutely, per master,
# against Gaia (post_stack_calibrate, graxpert_stack.py) — the only
# externally ground-truthed step in the pipeline — so panels arrive here
# gain-true and an overlap estimator with no ground truth could only add
# scene-dependent noise. Measured before deletion (run 20, 4x4 lab): with
# SPFC upstream the solve found ~unity and the merge landed at the chroma
# floor with or without it.


def _panel_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight (r0, r1, c0, c1) bounding box of the True region, or None if the
    mask is all-False. Row/col `any` reductions — no full-size intermediate."""
    rows = np.any(mask, axis=1)
    if not rows.any():
        return None
    cols = np.any(mask, axis=0)
    r0 = int(np.argmax(rows))
    r1 = int(len(rows) - np.argmax(rows[::-1]))
    c0 = int(np.argmax(cols))
    c1 = int(len(cols) - np.argmax(cols[::-1]))
    return (r0, r1, c0, c1)


def _intersect(a, b):
    """Canvas-coordinate intersection of two bboxes, or None if disjoint."""
    r0, r1 = max(a[0], b[0]), min(a[1], b[1])
    c0, c1 = max(a[2], b[2]), min(a[3], b[3])
    return (r0, r1, c0, c1) if (r0 < r1 and c0 < c1) else None


def _sub(mask_crop, bbox, region):
    """View of a panel's bbox-cropped mask over the canvas rectangle `region`
    (which must lie within `bbox`)."""
    br0, _, bc0, _ = bbox
    rr0, rr1, rc0, rc1 = region
    return mask_crop[rr0 - br0:rr1 - br0, rc0 - bc0:rc1 - bc0]


def _noise_sigma(crop: np.ndarray, mask: np.ndarray) -> float:
    """Per-pixel noise of a placed starless panel, in its merge units: the
    MAD of horizontal neighbour differences over the footprint, scaled to a
    Gaussian sigma and divided by sqrt(2) for the two pixels in each
    difference. Nebulosity is smooth at a 1 px lag and the layer is
    starless, so structure barely enters; BXT's sharpened noise texture is
    measured as it is, which is what the blend sees. Every 4th row, all
    columns. NaN when the footprint is too small to measure."""
    rows, m = crop[::4], mask[::4]
    d = np.diff(rows, axis=1)[m[:, 1:] & m[:, :-1]]
    if d.size < 1000:
        return float("nan")
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


def _inverse_variance(crops, masks) -> list[float]:
    """Per-panel 1/sigma^2, normalised to a maximum of 1 so the distance
    weights keep their pixel scale; 1 for a panel whose noise could not be
    measured."""
    sig = np.array([float("nan") if c is None else _noise_sigma(c, m) for c, m in zip(crops, masks)])
    inv = np.where(np.isfinite(sig) & (sig > 0), 1.0 / np.square(sig), np.nan)
    if not np.isfinite(inv).any():
        return [1.0] * len(crops)
    inv = inv / np.nanmax(inv)
    return [1.0 if not np.isfinite(v) else float(v) for v in inv]


def _edt_weight(mask_crop, bbox, canvas_shape):
    """`edt.edt` of a panel restricted to its bbox — bit-identical to
    transforming the full-canvas mask and slicing to the bbox.

    Distance-to-background is local, so we transform the crop, not the canvas.
    A 1-px zero border is added on every side interior to the canvas so an
    edge-of-footprint pixel finds the same nearest background it would across
    the full canvas; a side flush against the canvas edge gets no pad, matching
    `black_border=False` there. 1 px suffices: a footprint pixel's nearest
    background is either the footprint boundary (inside the bbox) or one step
    outside the bbox (supplied by the pad)."""
    r0, r1, c0, c1 = bbox
    H, W = canvas_shape
    pt, pb = int(r0 > 0), int(r1 < H)
    pl, pr = int(c0 > 0), int(c1 < W)
    padded = np.pad(mask_crop.astype(np.uint8), ((pt, pb), (pl, pr)))
    # `black_border=False` + `parallel=0` (all cores): matches
    # scipy.ndimage.distance_transform_edt, ~13x faster (was ~91% of wall time).
    dist = edt.edt(padded, black_border=False, parallel=0).astype(
        np.float32, copy=False)
    return dist[pt:pt + (r1 - r0), pl:pl + (c1 - c0)]


# Sentinel: use the house linear pedestal. Callers in display space pass
# a target of their own, or None to equalize to the panels' OWN MEAN sky
# (space-agnostic: no assumption about absolute level).
_UNSET_PEDESTAL = object()
_ADU_FULL_SCALE = 65535.0   # a raw 16-bit light's saturation


_LEVEL_MIN_OVERLAP_PX = 256  # a few-pixel seam cannot carry a level constraint

# Robust pairwise-offset clip, mFitplane-shaped: reject beyond 3*MAD of the
# median difference before taking the final median.
_LEVEL_REJECT_SIGMA = 3.0


def _overlap_offsets(bboxes, masks, crops) -> list[tuple]:
    """Pairwise sky offsets from the overlap DIFFERENCES.

    One `(i, j, d_ij, npix)` per overlapping pair, where `d_ij` is the
    sigma-clipped median of `crop_i - crop_j` over the both-covered pixels.
    In a difference of photometrically-matched panels the NEBULOSITY
    CANCELS, so `d_ij` measures the relative sky offset directly — the
    estimator an absolute per-panel quantile cannot be in a field with no
    empty sky (JWST skymatch's "match" method; Montage mBgModel's input).
    Measured 2026-08-31 on the 24-panel gamma Cyg mosaic: per-panel
    quantile skies agreed to 1e-4 of the pedestal while the true offsets
    spanned 1.3e-4 — the visible S/O hue quilt."""
    out = []
    n = len(crops)
    for i in range(n):
        if bboxes[i] is None:
            continue
        for j in range(i):
            if bboxes[j] is None:
                continue
            region = _intersect(bboxes[i], bboxes[j])
            if region is None:
                continue
            both = _sub(masks[i], bboxes[i], region) & _sub(masks[j], bboxes[j], region)
            npix = int(both.sum())
            if npix < _LEVEL_MIN_OVERLAP_PX:
                continue
            d = (_sub(crops[i], bboxes[i], region)[both]
                 - _sub(crops[j], bboxes[j], region)[both]).astype(np.float64)
            med = float(np.median(d))
            mad = float(np.median(np.abs(d - med))) * 1.4826
            keep = np.abs(d - med) <= _LEVEL_REJECT_SIGMA * max(mad, 1e-12)
            out.append((i, j, float(np.median(d[keep])), npix))
    return out


def _solve_level_offsets(n: int, pairs: list[tuple]) -> np.ndarray:
    """One EXACT weighted least-squares over every pairwise offset.

    Rows `o_i - o_j = d_ij` (weight sqrt(npix)) plus a mean-zero gauge row;
    the panel grid's adjacency loops force closure, so chain drift cannot
    accumulate. Solved directly (24 unknowns is microseconds) — NOT via
    mBgModel: its iterative half-strength relaxation left ~half the
    correction unconverged on the same measured pairs (2026-08-31)."""
    from mosaics.merging.level_solve import solve_level_offsets
    return solve_level_offsets(n, pairs)


def _apply_level_offsets(crops, masks, offsets, pedestal) -> "float | None":
    """Subtract each panel's solved offset, then ONE global shift to the
    shared pedestal — identical for every panel, so it cannot introduce a
    seam."""
    from stacks.post_processing.background_pedestal import DEFAULT_PEDESTAL
    tiny = np.finfo(np.float32).tiny
    for crop, mask, off in zip(crops, masks, offsets):
        if crop is None or off == 0.0:
            continue
        np.subtract(crop, np.float32(off), out=crop, where=mask)
    skies = [None if c is None else _panel_sky(c[m])
             for c, m in zip(crops, masks)]
    measured = [v for v in skies if v is not None]
    if not measured:
        return None
    if pedestal is _UNSET_PEDESTAL:
        target = float(DEFAULT_PEDESTAL)
    elif pedestal is None:
        target = float(np.mean(measured))
    else:
        target = float(pedestal)
    shift = np.float32(target - float(np.mean(measured)))
    for crop, mask in zip(crops, masks):
        if crop is None:
            continue
        if shift != 0:
            np.add(crop, shift, out=crop, where=mask)
        np.maximum(crop, tiny, out=crop, where=mask)
    print("### merge level solve: %d panels, offsets spread %.6f, "
          "global shift %.6f, target %.6f" % (
              len(measured), float(np.max(offsets) - np.min(offsets)),
              float(shift), target))
    return target


def _panel_sky(values: np.ndarray) -> "float | None":
    """The panel's sky level — lower-quantile median, the same estimator
    `stacks.post_processing.background_pedestal` uses (single source of
    truth for the cut). Verified 2026-08-22 against a histogram-mode sky:
    bias 3.8e-5, uncorrelated with nebula fraction, so it is not scene
    contaminated even on wall-to-wall emission fields."""
    from stacks.post_processing.background_pedestal import _SKY_QUANTILE
    if values.size == 0:
        return None
    sky = values[values <= np.quantile(values, _SKY_QUANTILE)]
    return float(np.median(sky)) if sky.size else None


def _equalize_sky(crops, masks, pedestal) -> "float | None":
    """Shift every panel in place so its own sky sits on the shared target.

    `pedestal` is the house linear pedestal by default, an explicit float
    for callers in another space, or None to equalize to the panels' OWN
    MEAN sky (which assumes nothing about absolute level, so it works in
    display space too)."""
    from stacks.post_processing.background_pedestal import DEFAULT_PEDESTAL
    skies = []
    for crop, mask in zip(crops, masks):
        skies.append(None if crop is None else _panel_sky(crop[mask]))
    measured = [v for v in skies if v is not None]
    if not measured:
        return None
    if pedestal is _UNSET_PEDESTAL:
        target = float(DEFAULT_PEDESTAL)
    elif pedestal is None:
        target = float(np.mean(measured))
    else:
        target = float(pedestal)
    tiny = np.finfo(np.float32).tiny
    for crop, mask, sky in zip(crops, masks, skies):
        if crop is None or sky is None:
            continue
        shift = np.float32(target - sky)
        if shift == 0:
            continue
        np.add(crop, shift, out=crop, where=mask)
        # Hold the mask: a pixel pushed to <= 0 would read as no-data.
        np.maximum(crop, tiny, out=crop, where=mask)
    print("### merge sky equalization: target %.6f, panel skies %.6f..%.6f "
          "(spread %.6f)" % (target, min(measured), max(measured),
                             max(measured) - min(measured)))
    return target


def _blend_many(paths: list[Path], level_solve: "str | None" = None,
                pedestal: "float | None" = _UNSET_PEDESTAL,
                depth_weighted: bool = False) -> np.ndarray:
    """Variadic distance-weighted blend (gains arrive pre-calibrated).

    `depth_weighted` multiplies each panel's distance weight by its inverse
    noise variance (`_inverse_variance`), so where panels of unequal depth
    overlap the deeper one dominates instead of a 50/50 average that is
    NOISIER than the deep panel alone (2 h over 10 h: 1.5x its variance,
    2026-10-07). With equal panels it is the plain mean, sqrt(2) deeper.
    Geometry-only weights remain the default for the stars layer, whose
    zero background has no noise to measure.

    Streams every panel off disk one at a time and works per-panel in its
    on-disk BOUNDING BOX, not the full mosaic canvas: a placed panel fills only
    a fraction of the canvas (the rest is `place_image`-zero), so its retained
    mask, its distance-transform weight, and its accumulation slice are all
    bbox-sized — only the numerator/denominator accumulators are full-canvas.
    The weighting is the Euclidean distance transform of each panel's mask, the
    flux scaling is a global log-space least-squares over the overlap graph, and
    the value is `Σ wᵢ · scaleᵢ · imageᵢ / Σ wᵢ` — a single global feather, no
    order dependence.

    Bit-identical to the full-canvas formulation: every term it skips outside a
    panel's bbox is exactly `+0` (there `imageᵢ` and its weight are both zero),
    and `_edt_weight` reproduces the canvas EDT within the bbox. See
    `test_oss_blend.py`.

    SKY EQUALIZATION runs on matched merges: each panel is shifted so
    its OWN sky median lands on the shared pedestal. Panel masters all
    leave post-processing at sky = 0.005 exactly, but BXT (deconvolution,
    now `bxt_channel`) and SXT each shift a panel's background by a
    content-dependent amount — measured 2026-08-22 (BXT era): masters
    spread 0.000000, BXT outputs spread 0.001247, merged panels 0.000120,
    which IS the visible quilt. Equalizing here needs no solve, no gauge,
    and no overlap gates: it is the direct form of "every panel's sky is
    the same number", and it happens at the point the panels actually
    meet.

    Panels are stored **bbox-on-disk** (bbox pixels + offset in the header; see
    `panel_io`), so each is read exactly ONCE as a small array — no full-canvas
    decompression, and the three passes reuse the retained crop instead of the
    old 3x re-read. The tight non-zero bbox is recomputed from the small array
    (it lies inside the stored bbox rectangle), keeping the math identical.

    `level_solve` picks the panel-level matching mode:
      * "overlap"  — constants derived from the pairwise overlap differences
        (one exact global least-squares; see `_overlap_offsets`/
        `_solve_level_offsets`), then one uniform shift to the pedestal.
        Plane fitting was measured and retired: on nebulosity-filled fields
        the tilt is fit to scene, not sky (A/B 2026-08-26, 8-15x worse
        block-sky-spread).
      * "quantile" — the absolute lower-quantile sky equalization
        (`_equalize_sky`); scene-biased in fields with no empty sky, kept
        for the Subframe previews.
      * None       — no matching at all. For the STARS layer: additive
        offsets cancel exactly in `stars = channel − starless`, and star
        flux is multiplicatively calibrated upstream (SPFC), so panels
        already agree; any equalization here would inject per-panel lifts
        into a zero-background image (its "sky" estimator would read star
        wings, not sky).
    """
    if len(paths) == 1:
        # No equalization: with one panel there is nothing to agree WITH,
        # and single-panel passthrough is a contract (test_oss_blend).
        arr, (x0, y0), (cw, ch) = read_placed_panel(paths[0])
        out = np.zeros((ch, cw), dtype=np.float32)
        h, w = arr.shape
        out[y0:y0 + h, x0:x0 + w] = arr
        return out

    n = len(paths)

    # Read every panel ONCE (small bbox array + its canvas offset), and retain
    # the tight non-zero crop + mask for all three passes. `bbox` is in canvas
    # coordinates (offset + tight bbox within the stored array); `canvas_shape`
    # comes from the header, not from an `img.shape`.
    canvas_shape: tuple[int, int] | None = None
    bboxes: list[tuple[int, int, int, int] | None] = []
    masks: list[np.ndarray] = []
    crops: list[np.ndarray | None] = []
    for p in paths:
        arr, (x0, y0), (cw, ch) = read_placed_panel(p)
        canvas_shape = (ch, cw)
        tight = _panel_bbox(arr > 0)
        if tight is None:
            bboxes.append(None)
            masks.append(np.zeros((0, 0), dtype=bool))
            crops.append(None)
            continue
        tr0, tr1, tc0, tc1 = tight
        crop = np.ascontiguousarray(arr[tr0:tr1, tc0:tc1], dtype=np.float32)  # writable for pass 3
        bboxes.append((y0 + tr0, y0 + tr1, x0 + tc0, x0 + tc1))
        masks.append(crop > 0)
        crops.append(crop)

    if level_solve == "overlap":
        pairs = _overlap_offsets(bboxes, masks, crops)
        offsets = _solve_level_offsets(n, pairs)
        _apply_level_offsets(crops, masks, offsets, pedestal)
    elif level_solve == "quantile":
        _equalize_sky(crops, masks, pedestal)
    else:
        assert level_solve is None, f"unknown level_solve mode {level_solve!r}"

    inv_var = _inverse_variance(crops, masks) if depth_weighted else [1.0] * n
    if depth_weighted:
        log.info("OSS merge depth weights", inverse_variance=[round(v, 3) for v in inv_var])

    # Pass 3: weighted sum, accumulated per-panel into its bbox slice.
    # (No per-panel flux scaling: gains are calibrated upstream against
    # Gaia by post_stack_calibrate — see the deletion note at the top.)
    numer = np.zeros(canvas_shape, dtype=np.float32)
    denom = np.zeros(canvas_shape, dtype=np.float32)
    for i in range(n):
        if bboxes[i] is None:
            continue
        r0, r1, c0, c1 = bboxes[i]
        weight = _edt_weight(masks[i], bboxes[i], canvas_shape)
        if inv_var[i] != 1.0:
            weight *= np.float32(inv_var[i])
        sub = crops[i]  # the retained crop, mutated in place (last use)
        sub *= weight
        numer[r0:r1, c0:c1] += sub
        denom[r0:r1, c0:c1] += weight

    denom[denom == 0] = 1.0
    return numer / denom


def _merge_layer(out, filter: str, inputs, level_solve, skyver,
                 layer, depth_weighted=False):
    """Shared body of the two layer merges: variadic distance-weighted blend
    (`_blend_many`) + the house TIFF write with the layer's SKYVER stamp.
    `skyver` is the merge-time matching version header — bumping a caller's
    literal is the deliberate re-key for changes to the (unhashed) blend
    helpers' matching maths. `layer` stamps which PHYSICAL layer this is
    ("starless"/"stars"), which until now was legible only from the op's
    output directory name plus SKYVER."""
    assert len(inputs) >= 1
    paths = [inp.filename.resolve(strict=True) for inp in inputs]
    result = _blend_many(paths, level_solve=level_solve, depth_weighted=depth_weighted)

    write_tiff_image(Path(out), result.astype(np.float32),
                     {"FILTER": filter,
                      "STBLHASH": hash_list([inp.stable_hash for inp in inputs]),
                      "PIPELINE": "AstroFlow-OSS",
                      "LAYER": layer,
                      "SKYVER": skyver})

    new_count = sum(_panel_count(inp) for inp in inputs)
    log.info("OSS merge complete", panel_count=new_count, n_inputs=len(inputs))
    return Produced(out, sections={PANEL_COUNT: new_count})


@node(display="Mosaic Starless", final=True, auto_memory=True)  # mosaic projection — non-square pixels
def merge_images_oss(
    mosaic: Mosaic, filter: str, *inputs: ImageMetadata, out: TiffFilePath,
) -> Produced:
    """Variadic distance-weighted OSS blend of the placed STARLESS panels.

    This single variadic call IS the finished per-filter starless mosaic
    channel (`display="Mosaic Starless"`, `final=True`; multi-panel
    mosaics append `polish_background`, which then carries the tap).
    Star removal runs pre-placement (`sxt_channel` — SXT is measured
    spotless on the native grid and leaves halo donuts on the resampled
    canvas, 2026-09-02), so every placed input here is a starless layer:
    the overlap level solve reads pure sky + nebulosity, with no star
    flux biasing the pair statistics.

    Pipeline-agnostic: every OSS-merging pipeline (WBPP or Siril stacks)
    calls this with their own placed inputs. One call per mosaic per
    filter; no row/strip decomposition (see `_blend_many` for why)."""
    return _merge_layer(out, filter, inputs,
                        level_solve="overlap", skyver="overlap-3-ivw",
                        layer="starless", depth_weighted=True)


@node(display="Mosaic Stars", final=True, auto_memory=True)  # mosaic projection — non-square pixels
def merge_images_stars(
    mosaic: Mosaic, filter: str, *inputs: ImageMetadata, out: TiffFilePath,
) -> Produced:
    """The STARS-layer twin of `merge_images_oss`: same distance-weighted
    blend, NO level matching (`level_solve=None`). A stars layer is
    `channel − starless` — additive panel offsets cancel in the
    subtraction, and star flux is multiplicatively calibrated upstream
    (SPFC against Gaia) — so the panels already agree; equalizing on a
    zero-background image would read star wings as "sky" and inject
    per-panel lifts. With-stars views are never baked: the viewer's
    finishes add `starless + star_s·stars` live, and placement/merging
    are linear, so the layer sum reproduces the composite."""
    return _merge_layer(out, filter, inputs,
                        level_solve=None, skyver="stars-none-1",
                        layer="stars")


@ray.remote(num_cpus=0.1)
def merge_scope(mosaic, filter, viewbox, config, targets, *corrected,
                dry_run=False):
    """One filter's `{N× place_image → merge_images_oss}` as a dynamic
    subgraph with `delete_intermediates=True`: the placements — multi-GB
    views whose only consumer is this merge — are deleted when the scope
    concludes; the merge (the "Mosaic" deliverable) is the terminal and
    persists. The caller presents this builder AS the merge via
    `present_terminal`, so the tap snapshot, version-supersede map, and
    path resolution see the deliverable exactly as if it were inline.

    `targets` is a tuple aligned with `*corrected` (the BXT'd channels —
    GPU outputs, deliberately OUTSIDE the scope so they persist and
    cache-hit on incremental runs)."""
    del dry_run
    from dependency_graph.classes import Graph
    from dependency_graph.tracing import trace
    from mosaics.placement.astroflow import place_image

    # RUNTIME FOLD CONFESSION: print each resolved input's content hash and
    # the resulting merge identity. The R-channel anomaly (2026-08-11 —
    # landed 423e2666 vs predicted 01b99d25 with every data/code check
    # equal) could not be reconstructed from snapshots; this line makes the
    # next divergence name its own input instead.
    print("### merge_scope fold %s/%s: %s" % (
        getattr(mosaic, "stable_hash", "?")[:12], filter,
        " ".join(str(getattr(c, "content_hash", "?"))[:16] for c in corrected)))

    def _body():
        placed = [
            place_image(mosaic, target, filter, viewbox, config, channel)
            for target, channel in zip(targets, corrected)
        ]
        return merge_images_oss(mosaic, filter, *placed)

    return Graph(trace(_body))


@ray.remote(num_cpus=0.1)
def merge_stars_scope(mosaic, filter, viewbox, config, targets, *stars,
                      dry_run=False):
    """The STARS-layer twin of `merge_scope`:
    `{N× place_image → merge_images_stars}`, placements ephemeral, the
    stars mosaic the presented terminal. Two star-layer specifics:

    * NO edge erosion (`edge_erosion_px=0`): `_erode_edges` reads
      `image != 0` as the footprint, and a stars layer is mostly exact
      zeros — erosion would shave every star blob (and erase stars
      smaller than the erosion radius outright). The interpolation
      rolloff erosion exists to cut lives at panel borders, where the
      feathered overlap draws the same stars from neighbouring panels.
    * NO polish ever: GraXpert has no business modelling a background
      that is identically zero.
    """
    del dry_run
    import dataclasses

    from dependency_graph.classes import Graph
    from dependency_graph.tracing import trace
    from mosaics.placement.astroflow import place_image

    print("### merge_stars_scope fold %s/%s: %s" % (
        getattr(mosaic, "stable_hash", "?")[:12], filter,
        " ".join(str(getattr(c, "content_hash", "?"))[:16] for c in stars)))

    star_config = dataclasses.replace(config, edge_erosion_px=0)

    def _body():
        placed = [
            place_image(mosaic, target, filter, viewbox, star_config, channel)
            for target, channel in zip(targets, stars)
        ]
        return merge_images_stars(mosaic, filter, *placed)

    return Graph(trace(_body))


@node(
    display="Mosaic",
    resources={"fastest_storage": 0.001},
)
def merge_images_subframe(
    mosaic: Mosaic, filter: str, *inputs, out: TiffFilePath,
) -> Produced:
    """The Subframe preview pipeline's mosaic merge — the SAME distance-weighted
    blend as `merge_images_oss` (`_blend_many`), emitting the preview
    pipeline's per-filter mosaic (`display="Mosaic"`: the linear image the
    channel finishes read, the same name the PI pipeline's post-processed
    channel carries) instead of the final Mosaic Starless deliverable.

    Subframe panels are tiny (downscaled), so this variadic single-call blend
    over every placed panel is near-instant. One call per (mosaic, filter) →
    one output file in `merge_images_subframe/`."""
    assert len(inputs) >= 1
    # `*inputs` MUST flow into the call_hash so distinct panel sets get distinct
    # filenames (no writer races / half-written TIFFs).
    paths = [i.filename.resolve(strict=True) for i in inputs]
    # The panels are raw 16-bit lights (ADU), but the finishes read the 0-1
    # linear gauge (sky at the house pedestal, saturation at 1.0) and clamp
    # their stretch to it — in ADU every pixel sat above white (2026-10-07).
    # Equalise the sky to the pedestal in ADU, then scale onto the gauge.
    from stacks.post_processing.background_pedestal import DEFAULT_PEDESTAL
    acc = _blend_many(paths, level_solve="quantile", pedestal=float(DEFAULT_PEDESTAL) * _ADU_FULL_SCALE)
    acc /= _ADU_FULL_SCALE
    assert not np.isnan(np.min(acc))

    stable_hash = hash_list([i.stable_hash for i in inputs])
    write_tiff_image(Path(out), acc, {"FILTER": filter, "STBLHASH": stable_hash})

    new_count = sum(_panel_count(i) for i in inputs)
    log.info("Merge images (subframe) complete", panel_count=new_count, input_count=len(inputs))
    return Produced(out, sections={PANEL_COUNT: new_count})
