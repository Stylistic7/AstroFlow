"""OSS mosaic merging: distance-weighted N-way blend with flux equalization.

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
from file_io.types import TiffFilePath, WithSidecar
from hashing import hash_list
from metadata.classes import ImageMetadata
from mosaics.merging.pixinsight import (
    MergedImage, _panel_count, _set_panel_count,
)
# MosaicLike bound as `Mosaic`: widens beartype to accept a Target acting as
# its own single-panel mosaic WITHOUT touching the `mosaic: Mosaic`
# annotations — those serialize into the op code hash, so editing them would
# re-key every op in this module (see mosaics/types.py).
from mosaics.types import MosaicLike as Mosaic

log = structlog.get_logger()


def _solve_flux_scales(n: int, overlap_means: dict) -> np.ndarray:
    """Per-image multiplicative scale factors minimizing pairwise overlap
    discrepancies. For every overlapping pair (i, j) with overlap means
    (mi, mj), we want `mi · scale_i ≈ mj · scale_j`. Solve in log space
    with image 0 anchored at scale=1; clamp to [0.5, 2.0] to match the
    sanity guard the old pairwise blend used."""
    rows = [
        (i, j, np.log(mj / mi))
        for (i, j), (mi, mj) in overlap_means.items()
        if mi and mj and mi > 0 and mj > 0
    ]
    if not rows:
        return np.ones(n, dtype=np.float32)
    A = np.zeros((len(rows) + 1, n), dtype=np.float32)
    b = np.zeros(len(rows) + 1, dtype=np.float32)
    for r, (i, j, k) in enumerate(rows):
        A[r, i] = 1.0
        A[r, j] = -1.0
        b[r] = k
    A[-1, 0] = 1.0
    log_scales, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.clip(np.exp(log_scales), 0.5, 2.0).astype(np.float32)


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


_LEVEL_MIN_OVERLAP_PX = 256  # a few-pixel seam cannot carry a level constraint

# mFitplane discipline: iteratively reject pixels beyond 2*rms of the current
# plane fit and refit, until stable (Montage/mFitplane.c does exactly this).
_PLANE_REJECT_SIGMA = 2.0
_PLANE_REJECT_ITERS = 5


def _fit_plane(x, y, z):
    """Least-squares z = a*x + b*y + c via 3x3 normal equations (no big
    design matrix), with mFitplane's iterative 2*rms outlier rejection.
    Returns (a, b, c, rms, n_used)."""
    keep = np.ones(z.shape, dtype=bool)
    a = b = c = 0.0
    rms = 0.0
    for _ in range(_PLANE_REJECT_ITERS):
        xs, ys, zs = x[keep], y[keep], z[keep]
        n = xs.size
        if n < 16:
            break
        M = np.array([
            [np.dot(xs, xs), np.dot(xs, ys), xs.sum()],
            [np.dot(xs, ys), np.dot(ys, ys), ys.sum()],
            [xs.sum(),       ys.sum(),       float(n)],
        ])
        v = np.array([np.dot(xs, zs), np.dot(ys, zs), zs.sum()])
        try:
            a, b, c = np.linalg.solve(M, v)
        except np.linalg.LinAlgError:
            break
        resid = z - (a * x + b * y + c)
        rms = float(np.sqrt(np.mean(resid[keep] ** 2)))
        new_keep = np.abs(resid) <= _PLANE_REJECT_SIGMA * max(rms, 1e-12)
        if new_keep.sum() == keep.sum():
            keep = new_keep
            break
        keep = new_keep
    return float(a), float(b), float(c), max(rms, 1e-9), int(keep.sum())


def _level_fit_rows(bboxes, masks, crops) -> list[dict]:
    """Montage-shaped PLANE fit rows from the blend's retained crops —
    the first-degree ("LNC") generalisation of the original constant fits.

    One row per overlapping pair: `diff = a*x + b*y + c` fitted to
    (plus − minus) over the both-covered pixels with mFitplane's iterative
    2*rms rejection, in ZERO-BASED CANVAS coordinates (crpix1 = crpix2 = 0,
    so x = canvas column, y = canvas row — the same frame the boxes are
    written in and the apply step subtracts in; the ground-truth lab in
    test_background.py pins the round trip). Boxes are the REAL overlap
    rectangles: mBgModel silently zeroes every correction when handed
    degenerate boxes (the 2026-08-19 lab regression)."""
    rows = []
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
            r0, r1, c0, c1 = region
            yy, xx = np.nonzero(both)
            x = (xx + c0).astype(np.float64)
            y = (yy + r0).astype(np.float64)
            z = (_sub(crops[j], bboxes[j], region)[both]
                 - _sub(crops[i], bboxes[i], region)[both]).astype(np.float64)
            a, b, c, rms, used = _fit_plane(x, y, z)
            rows.append({
                "plus": j, "minus": i, "a": a, "b": b, "c": c,
                "npixel": used, "rms": rms,
                "xmin": c0, "xmax": c1, "ymin": r0, "ymax": r1,
                "xcenter": (c0 + c1) / 2.0, "ycenter": (r0 + r1) / 2.0,
                "boxwidth": float(c1 - c0), "boxheight": float(r1 - r0),
            })
    return rows


def _overlap_means(bboxes, masks, crops):
    """Pass 1+2 of the blend: i-side and j-side overlap means per pair."""
    n = len(crops)
    means = {}
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
            if not both.any():
                continue
            mi = float(_sub(crops[i], bboxes[i], region)[both].mean())
            mj = float(_sub(crops[j], bboxes[j], region)[both].mean())
            means[(j, i)] = (mj, mi)
    return means


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


def _signal_means(overlap_means, sky_target):
    """Overlap means with the shared pedestal removed, for the flux solve.

    Floored just above zero: `_solve_flux_scales` works in log space, and a
    faint overlap whose mean sits at or below the pedestal would otherwise
    produce a non-finite constraint."""
    if sky_target is None:
        return overlap_means
    def strip(v):
        return None if v is None else max(v - sky_target, 1e-9)
    return {k: (strip(a), strip(b)) for k, (a, b) in overlap_means.items()}


def _blend_many(paths: list[Path], level_match: bool = False,
                pedestal: "float | None" = _UNSET_PEDESTAL) -> np.ndarray:
    """Variadic distance-weighted blend with least-squares flux solve.

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

    SKY EQUALIZATION runs first, on every merge: each panel is shifted so
    its OWN sky median lands on the shared pedestal. Panel masters all
    leave post-processing at sky = 0.005 exactly, but `bxt_correct_only`
    (deconvolution) then shifts each panel's background by a
    content-dependent amount — measured 2026-08-22: masters spread
    0.000000, BXT outputs spread 0.001247, merged panels 0.000120, which
    IS the visible quilt. Equalizing here needs no solve, no gauge, and no
    overlap gates: it is the direct form of "every panel's sky is the same
    number", and it happens at the point the panels actually meet.

    Panels are stored **bbox-on-disk** (bbox pixels + offset in the header; see
    `panel_io`), so each is read exactly ONCE as a small array — no full-canvas
    decompression, and the three passes reuse the retained crop instead of the
    old 3x re-read. The tight non-zero bbox is recomputed from the small array
    (it lies inside the stored bbox rectangle), keeping the math identical.

    `level_match=True` fuses Montage background rectification into the same
    pass — rectify, then feather, FIRST DEGREE ("LNC"): a plane per overlap
    difference (mFitplane's model + 2*rms rejection), one global two-phase
    `mBgModel` solve (levels first, then slopes; its area/linear/2-sigma
    overlap gates active), and in-place subtraction of each panel's solved
    plane before the flux solve sees the corrected sky. No extra I/O, no
    intermediate artifacts.
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

    sky_target = _equalize_sky(crops, masks, pedestal)

    overlap_means = _overlap_means(bboxes, masks, crops)

    if level_match:
        from mosaics.merging.background import solve_planes
        planes = solve_planes(_level_fit_rows(bboxes, masks, crops),
                              n, canvas_shape)
        # Rectify-then-feather, first degree: subtract each panel's solved
        # plane over its crop's canvas coordinates (no-data zeros stay
        # zero), then RECOMPUTE the overlap means so the flux solve sees
        # the corrected sky.
        for i in range(n):
            if crops[i] is None:
                continue
            pa, pb, pc = planes[i]
            if pa == 0.0 and pb == 0.0 and pc == 0.0:
                continue
            r0, r1, c0, c1 = bboxes[i]
            ramp = (pa * np.arange(c0, c1, dtype=np.float32)[None, :]
                    + pb * np.arange(r0, r1, dtype=np.float32)[:, None]
                    + np.float32(pc))
            np.subtract(crops[i], ramp, out=crops[i], where=masks[i])
        overlap_means = _overlap_means(bboxes, masks, crops)

    # FLUX SCALING IS FOR FLUX, NOT FOR THE PEDESTAL. Both the solve and
    # the application work on signal ABOVE the shared sky:
    #
    #   solve   on (mean - sky), so the scales match real signal rather
    #           than signal+pedestal;
    #   apply   as (panel - sky)*scale + sky, so a panel's sky comes out
    #           exactly where the pedestal put it.
    #
    # Multiplying the whole panel instead moves its sky by (scale-1)*sky.
    # Measured 2026-08-22: two panels with IDENTICAL 0.005 skies differing
    # only in signal gain came out of the blend 6e-4 apart — five times the
    # production quilt, and the reason every additive fix upstream of this
    # point (level constants, planes, SPFC gain, merge-time equalization)
    # was silently undone here.
    scales = _solve_flux_scales(n, _signal_means(overlap_means, sky_target))

    # Pass 3: weighted sum, accumulated per-panel into its bbox slice.
    numer = np.zeros(canvas_shape, dtype=np.float32)
    denom = np.zeros(canvas_shape, dtype=np.float32)
    for i in range(n):
        if bboxes[i] is None:
            continue
        r0, r1, c0, c1 = bboxes[i]
        weight = _edt_weight(masks[i], bboxes[i], canvas_shape)
        sub = crops[i]  # the retained crop, mutated in place (last use)
        if sky_target is None:
            sub *= scales[i]
        else:
            # (sub - t)*s + t, masked so no-data zeros stay exactly zero.
            np.multiply(sub, scales[i], out=sub, where=masks[i])
            np.add(sub, np.float32(sky_target * (1.0 - scales[i])),
                   out=sub, where=masks[i])
        sub *= weight
        numer[r0:r1, c0:c1] += sub
        denom[r0:r1, c0:c1] += weight

    denom[denom == 0] = 1.0
    return numer / denom


@node(display="Mosaic", final=True, auto_memory=True, plate_solve=False)  # mosaic projection — non-square pixels
def merge_images_oss(
    mosaic: Mosaic, filter: str, *inputs: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """Variadic distance-weighted OSS blend across every input panel.

    This single variadic call IS the finished per-filter mosaic channel —
    the OSS deliverable (`display="Mosaic"`, `final=True`). Unlike the PI
    pairwise path (intermediate `merge_images` → separate flatten/finalize
    `post_process_mosaic_color_channel`), the OSS blend needs no separate
    finalize op, so there is no `store_*` step after it.

    Pipeline-agnostic: PI Master and PI Downscaled hybrid variants both
    call this with their own placed inputs. One call per mosaic per
    filter; no row/strip decomposition (see `_blend_many` for why)."""
    assert len(inputs) >= 1
    output_filename = merge_images_oss.filename_fn(mosaic, filter, *inputs)
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)

    paths = [inp.filename.resolve(strict=True) for inp in inputs]
    result = _blend_many(paths)

    write_tiff_image(Path(output_filename), result.astype(np.float32),
                     {"FILTER": filter,
                      "STBLHASH": hash_list([inp.stable_hash for inp in inputs]),
                      "PIPELINE": "AstroFlow-OSS",
                      # Merge-time sky equalization version. Bumping this
                      # line is the deliberate re-key for changes to the
                      # matching maths in the (unhashed) blend helpers.
                      "SKYVER": "pedestal-2"})

    new_count = sum(_panel_count(inp) for inp in inputs)
    _set_panel_count(output_filename, new_count)

    log.info("OSS merge complete", output=str(output_filename),
             panel_count=new_count, n_inputs=len(inputs))
    return WithSidecar(output_filename)


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


@node(
    display="Merged",
    resources={"fastest_storage": 0.001},
    plate_solve=False,  # mosaic projection — non-square pixels, can't solve
)
def merge_images_subframe(
    mosaic: Mosaic, filter: str, *inputs,
) -> WithSidecar[TiffFilePath]:
    """The Subframe preview pipeline's mosaic merge — the SAME distance-weighted
    blend as `merge_images_oss` (`_blend_many`), just emitting the preview tap
    (`display="Merged"`, non-final, TIFF) instead of the final Mosaic deliverable.

    Subframe panels are tiny (downscaled), so this variadic single-call blend
    over every placed panel is near-instant. One call per (mosaic, filter) →
    one output file in `merge_images_subframe/`."""
    assert len(inputs) >= 1
    # `*inputs` MUST flow into the call_hash so distinct panel sets get distinct
    # filenames (no writer races / half-written TIFFs).
    output_filename = merge_images_subframe.filename_fn(mosaic, filter, *inputs)
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)

    paths = [i.filename.resolve(strict=True) for i in inputs]
    acc = _blend_many(paths)
    assert not np.isnan(np.min(acc))

    stable_hash = hash_list([i.stable_hash for i in inputs])
    write_tiff_image(Path(output_filename), acc, {"FILTER": filter, "STBLHASH": stable_hash})

    new_count = sum(_panel_count(i) for i in inputs)
    _set_panel_count(output_filename, new_count)
    log.info("Merge images (subframe) complete", output=output_filename,
             panel_count=new_count, input_count=len(inputs))
    return WithSidecar(output_filename)
