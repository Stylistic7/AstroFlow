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
import structlog
from pathlib import Path

import astropy.io.fits as pyfits
import edt

from decorators import node
from file_io.generic import read_image
from mosaics.placement.panel_io import read_placed_panel
from file_io.types import FitsFilePath, WithSidecar
from hashing import hash_list
from metadata.classes import ImageMetadata
from mosaics.merging.pixinsight import (
    MergedImage, _panel_count, _set_panel_count,
)
from mosaics.types import Mosaic

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


def _blend_many(paths: list[Path]) -> np.ndarray:
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

    Panels are stored **bbox-on-disk** (bbox pixels + offset in the header; see
    `panel_io`), so each is read exactly ONCE as a small array — no full-canvas
    decompression, and the three passes reuse the retained crop instead of the
    old 3x re-read. The tight non-zero bbox is recomputed from the small array
    (it lies inside the stored bbox rectangle), keeping the math identical.
    """
    if len(paths) == 1:
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

    # Pass 1: i-side (higher-index) overlap mean of every overlapping pair.
    overlap_means: dict[tuple[int, int], tuple[float | None, float | None]] = {}
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
            overlap_means[(j, i)] = (None, mi)  # j-side filled in pass 2

    # Pass 2: fill in the j (lower-index) side mean — from the retained crop.
    for (a, b) in list(overlap_means):
        _, mi = overlap_means[(a, b)]
        region = _intersect(bboxes[a], bboxes[b])
        both = _sub(masks[a], bboxes[a], region) & _sub(masks[b], bboxes[b], region)
        mj = float(_sub(crops[a], bboxes[a], region)[both].mean()) if both.any() else 0.0
        overlap_means[(a, b)] = (mj, mi)

    scales = _solve_flux_scales(n, overlap_means)

    # Pass 3: weighted sum, accumulated per-panel into its bbox slice.
    numer = np.zeros(canvas_shape, dtype=np.float32)
    denom = np.zeros(canvas_shape, dtype=np.float32)
    for i in range(n):
        if bboxes[i] is None:
            continue
        r0, r1, c0, c1 = bboxes[i]
        weight = _edt_weight(masks[i], bboxes[i], canvas_shape)
        sub = crops[i]  # the retained crop, mutated in place (last use)
        sub *= scales[i]
        sub *= weight
        numer[r0:r1, c0:c1] += sub
        denom[r0:r1, c0:c1] += weight

    denom[denom == 0] = 1.0
    return numer / denom


@node(display="Mosaic", final=True, auto_memory=True, plate_solve=False)  # mosaic projection — non-square pixels
def merge_images_oss(
    mosaic: Mosaic, filter: str, *inputs: ImageMetadata,
) -> WithSidecar[FitsFilePath]:
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

    header = pyfits.Header()
    header["FILTER"] = filter
    header["STBLHASH"] = hash_list([inp.stable_hash for inp in inputs])
    header["PIPELINE"] = "AstroFlow-OSS"
    pyfits.PrimaryHDU(data=result.astype(np.float32), header=header).writeto(
        str(output_filename), overwrite=True,
    )

    new_count = sum(_panel_count(inp) for inp in inputs)
    _set_panel_count(output_filename, new_count)

    log.info("OSS merge complete", output=str(output_filename),
             panel_count=new_count, n_inputs=len(inputs))
    return WithSidecar(output_filename)
