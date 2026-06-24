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


def _blend_many(paths: list[Path]) -> np.ndarray:
    """Variadic distance-weighted blend with least-squares flux solve.

    Streams every panel off disk one at a time; never holds more than
    one full-res image in memory alongside the accumulators. The
    weighting is `distance_transform_edt` of each panel's mask, the
    flux scaling comes from a global log-space least-squares over the
    overlap graph, and the final value is `Σ wᵢ · scaleᵢ · imageᵢ /
    Σ wᵢ`. Single global feather, no order dependence.
    """
    if len(paths) == 1:
        return read_image(paths[0]).astype(np.float32)

    n = len(paths)

    # Pass 1: collect each panel's mask, plus the i-side of every pairwise
    # overlap mean. Each image is loaded once here. Storing only one mask
    # at a time would force a second pass per pair to recover image_i's
    # mean over its overlap with each image_j, so we keep all masks
    # (bool, ~1/32 the size of float32) and the running list of partial
    # overlap means.
    masks: list[np.ndarray] = []
    overlap_means: dict[tuple[int, int], tuple[float, float | None]] = {}
    for i, p in enumerate(paths):
        img = read_image(p).astype(np.float32)
        mask = img > 0
        for j, mask_j in enumerate(masks):
            both = mask & mask_j
            if not both.any():
                continue
            mi = float(img[both].mean())
            overlap_means[(j, i)] = (None, mi)  # j-side filled in pass 2
        masks.append(mask)
        # img drops out of scope; numpy frees the buffer.

    # Pass 2: fill in the j-side mean for each pair. We re-load each
    # image once and compute its mean over the overlap masks where it's
    # the j (lower-index) side.
    for j, p in enumerate(paths):
        img = read_image(p).astype(np.float32)
        for (a, b), (_, mi) in list(overlap_means.items()):
            if a != j:
                continue
            both = masks[a] & masks[b]
            mj = float(img[both].mean()) if both.any() else 0.0
            overlap_means[(a, b)] = (mj, mi)

    scales = _solve_flux_scales(n, overlap_means)

    # Pass 3: streaming weighted sum.
    shape = masks[0].shape
    numer = np.zeros(shape, dtype=np.float32)
    denom = np.zeros(shape, dtype=np.float32)
    for i, p in enumerate(paths):
        img = read_image(p).astype(np.float32)
        # Multithreaded exact Euclidean distance transform. `black_border=
        # False` matches scipy.ndimage.distance_transform_edt (the array
        # border is NOT treated as background); `parallel=0` uses all cores.
        # Verified numerically identical to scipy on real panels (max abs
        # diff ~5e-4, float32-sqrt noise) at ~13x the speed — scipy's EDT
        # was single-threaded and ~91% of merge wall-time.
        weight = edt.edt(masks[i].astype(np.uint8),
                         black_border=False, parallel=0).astype(np.float32)
        numer += weight * (img * scales[i])
        denom += weight

    denom[denom == 0] = 1.0
    return numer / denom


@node(display="Mosaic", final=True, auto_memory=True)
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
