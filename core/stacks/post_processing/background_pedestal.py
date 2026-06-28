"""Per-panel background equalization for mosaic seam removal.

Each panel goes through its own WBPP invocation, and WBPP's
`ImageIntegration` normalizes each integration to its own reference
frame. That gives adjacent panels different "zero points" even when
their physics is identical — and the resulting offset, placed
adjacent in a mosaic, is the visible seam.

This module's job: shift every panel's sky-background median to a
common positive target value (default 0.005) so adjacent panels agree
on the baseline at any boundary they share. The pedestal step is
strictly additive — the multiplicative photometric ratio between
panels is preserved exactly, leaving PMM's star-based scaling free
to do its job. (LinearFit also adjusts scale, which IS the reason
the community warns against LinearFit before PMM. The additive-only
piece of that operation has no such conflict.)

Why this lives here (NumPy) rather than PixelMath:
    - It's `panel - median + target` in float32. Sub-millisecond on
      a CMOS-sized panel; PixInsight's PixelMath measured ~600 ms
      for the same operation including PI's image-cache writes.
    - `Med($T)` in PixInsight's PixelMath is the whole-image median,
      which sits ~1 % above the sky-only median because stars and
      nebulosity pull it up. For panels with different star/sky
      ratios the bias differs per panel, so the pedestal shifts
      land slightly off-target — the seam shrinks but doesn't fully
      vanish. NumPy lets us use a proper lower-quantile sky-only
      median (immune to star contamination).
    - place_image already reads the panel and operates on numpy
      arrays; PI involvement here is pure overhead.

The black mask is preserved: pixels at exactly zero stay zero so
that downstream OSS / PMM mask logic (`img > 0`) still finds panel
boundaries unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# Lower-quantile cut for sky-only median estimation. Pixels below the
# 25th percentile of the valid panel are overwhelmingly sky (typical
# field has < 5 % stellar coverage; nebulosity is brighter still).
# Lowering this further (e.g. 10 %) would shrink the sample at no
# benefit on a typical field.
_SKY_QUANTILE = 0.25

# Default pedestal level, on the [0,1] normalized-master scale. Kept near a
# true LINEAR black point, so the sky lands where stretches (GHS/STF/arcsinh)
# and the ML tools (StarXTerminator, BlurXTerminator) expect it. This is NOT a
# stretch-target background — 0.1–0.25 is the FINISHED-(stretched-)image
# convention and is ~10–50× too high for linear data, which lifts the
# background under any stretch and forces a razor-exact symmetry point. Still
# safely above zero: measured deep masters have sky-noise σ ≈ 2.5e-5, so 0.005
# sits ~200 σ above 0 and clips no noise. Matches the linear "output pedestal"
# convention (Landmann: a few hundred DN, ≤ 1000 DN ≈ 0.015 normalized).
DEFAULT_PEDESTAL = 0.005


def apply_background_pedestal(
    image: np.ndarray, target: float = DEFAULT_PEDESTAL,
    mask_zeros: bool = True,
) -> np.ndarray:
    """Return `image` shifted so its sky-pixel median equals `target`.

    The shift is additive. With `mask_zeros=True` (default) it acts only on the
    non-zero ("valid panel") pixels and zeros stay zero, preserving a placed
    panel's boundary geometry. Set `mask_zeros=False` for a SOLID master that
    isn't placed yet (e.g. the OSS GraXpert path, whose background-subtracted
    sky has sub-zero pixels): then every pixel is shifted — sub-zero sky is
    lifted up rather than read as boundary — and the result is clamped to a
    tiny positive so nothing downstream (`img > 0`) mistakes it for an edge.

    `image` is treated as read-only; the returned array is a new
    allocation.
    """
    if image.size == 0:
        return image
    if mask_zeros:
        mask = image > 0
        if not mask.any():
            return image
    else:
        mask = np.ones(image.shape, dtype=bool)
    panel = image[mask]

    # Sky-only median via lower-quantile cut: stars and nebulosity sit
    # above the 25 % line, so the median of the lower quarter is a
    # robust sky-floor estimate.
    cutoff = np.quantile(panel, _SKY_QUANTILE)
    sky = panel[panel <= cutoff]
    if sky.size == 0:
        return image  # Nothing recognizable as sky — leave alone.
    sky_median = float(np.median(sky))

    shifted = image.copy()
    shifted[mask] = panel - sky_median + float(target)
    # Hold the mask: any pixel pushed below zero by an unusually low
    # sky_median (e.g. heavy negative excursions in the input) would
    # otherwise become indistinguishable from the panel boundary.
    shifted[mask] = np.maximum(shifted[mask], np.finfo(image.dtype).tiny)
    return shifted


def apply_background_pedestal_to_xisf(
    filename: Path, target: float = DEFAULT_PEDESTAL,
) -> None:
    """In-place pedestal application on an XISF file, preserving its
    FITS keywords + XISF metadata.

    Modeled on `file_io.xisf_file_io.update_xisf_header`'s read-modify-
    write pattern: open via the XISF reader, swap the pixel buffer for
    the pedestal-shifted version, write back with the original image
    and file metadata. Used by `post_process_stack` to equalize each
    panel's sky baseline before any downstream consumer reads it.
    """
    # Local import: xisf_file_io pulls in the XISF library, which we
    # don't want loading every time someone imports this module just
    # for the in-memory `apply_background_pedestal`.
    from xisf import XISF  # type: ignore

    xisf = XISF(str(filename))
    image = xisf.read_image(0)
    file_meta = xisf.get_file_metadata()
    ims_meta = xisf.get_images_metadata()

    image = apply_background_pedestal(image, target=target)

    del xisf

    XISF.write(
        str(filename),
        image,
        creator_app="AstroFlow",
        image_metadata=ims_meta[0],
        xisf_metadata=file_meta,
    )
