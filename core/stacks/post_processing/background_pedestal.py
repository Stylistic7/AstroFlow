"""Per-panel background equalization for mosaic seam removal.

Each panel goes through its own WBPP invocation, and WBPP's
`ImageIntegration` normalizes each integration to its own reference
frame. That gives adjacent panels different "zero points" even when
their physics is identical — and the resulting offset, placed
adjacent in a mosaic, is the visible seam.

This module's job: shift every panel's sky-background median to a
common positive target value (default 0.1) so adjacent panels agree
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

# Default pedestal level: low enough that even a 10× brighter star
# patch won't clip at the 1.0 ceiling of a normalized master; high
# enough that downstream additive operations stay safely positive.
# Landmann §8.1 ("output pedestal") uses the same convention for the
# same reason in the earlier calibration stage.
DEFAULT_PEDESTAL = 0.1


def apply_background_pedestal(
    image: np.ndarray, target: float = DEFAULT_PEDESTAL,
) -> np.ndarray:
    """Return `image` shifted so its sky-pixel median equals `target`.

    The shift is additive on the non-zero ("valid panel") pixels. Zero
    pixels stay zero so the panel's mask geometry is preserved.

    `image` is treated as read-only; the returned array is a new
    allocation.
    """
    if image.size == 0:
        return image
    mask = image > 0
    if not mask.any():
        return image
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
