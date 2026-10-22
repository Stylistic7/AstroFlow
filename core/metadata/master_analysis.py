"""`analyze_master`: measurements of a finished master, for planning and for
the Tier 2 coadd.

A LEAF node: it consumes a master's `ImageMetadata` (whatever stacker made
it) and nothing consumes its result, so adding it — or new analyses to it —
re-keys nothing in the graph. Results are written into the master's sidecar
as the `analysis` section and returned; readers (the night planner, the
dashboard, the all-sky combiner) read the sidecar, never the graph.

    analysis = {
      "version": ANALYSIS_VERSION,
      "stack": STBLHASH of the master,          # which stack it was built from
      "filter": FILTER,
      "integration_s": master_integration_s,     # what the curve is relative to
      "noise_px": σ,                             # the pixels' Gaussian noise, this file's units
      "pixel_arcsec": p,                         # the pixel's side on the sky (None unsolved)
      "psf": {"fwhm_px": f, "fwhm_arcsec": f·p, "n_stars": n},   # see `psf_of`
      "visibility": {                            # metadata.visibility
        "thresholds": [3, 10, 30, 100],          # SNR: detected, usable, clean, portfolio
        "scales": [1, 2, 4, 8], "bin": 2,
        "factors":  [1, 1.25, 1.5, 2, 3, 4],     # x the master's integration
        "fraction": [[...], ...],                # per threshold: field at or above it, at each factor
      },
    }

    Version 2 (2026-10-12) replaced the single detection threshold with the
    quality levels: a panel is not done at SNR 3, and an hour's worth is
    the fraction of the field that climbs a level.
    Version 3 (2026-10-22) added what proper coaddition needs of a master
    besides its WCS: its noise and its PSF, measured on the pixels the
    sidecar sits beside (a header NOISE00 survives a gain change; a
    measurement does not), so a master shared through a folder carries its
    own weights. `measure_master` is the pure core; `post_stack_calibrate`
    runs it on the calibrated pixels so the shared master describes itself.

More analyses belong here as further keys. If one becomes expensive, give
it its own leaf node so the others stay cached when it changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from decorators import node
from metadata.classes import ImageMetadata

ANALYSIS_VERSION = 3
SECTION = "analysis"
FACTORS = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
BIN = 2
NOISE_CROP = 2048            # the noise is read off the central crop: a 5x5 median over 60 MP is seconds for nothing
PSF_PEAK_SIGMA = 20.0        # a star is a local maximum this far above the sky
PSF_MAX_STARS = 3000
PSF_MIN_STARS = 10


def pixel_arcsec_of(wcs_header: dict | None) -> float | None:
    """The pixel's side on the sky from a plate solution's header, or None."""
    if not wcs_header:
        return None
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_area
    cards = {k: v for k, v in wcs_header.items()
             if isinstance(v, (int, float, str)) and k not in ("COMMENT", "HISTORY")}
    try:
        return float(np.sqrt(proj_plane_pixel_area(WCS(fits.Header(cards)))) * 3600.0)
    except Exception:          # noqa: BLE001 — an unusable solution is "unsolved" here
        return None


def psf_of(img: np.ndarray, noise: float) -> dict:
    """The master's PSF width: the median second-moment FWHM of its bright
    isolated peaks, in pixels, by the star photometry's moment estimator.
    This is the Gaussian whose second moment matches the star core — what a
    Gaussian PSF model wants — not the PSF-fit FWHM that estimator is
    calibrated to (`FWHM_SCALE` is undone). None with too few stars: a
    star-removed master has no PSF to measure, and says so."""
    from scipy.ndimage import maximum_filter
    from photometry.forced_photometry import FWHM_SCALE, measure
    finite = np.isfinite(img)
    sky = float(np.nanmedian(img[::7, ::7]))
    peaks = finite & (img == maximum_filter(np.where(finite, img, -np.inf), size=7)) & (img > sky + PSF_PEAK_SIGMA * noise)
    ys, xs = np.nonzero(peaks)
    if ys.size > PSF_MAX_STARS:
        keep = np.argpartition(-img[ys, xs], PSF_MAX_STARS)[:PSF_MAX_STARS]
        ys, xs = ys[keep], xs[keep]
    if ys.size < PSF_MIN_STARS:
        return {"fwhm_px": None, "n_stars": int(ys.size)}
    m = measure(np.where(finite, img, np.float32(sky)), xs + 1.0, ys + 1.0)     # FITS 1-based positions
    # Not `m["ok"]`: that carries PI's photometric rule rejecting any star
    # brighter than 0.75 x the brightest, which on a field of equal stars is
    # every star. A PSF width wants the measured, centred, unsaturated ones;
    # saturated cores are flat and read broad, and are out when the image is
    # in the [0, 1] convention.
    full_scale = 0.95 if np.nanmax(img) <= 1.0 else np.inf
    good = (m["on_frame"] & (m["signal"] > 0) & (m["offset"] <= 1.0) & (m["snr"] > 20)
            & np.isfinite(m["fwhm"]) & (m["peak"] < full_scale))
    if good.sum() < PSF_MIN_STARS:
        return {"fwhm_px": None, "n_stars": int(good.sum())}
    return {"fwhm_px": float(np.median(m["fwhm"][good]) / FWHM_SCALE), "n_stars": int(good.sum())}


def measure_master(img: np.ndarray, *, stable_hash: str, filter: str, integration_s: float | None,
                   pixel_arcsec: float | None) -> dict:
    """The `analysis` section of a master from its pixels, this file's units."""
    from metadata import visibility as V
    img = np.asarray(img, np.float32).squeeze()
    h, w = img.shape
    crop = img[max(0, h // 2 - NOISE_CROP // 2):h // 2 + NOISE_CROP // 2, max(0, w // 2 - NOISE_CROP // 2):w // 2 + NOISE_CROP // 2]
    noise = V._noise(np.where(np.isfinite(crop), crop, np.nan))
    psf = psf_of(img, noise)
    fwhm_px = psf["fwhm_px"]
    binned = V._bin(img, BIN)
    valid, maps = V.snr_maps(binned)
    return {
        "version": ANALYSIS_VERSION,
        "stack": str(stable_hash),
        "filter": str(filter),
        "integration_s": integration_s,
        "noise_px": noise,
        "pixel_arcsec": pixel_arcsec,
        "psf": {**psf, "fwhm_arcsec": fwhm_px * pixel_arcsec if fwhm_px and pixel_arcsec else None},
        "visibility": {
            "thresholds": list(V.THRESHOLDS), "scales": list(V.SCALES), "bin": BIN,
            "factors": list(FACTORS),
            "fraction": [[V.visible_fraction(valid, maps, f, t) for f in FACTORS] for t in V.THRESHOLDS],
        },
    }


@node(num_cpus=1, auto_memory=True, resources={"diskcache": 0.001})
def analyze_master(master: ImageMetadata) -> dict:
    """Measure `master` and write the results into its sidecar."""
    from file_io.generic import read_image
    from file_io.sidecar import set_section
    solution = getattr(master, "plate_solution", None)
    result = measure_master(
        np.asarray(read_image(Path(master.filename)), np.float32),
        stable_hash=str(master.stable_hash), filter=str((master.header or {}).get("FILTER", "")),
        integration_s=master.master_integration_s,
        pixel_arcsec=pixel_arcsec_of(solution.header if solution else None),
    )
    set_section(Path(master.filename), SECTION, result)
    return result
