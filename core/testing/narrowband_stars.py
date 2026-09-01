"""Render star fields with LEGITIMATE narrowband fluxes from Gaia XP spectra.

CatalogStarGenerator paints stars from one broadband magnitude column, so a
synthetic frame it renders cannot close the loop with SPFC's narrowband
prediction (XP spectrum × rectangular band). This renderer computes each
star's band flux with THE SAME model SPFC uses — the line centers and
bandwidth are imported from the production SPFC task, not copied — so a
frame rendered here and calibrated by SPFC must agree up to the injected
flux scale plus photon noise. That closure is the whole point.
"""
from __future__ import annotations

import numpy as np

from pix.tasks.spectrophotometric_flux_calibration import (
    _CHROMA_BANDWIDTH_NM,
    _NARROWBAND_LINE_NM,
)

# SPFC integrates the XP spectrum over the rectangular band in
# `narrowbandIntegrationSteps` = 10 steps; mirror the sample count.
_INTEGRATION_STEPS = 10


def band_flux(
    wavelength_nm: np.ndarray, spectrum: np.ndarray, filter_key: str
) -> float:
    """PHOTON flux through the production narrowband rectangle for one star.

    Rectangle center/width come from the production SPFC constants; the
    spectrum is sampled at the band's step midpoints. Samples are
    photon-weighted (× λ): a sensor counts photons, and SPFC's predicted
    fluxes carry the same weighting — verified 2026-09-01 by the
    cross-filter unit constants, whose energy-flux residuals matched the
    band-wavelength ratios exactly (O/S 1.340 vs λ_S/λ_O 1.341).
    Units: (spectrum flux density) × nm × (λ/nm) — arbitrary but common
    to all filters, which is all the closure needs.
    """
    center = _NARROWBAND_LINE_NM[filter_key]
    width = _CHROMA_BANDWIDTH_NM
    edges = np.linspace(center - width / 2, center + width / 2,
                        _INTEGRATION_STEPS + 1)
    midpoints = (edges[:-1] + edges[1:]) / 2
    samples = np.interp(midpoints, wavelength_nm, spectrum) * midpoints
    return float(np.clip(samples, 0.0, None).mean() * width)


def _moffat_stamp(fwhm_px: float, beta: float = 4.0) -> np.ndarray:
    """Unit-total-flux Moffat PSF stamp (odd-sized, radius 6×FWHM)."""
    alpha = fwhm_px / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    r = int(np.ceil(6.0 * fwhm_px))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    stamp = (1.0 + (xx ** 2 + yy ** 2) / alpha ** 2) ** (-beta)
    return stamp / stamp.sum()


def render_narrowband_stars(
    height: int,
    width: int,
    wcs_header: dict,
    catalog: dict,
    filter_key: str,
    adu_per_flux: float,
    fwhm_px: float = 2.8,
) -> np.ndarray:
    """Paint the catalog's stars, each with `band_flux × adu_per_flux` total
    ADU, at their WCS pixel positions. Returns float64 (caller adds sky /
    noise and quantizes). Stars outside the frame (with stamp margin) are
    skipped.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    # read_header hands back the full header with non-FITS-scalar values
    # (e.g. DATE-OBS as datetime); keep only the WCS cards astropy needs.
    wcs_prefixes = ("NAXIS", "CTYPE", "CRVAL", "CRPIX", "CD1_", "CD2_",
                    "CDELT", "CROTA", "PC1_", "PC2_", "EQUINOX", "RADESYS")
    cards = {k: v for k, v in wcs_header.items()
             if isinstance(k, str) and k.upper().startswith(wcs_prefixes)
             and isinstance(v, (str, int, float, bool))}
    wcs = WCS(fits.Header(cards))
    stamp = _moffat_stamp(fwhm_px)
    r = stamp.shape[0] // 2

    ras = np.array([s[0] for s in catalog["stars"]])
    decs = np.array([s[1] for s in catalog["stars"]])
    xs, ys = wcs.wcs_world2pix(ras, decs, 0)

    img = np.zeros((height, width), np.float64)
    rendered = 0
    for (_, _, _, spectrum), x, y in zip(catalog["stars"], xs, ys):
        cx, cy = int(round(x)), int(round(y))
        if not (r <= cx < width - r and r <= cy < height - r):
            continue
        flux_adu = band_flux(catalog["wavelength_nm"], spectrum,
                             filter_key) * adu_per_flux
        img[cy - r:cy + r + 1, cx - r:cx + r + 1] += flux_adu * stamp
        rendered += 1
    if rendered < 20:
        raise RuntimeError(
            f"only {rendered} stars landed in the {width}x{height} frame — "
            f"field/WCS mismatch or catalog too shallow"
        )
    return img
