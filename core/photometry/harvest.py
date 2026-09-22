"""Per-sub star photometry against Gaia without PixInsight: the drop-in for
`stacks.starflat.harvest_sub` (same `(stars, info)` contract plus an
SNR column, same quality gate), ~0.25 s per 62 MP sub instead of ~6.6 s on the
PixInsight slot, and trivially parallel across subs.

Validation against SpectrophotometricFluxCalibration on the same stars
(2026-09-07, NGC 2244 field, 1,461 stars): per-star log(scale) agrees to
1.24% rms with no dependence on position, FWHM or colour; catalogue flux
0.13%; measured signal 1.4%.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from file_io.sidecar import get_section
from photometry.forced_photometry import CENTROID_STEP_PX, measure
from photometry.gaia_catalog import CATALOG_PATH, GaiaCatalog

# Depth: G 12-13 roughly doubles the star flat's information per sub (photon
# noise in a 300 s sub, weighted in the solve); 13-14 adds ~0.5x more at 2x the
# sidecar bytes. Faint stars are weighted by SNR in the solve, not cut.
MAX_G = 13.5
MIN_SNR = 10.0
# A MASTER's flux scale is ONE number whose precision is scatter/sqrt(n), so
# unlike a star flat (where each star informs its own cell and sufficiency is
# judged per cell) a count floor is meaningful here: `spfc_scale` refuses
# rather than put a channel on a gauge it cannot determine.
_SPFC_MIN_STARS = 20
_SPFC_MAX_SCATTER = 0.25
_catalog: GaiaCatalog | None = None


def catalog() -> GaiaCatalog:
    """The bright-star catalogue, loaded once per process (~0.5 s)."""
    global _catalog
    if _catalog is None:
        _catalog = GaiaCatalog(CATALOG_PATH)
    return _catalog


def passband_key(filter: str | None) -> str | None:
    """The catalogue passband letter for a FITS FILTER value ("Ha" -> "H",
    "OIII" -> "O", "Lum" -> "L"), or None when there is no filter."""
    if filter is None or not str(filter).strip():
        return None
    return str(filter).strip().upper()[:1]


def wcs_from_sidecar(path: Path) -> WCS:
    hdr = get_section(path, "plate_solution")["data"]["header"]
    return WCS(fits.Header({k: v for k, v in hdr.items()
                            if isinstance(v, (int, float, str)) and k not in ("COMMENT", "HISTORY")}))


def harvest_sub_xp(path: str | Path, *, filter: str | None, cat: GaiaCatalog | None = None) -> tuple[np.ndarray, dict]:
    """Forced photometry of every catalogue star on the sub → (stars, info)
    in `stacks.starflat.STAR_COLUMNS` order (without `sub`): x, y (PI
    convention), ra, dec, G, colour, fwhm, signal, scale, snr."""
    from stacks.starflat import _match_quality, sub_is_good
    path = Path(path)
    if filter is None:
        raise ValueError(f"{path}: no FITS FILTER; cannot pick a catalogue passband")
    key = filter.strip().upper()[:1]
    cat = cat or catalog()
    with fits.open(path, memmap=False) as h:
        image = h[0].data.astype(np.float32) / 65535.0
    H, W = image.shape
    st = cat.stars_in_field(wcs_from_sidecar(path), W, H, key)
    st = {k: v[st["G"] <= MAX_G] for k, v in st.items()}       # cut BEFORE measuring: 40% of the work
    m = measure(image, st["x"], st["y"])
    ok = m["ok"] & (st["catalog_flux"] > 0) & (m["snr"] >= MIN_SNR)
    stars = np.column_stack([
        m["x"][ok], m["y"][ok], st["ra"][ok], st["dec"][ok], st["G"][ok], st["colour"][ok],
        m["fwhm"][ok], m["signal"][ok], m["signal"][ok] / st["catalog_flux"][ok], m["snr"][ok],
    ]).astype(np.float64)
    n, scatter, gross = _match_quality(stars)
    # How the sub did against WHAT WAS THERE — the catalogue says how many
    # stars the field holds, so the only question is whether they were found
    # and measured (`stacks.starflat.sub_is_good`). `n_detectable` counts the
    # stars bright enough to measure whose stamp fits on the frame; `n_found`
    # those the centroid then located within a pixel of the predicted spot.
    detectable = m["on_frame"] & (m["snr"] >= MIN_SNR)
    info = {"engine": "astroflow-xp", "n_available": int(len(st["G"])),
            "n_detectable": int(detectable.sum()),
            "n_found": int((detectable & (m["offset"] <= CENTROID_STEP_PX)).sum()),
            "n_window": n, "scatter": scatter, "gross": gross}
    return stars, {**info, "good": sub_is_good(info)}


# PI-SPFC ScaleFactor per unit of THIS engine's raw ratio. Our catalogue
# flux is the XP passband integral in W m^-2 nm^-1 (gaia_catalog.py); PI's
# is in its own internal units, and the two differ by a per-passband factor
# — the narrowband box is 3 nm while a broadband curve integrates ~90 nm,
# so the raw ratios are ~33x apart between H and R. Any global factor
# cancels in the star flat's per-sub zero points, which is why this never
# mattered until `post_stack_calibrate` started dividing the PI-unit
# SPFC_REFERENCE_SCALE by a raw ratio: gain came out ~1e-16 and 151 masters
# were scaled to numerically zero (2026-09-11).
#
# MEASURED head-to-head against PixInsight on production WBPP masters
# (2026-09-11): run SPFC on the master, read `PCL:SPFC:ScaleFactors`, divide
# by this engine's raw ratio on the same pixels. Reproducible per filter
# across DIFFERENT masters — R 2.931/2.891e-15 (1.4%), G 2.451/2.459e-15
# (0.35%) — and H agrees with a surviving PI-era production certificate
# (8.752e-17) to 1.5%. NOT derivable from bandwidth alone: R matches
# width/3.0 to 1.4% but G misses by 15% and L by 2.3x, so these are
# measurements, not a formula. A new filter needs a new measurement.
_XP_TO_PI_SCALE: dict[str, float] = {
    "H": 8.880e-17,
    "S": 8.667e-17,
    "O": 1.1672e-16,
    "R": 2.911e-15,
    "G": 2.455e-15,
    "B": 3.553e-15,
    "L": 3.185e-14,
}


def spfc_scale(
    image: np.ndarray, wcs: WCS, *, filter: str | None,
    cat: GaiaCatalog | None = None,
) -> tuple[float, int, float]:
    """PI SPFC's `ScaleFactors` for an already-plate-solved image, computed
    here instead of in PixInsight: `(scale, n_stars, sigma)`.

    `scale` is the robust image/catalogue flux ratio over matched stars,
    CONVERTED to PI's units via `_XP_TO_PI_SCALE`, so it is numerically the
    quantity PI publishes as `PCL:SPFC:ScaleFactors` and
    `SPFC_REFERENCE_SCALE / scale` is the gain that lands pixels on the
    house photometric gauge (`units.py`). `harvest_sub_xp`'s per-star
    `scale` column is the same ratio UNCONVERTED — correct there, because
    the star flat only ever compares sub to sub within one passband.

    `image` must already be in the [0, 1] convention PI uses — raw 16-bit
    subs divide by 65535 (as `harvest_sub_xp` does), a float XISF master is
    already there. Getting that wrong rescales the gauge silently.

    Raises ValueError when too few stars match, mirroring what PI's
    "Insufficient data" produced: a master that cannot be flux-calibrated
    must not pass downstream.
    """
    from stacks.starflat import _match_quality

    if filter is None:
        raise ValueError("spfc_scale: no FITS FILTER; cannot pick a catalogue passband")
    key = filter.strip().upper()[:1]
    if key not in _XP_TO_PI_SCALE:
        raise ValueError(
            f"spfc_scale: no measured PI-unit conversion for filter {filter!r} "
            f"(key {key!r}); have {sorted(_XP_TO_PI_SCALE)}. Measure it against "
            f"PixInsight before calibrating this passband — guessing the "
            f"conversion puts the channel on the wrong photometric gauge."
        )
    cat = cat or catalog()
    H, W = image.shape
    st = cat.stars_in_field(wcs, W, H, key)
    st = {k: v[st["G"] <= MAX_G] for k, v in st.items()}
    m = measure(image, st["x"], st["y"])
    ok = m["ok"] & (st["catalog_flux"] > 0) & (m["snr"] >= MIN_SNR)
    stars = np.column_stack([
        m["x"][ok], m["y"][ok], st["ra"][ok], st["dec"][ok], st["G"][ok], st["colour"][ok],
        m["fwhm"][ok], m["signal"][ok], m["signal"][ok] / st["catalog_flux"][ok], m["snr"][ok],
    ]).astype(np.float64)

    n, scatter, _gross = _match_quality(stars)
    if n < _SPFC_MIN_STARS or not np.isfinite(scatter) or scatter >= _SPFC_MAX_SCATTER:
        raise ValueError(
            f"spfc_scale: too few calibratable stars (n={n} in window, "
            f"scatter={scatter:.3f}); need n>={_SPFC_MIN_STARS} and "
            f"scatter<{_SPFC_MAX_SCATTER}"
        )
    # Median over the SAME magnitude window the quality gate uses. The
    # per-star ratio varies across the field (that is exactly what the star
    # flat solves for), so the frame-level number is its robust centre.
    G, ratio = stars[:, 4], stars[:, 8]
    win = (G >= 8.0) & (G <= MAX_G) & np.isfinite(ratio) & (ratio > 0)
    r = ratio[win] * _XP_TO_PI_SCALE[key]
    scale = float(np.median(r))
    sigma = float(np.median(np.abs(r - scale)) * 1.4826)
    return scale, int(win.sum()), sigma
