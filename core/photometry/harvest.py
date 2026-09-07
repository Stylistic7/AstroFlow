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
from photometry.forced_photometry import measure
from photometry.gaia_catalog import CATALOG_PATH, GaiaCatalog

# Depth: G 12-13 roughly doubles the star flat's information per sub (photon
# noise in a 300 s sub, weighted in the solve); 13-14 adds ~0.5x more at 2x the
# sidecar bytes. Faint stars are weighted by SNR in the solve, not cut.
MAX_G = 13.5
MIN_SNR = 10.0
_catalog: GaiaCatalog | None = None


def catalog() -> GaiaCatalog:
    """The bright-star catalogue, loaded once per process (~0.5 s)."""
    global _catalog
    if _catalog is None:
        _catalog = GaiaCatalog(CATALOG_PATH)
    return _catalog


def wcs_from_sidecar(path: Path) -> WCS:
    hdr = get_section(path, "plate_solution")["data"]["header"]
    return WCS(fits.Header({k: v for k, v in hdr.items()
                            if isinstance(v, (int, float, str)) and k not in ("COMMENT", "HISTORY")}))


def harvest_sub_xp(path: str | Path, *, filter: str | None, cat: GaiaCatalog | None = None) -> tuple[np.ndarray, dict]:
    """Forced photometry of every catalogue star on the sub → (stars, info)
    in `stacks.starflat.STAR_COLUMNS` order (without `sub`): x, y (PI
    convention), ra, dec, G, colour, fwhm, signal, scale, snr."""
    from stacks.starflat import _match_quality, _MAX_SCATTER, _MIN_WINDOW_STARS
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
    return stars, {"engine": "astroflow-xp", "n_window": n, "scatter": scatter, "gross": gross,
                   "good": n >= _MIN_WINDOW_STARS and np.isfinite(scatter) and scatter < _MAX_SCATTER}
