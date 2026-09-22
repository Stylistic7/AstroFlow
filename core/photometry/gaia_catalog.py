"""Bright-star Gaia DR3 catalogue with per-filter synthetic fluxes — the
reference for star photometry without PixInsight.

Built from ONE public Gaia DR3 product, the bulk XP sampled mean spectra
(cdn.gea.esac.esa.int/Gaia/gdr3/Spectroscopy/xp_sampled_mean_spectrum/:
3,386 gzip CSV files, 114 GB, 219 M sources to G≈17.6 — the same content
PixInsight packages as "DR3/SP"): per source, `source_id`, `ra`, `dec`
(epoch 2016.0) and 343 flux samples, 336–1020 nm in 2 nm steps, W m⁻² nm⁻¹.
The files are streamed once and only sources brighter than a synthetic
G < 14 are kept (~1 GB); no query service and no `gaia_source` table:
brightness and colour come from the spectrum itself (`synthetic_photometry`),
and proper motion is not needed at 1.5"/px (10 mas/yr over 8 yr = 0.05 px).

Per-filter catalogue flux is the passband integral of the sampled spectrum:
narrowband = a 3 nm box centred on the line, linearly interpolated on the
2 nm grid (reproduces SpectrophotometricFluxCalibration's catalogue flux to
0.13% rms per star on 870 stars — 2026-09-07); R/G/B = the transmission
curves the SPFC task hands PixInsight, L = their envelope. Any global factor
(units, photon vs energy) cancels in the star flat's per-sub zero-points.

Stored as one raw `.npy` per column, sorted by declination, and opened
memory-mapped read-only: worker processes share a single copy through the
page cache, a field lookup is a binary search on `dec` plus an RA filter,
and only the rows in the field are ever read. No HEALPix dependency.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from pix.tasks.spectrophotometric_flux_calibration import (
    _BROADBAND_FILTERS, _CHROMA_BANDWIDTH_NM, _NARROWBAND_LINE_NM,
)

CATALOG_PATH = Path(os.environ.get(
    "ASTROFLOW_GAIA_CATALOG", "/Volumes/AstroProcessed/Catalogs/GaiaXP/gaia_xp"))
_COLUMNS = ("source_id", "ra", "dec", "G", "colour", "flux")
WAVELENGTHS_NM = 336.0 + 2.0 * np.arange(343)
FILTERS = ("H", "S", "O", "L", "R", "G", "B")
# Synthetic photometry from the spectrum, calibrated on 11,303 stars with Gaia
# G and BP-RP (2026-09-07): G_syn = zero point - 2.5 log10(total 336-1020 nm
# flux) matches G to 0.056 mag rms (colour-independent); the colour index
# maps -2.5 log10(F[400-550] / F[650-800]) onto the BP-RP scale to 0.026 mag.
_G_ZERO_POINT = -19.4716
_COLOUR_SLOPE, _COLOUR_OFFSET = 0.8114, 1.1025


def _band_weights() -> dict[str, np.ndarray]:
    """Per-filter weights over the 343-sample grid such that
    flux = (weights * spectrum).sum(). Narrowband: 3 nm box, linear
    interpolation (61 sub-samples); broadband: transmission curve
    interpolated onto the grid, times the 2 nm step."""
    out = {}
    for key, line in _NARROWBAND_LINE_NM.items():
        fine = np.linspace(line - _CHROMA_BANDWIDTH_NM / 2, line + _CHROMA_BANDWIDTH_NM / 2, 61)
        w = np.zeros(343)
        for lam in fine:                       # linear interpolation as a weight vector
            i = int(np.clip((lam - 336.0) / 2.0, 0, 341)); t = (lam - WAVELENGTHS_NM[i]) / 2.0
            w[i] += (1 - t) / len(fine); w[i + 1] += t / len(fine)
        out[key] = w * _CHROMA_BANDWIDTH_NM
    for key, (_, curve) in _BROADBAND_FILTERS.items():
        v = np.array(curve.split(","), float).reshape(-1, 2)
        out[key] = np.interp(WAVELENGTHS_NM, v[:, 0], v[:, 1], left=0.0, right=0.0) * 2.0
    # L: PI has no luminance curve and hands SPFC a flat 336-1020 nm "ideal QE",
    # which over-predicts red stars by tens of percent (the real L is ~400-700
    # nm); the envelope of the R/G/B curves is the honest luminance passband.
    out["L"] = np.maximum.reduce([out["R"], out["G"], out["B"]])
    return out


def band_fluxes(spectra: np.ndarray) -> np.ndarray:
    """(n, 343) sampled spectra -> (n, len(FILTERS)) fluxes."""
    W = _band_weights()
    return np.stack([spectra @ W[f] for f in FILTERS], axis=1).astype(np.float32)


def synthetic_photometry(spectra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(G on the Gaia G scale, colour on the BP-RP scale) from (n, 343)
    sampled spectra; NaN where the spectrum has no positive flux."""
    with np.errstate(divide="ignore", invalid="ignore"):
        total = spectra.sum(axis=1) * 2.0          # flat 336-1020 nm integral (2 nm samples), the calibrated quantity
        G = np.where(total > 0, _G_ZERO_POINT - 2.5 * np.log10(np.where(total > 0, total, 1.0)), np.nan)
        blue = spectra[:, (WAVELENGTHS_NM >= 400) & (WAVELENGTHS_NM <= 550)].mean(1)
        red = spectra[:, (WAVELENGTHS_NM >= 650) & (WAVELENGTHS_NM <= 800)].mean(1)
        ok = (blue > 0) & (red > 0)
        colour = np.where(ok, _COLOUR_SLOPE * (-2.5 * np.log10(np.where(ok, blue / np.where(ok, red, 1.0), 1.0))) + _COLOUR_OFFSET, np.nan)
    return G.astype(np.float32), colour.astype(np.float32)


def build_catalog(parts: list[Path], out_path: Path) -> int:
    """Concatenate the streamed per-file parts (source_id, ra, dec, G, colour,
    flux) into the declination-sorted catalogue. Returns the source count."""
    cols = {k: [] for k in _COLUMNS}
    for p in parts:
        z = np.load(p)
        for k in _COLUMNS:
            cols[k].append(z[k])
    out = {k: np.concatenate(v) for k, v in cols.items()}
    order = np.argsort(out["dec"], kind="stable")
    write_catalog(out_path, {k: v[order] for k, v in out.items()})
    return int(len(order))


def write_catalog(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """One raw .npy per column (memory-mappable) plus the filter order."""
    path.mkdir(parents=True, exist_ok=True)
    for k in _COLUMNS:
        np.save(path / f"{k}.npy", np.ascontiguousarray(arrays[k]))
    (path / "filters.json").write_text(json.dumps(list(FILTERS)))


class GaiaCatalog:
    """The catalogue, memory-mapped read-only (opened once per process)."""

    def __init__(self, path: Path = CATALOG_PATH):
        path = Path(path)
        for k in _COLUMNS:
            setattr(self, k, np.load(path / f"{k}.npy", mmap_mode="r"))
        self.filters = json.loads((path / "filters.json").read_text())

    def stars_in_field(self, wcs, width: int, height: int, filter_key: str, margin_px: float = 16.0):
        """Catalogue stars projected onto the frame: returns (x, y) in FITS
        1-based pixel coordinates with the catalogue rows. `wcs` is an
        astropy WCS on the frame's rows in file order."""
        corners = wcs.all_pix2world(np.array([[1, 1], [width, 1], [1, height], [width, height],
                                              [width / 2, height / 2]], float), 1)
        c_ra, c_dec = corners[-1]
        # angular radius: centre to the farthest corner (great-circle)
        radius = float(np.max(np.degrees(np.arccos(np.clip(
            np.sin(np.radians(c_dec)) * np.sin(np.radians(corners[:-1, 1]))
            + np.cos(np.radians(c_dec)) * np.cos(np.radians(corners[:-1, 1]))
            * np.cos(np.radians(corners[:-1, 0] - c_ra)), -1, 1))))) + 0.1
        lo, hi = np.searchsorted(self.dec, [c_dec - radius, c_dec + radius])
        sel = np.arange(lo, hi)
        if c_dec + radius < 89.0 and c_dec - radius > -89.0:
            dra = (self.ra[sel] - c_ra + 180) % 360 - 180
            sel = sel[np.abs(dra) * np.cos(np.radians(c_dec)) <= radius * 1.05]
        ra, dec = self.ra[sel], self.dec[sel]
        # `quiet=True`: the inverse SIP iteration is allowed to give up on
        # stars far OUTSIDE the frame (a cone query overshoots the corners
        # by design, and an order-4 polynomial extrapolates wildly there);
        # those land at nonsense pixels and the frame filter below drops
        # them. With the default it raised NoConvergence for the whole
        # field over 20 far-off stars (order-4 solutions, 2026-09-20).
        px = wcs.all_world2pix(np.c_[ra, dec], 1, quiet=True)
        inside = ((px[:, 0] > margin_px) & (px[:, 0] < width - margin_px)
                  & (px[:, 1] > margin_px) & (px[:, 1] < height - margin_px))
        sel, px = sel[inside], px[inside]
        fi = self.filters.index(filter_key)
        return {"source_id": self.source_id[sel], "x": px[:, 0], "y": px[:, 1],
                "ra": ra[inside], "dec": dec[inside], "G": self.G[sel], "colour": self.colour[sel],
                "catalog_flux": self.flux[sel, fi]}
