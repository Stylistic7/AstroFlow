"""The Tier 2 combiner: a folder of masters → a HiPS tree, by proper coaddition.

Every master is a plate-solved image (any format file_io reads, with a TAN
WCS in its header or its sidecar's plate solution) in one photometric
system; nothing else is assumed about it -- not the rig, the scale, the
depth or the field size. Its sidecar's `analysis` section (metadata/
master_analysis.py) gives its noise and its PSF width; a folder without
sidecars gives the PSF on the command line. The sky is cut into HEALPix
NESTED cells at one order (the finest input's pixel scale unless given),
and every tile is the proper coadd of the masters covering it (Zackay &
Ofek 2017, ApJ 836, 188): each master, sampled bilinearly at k × k
sub-positions of a cell (k = cell / pixel, so a fine master is box-filtered
onto a coarse cell and a coarse one interpolated) in surface brightness per
arcsec², is cross-correlated with its own PSF `P_j` (a Gaussian of its
measured width, widened by its pixel and the interpolation) and summed with
weight `w_j = F_j / σ_j²`; the sum is then whitened by the quadrature sum of
the PSFs' transfer functions:

    R̂(f) = Σ_j w_j P̂_j*(f) M̂_j(f) / √( Σ_j w_j |P̂_j(f)|² )          (their eq. 7)

and divided by Ω = Σ_j w_j to land in flux units. R is a proper image: PSF
`P̂_R = √(Σ w_j |P̂_j|² / Σ w_j)`, the quadrature-weighted mean of the inputs'
transfer functions (written to each tile as PSFFWHM), noise 1/√Ω, and no
information lost -- a coarse master contributes only the spatial
frequencies it has, a fine one adds the rest, and a region covered by one
master reproduces that master exactly. The numerator and denominator are
sums over masters, so a master's term is computable from that master alone
and terms add in any order on any machine. Both are formed in Fourier space
on the tile with a margin of MARGIN cells, each master's term carrying the
same analytic P̂_j the denominator does, so their ratio is bounded
everywhere and footprint edges and tile edges never meet a deconvolution.
The denominator's shape is the tile's (its level is each cell's own Ω):
exact where every master covers the tile, a small approximation at a
footprint edge.

The output is three packed planes (allsky/tiles.py), `mean` (R), `noise`
(1/√Ω) and `count`, each with its pyramid, which the sky viewer reads as they
are; the count is 16-bit and the noise half width (it varies slowly), so the
two together cost a third of the mean. Work is per shard of 64 tiles -- the
~10⁵ tiles of a hemisphere at 3″, each 512² cells -- in NESTED order so
consecutive tiles share masters, across a process pool; a worker owns its
shards, so no two write the same file.

    python -m allsky.reduce <masters folder> <out folder> [--pattern '*.fits']
                            [--order K] [--workers N] [--title ...] [--psf-fwhm ARCSEC]
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
import warnings
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from allsky.healpix_faces import compact_bits, xyf_from_vec
from allsky.tiles import Plane, create, shard_of

CELL_ARCSEC_NSIDE1 = np.degrees(np.sqrt(4 * np.pi / 12)) * 3600   # one face's side
TILE_WIDTH = 512
MAX_SUBSAMPLE = 4          # k × k samples per cell at most (a 0.8″ master on 3.2″ cells)
MASTER_CACHE = 6           # masters held in memory per worker
MARGIN = 48                # cells reduced beyond the tile: the PSF kernels' reach, so no tile edge is a Fourier edge
WHITEN_FLOOR = 1e-30       # guards the division where every P̂ has underflowed (the numerator has too)
FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))
# Each tree's tile dtype and whether its tiles are half width (the cells of one
# order coarser, a quarter of the bytes).
PLANES = {"mean": (np.float32, False), "noise": (np.float32, True), "count": (np.uint16, False)}
TREES = tuple(PLANES)


def tree_width(name: str, width: int) -> int:
    return width // 2 if PLANES[name][1] else width


def as_tile(name: str, raster: np.ndarray) -> np.ndarray:
    """A float plane in the tree's dtype: the count rounded, 0 where empty."""
    dtype = PLANES[name][0]
    if dtype is np.uint16:
        return np.rint(np.nan_to_num(raster)).astype(dtype)
    return raster.astype(dtype)


# --- HEALPix: cell (face, x, y) → direction; the inverse of healpix_faces.xyf_from_vec

def vec_from_xyf(face, x, y) -> np.ndarray:
    """Unit vectors (N, 3) of face coordinates x, y in [0, 1]: pix2ang_nest in
    its real-valued form. Twin of `xyf_from_vec` (round-trip tested)."""
    face = np.asarray(face, np.int64)
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    s = x + y
    north, south = face < 4, face >= 8
    equ = ~north & ~south
    cap_n, cap_s = north & (s > 1.0), south & (s < 1.0)
    quadrant = np.where(north, face, np.where(south, face - 8, face - 4)).astype(np.float64)
    # Equatorial belt, in each face's own coordinates.
    tt = quadrant + np.where(equ, 0.0, 0.5) + 0.5 * (x - y)
    z = np.where(north, s, np.where(south, s - 2.0, s - 1.0)) / 1.5
    # North cap: the equal-area radius from the pole is 2 − x − y.
    r_n = 2.0 - x - y
    with np.errstate(divide="ignore", invalid="ignore"):
        tp_n = np.where(r_n > 0, (1.0 - y) / r_n, 0.5)
        tp_s = np.where(s > 0, x / s, 0.5)
    tt = np.where(cap_n, quadrant + tp_n, np.where(cap_s, quadrant + tp_s, tt))
    z = np.where(cap_n, 1.0 - r_n * r_n / 3.0, np.where(cap_s, s * s / 3.0 - 1.0, z))
    phi = np.mod(tt * (np.pi / 2), 2 * np.pi)
    sin_theta = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    return np.stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), z], axis=-1)


def lonlat_of(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.degrees(np.arctan2(vec[..., 1], vec[..., 0])) % 360.0
    return lon, np.degrees(np.arcsin(np.clip(vec[..., 2], -1.0, 1.0)))


def tile_face_xy(order: int, npix: int) -> tuple[int, int, int]:
    face = npix >> (2 * order)
    within = np.array([npix & ((1 << (2 * order)) - 1)], dtype=np.uint64)
    return face, int(compact_bits(within)[0]), int(compact_bits(within >> np.uint64(1))[0])


def tile_cell_vectors(order: int, npix: int, width: int, k: int = 1, margin: int = 0) -> np.ndarray:
    """Directions (k·k, (width + 2·margin)², 3) of the cells of a tile and
    `margin` cells around it, each cell sampled at k × k sub-positions
    (k = 1: the cell centres). The face coordinates run past the face's
    edge for the margin: `vec_from_xyf` continues the mapping there."""
    face, tx, ty = tile_face_xy(order, npix)
    n = width << order
    ext = width + 2 * margin
    iy, ix = np.mgrid[-margin:width + margin, -margin:width + margin]
    out = np.empty((k * k, ext * ext, 3))
    for a in range(k):
        for b in range(k):
            x = (tx * width + ix + (a + 0.5) / k) / n
            y = (ty * width + iy + (b + 0.5) / k) / n
            out[a * k + b] = vec_from_xyf(np.full(x.size, face), x.ravel(), y.ravel())
    return out


def tile_centres(order: int) -> np.ndarray:
    """Unit vectors (12·4^order, 3) of every tile's centre, in NESTED order."""
    n = 1 << order
    npix = np.arange(12 << (2 * order), dtype=np.uint64)
    face = (npix >> np.uint64(2 * order)).astype(np.int64)
    within = npix & np.uint64((1 << (2 * order)) - 1)
    tx = compact_bits(within).astype(np.float64)
    ty = compact_bits(within >> np.uint64(1)).astype(np.float64)
    return vec_from_xyf(face, (tx + 0.5) / n, (ty + 0.5) / n)


# --- masters

@dataclass(frozen=True)
class Master:
    path: Path
    header: dict            # the WCS cards (picklable; astropy.wcs.WCS is built in the worker)
    shape: tuple[int, int]
    pixel_arcsec: float
    scale: float            # file units × scale = surface brightness per arcsec²
    sigma: float            # background noise, scaled units
    fwhm_arcsec: float      # the PSF's Gaussian FWHM on the sky
    centre: tuple[float, float, float]
    radius_deg: float       # centre → farthest corner
    offset: float = 0.0     # additive sky offset (scaled units) subtracted when sampling
    flux: float = 1.0       # F_j: the master's flux scale relative to the collection (1 when calibrated)

    def weight(self) -> float:
        return self.flux / (self.sigma * self.sigma)

    def psf_sigma_cells(self, cell_arcsec: float, k: int) -> float:
        """The Gaussian σ, in cells, of this master's PSF as it lands on the
        cell grid: its own width, its pixel's box (p²/12) and, when it is
        interpolated rather than box-averaged (k = 1), the bilinear triangle
        (p²/6)."""
        p2 = self.pixel_arcsec ** 2
        var = (self.fwhm_arcsec * FWHM_TO_SIGMA) ** 2 + p2 / 12 + (p2 / 6 if k == 1 else 0.0)
        return float(np.sqrt(var)) / cell_arcsec


def _wcs_cards(path: Path) -> dict:
    """The master's WCS header: its sidecar's plate solution when it has one
    (an AstroFlow master), else the image header."""
    from file_io.generic import read_header
    from file_io.sidecar import get_section
    try:
        section = get_section(path, "plate_solution")
    except Exception:          # noqa: BLE001 — no sidecar, or not ours
        section = None
    hdr = (section or {}).get("data", {}).get("header") if section else None
    hdr = hdr or read_header(path)
    return {k: v for k, v in hdr.items()
            if isinstance(v, (int, float, str)) and k not in ("COMMENT", "HISTORY")}


def _wcs(cards: dict):
    from astropy.io import fits
    from astropy.wcs import WCS
    return WCS(fits.Header(cards))


def robust_sigma(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    if v.size < 100:
        return float("nan")
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med)))


def _analysis(path: Path) -> dict:
    """The master's sidecar `analysis` section (metadata/master_analysis.py), or {}."""
    from file_io.sidecar import get_section
    try:
        return get_section(path, "analysis") or {}
    except Exception:          # noqa: BLE001 — no sidecar, or not ours
        return {}


def open_master(path: Path, psf_fwhm: float | None = None) -> Master:
    """Measure a master without holding its pixels: WCS, scale, footprint,
    noise and PSF. Noise and PSF come from the sidecar's `analysis`, measured
    on these very pixels; failing that the noise is the header's NOISE00 or
    the MAD of a strided sample, and the PSF is `psf_fwhm` -- a folder of
    masters that carry no PSF must say what it is, never guess."""
    from astropy.wcs.utils import proj_plane_pixel_area
    cards = _wcs_cards(path)
    wcs = _wcs(cards)
    ny, nx = int(cards.get("NAXIS2") or 0), int(cards.get("NAXIS1") or 0)
    if not (nx and ny):
        from file_io.generic import read_image
        ny, nx = np.asarray(read_image(path)).squeeze().shape[-2:]
    area = proj_plane_pixel_area(wcs) * 3600.0 ** 2
    scale = 1.0 / area
    analysis = _analysis(path)
    noise = analysis.get("noise_px")
    if not (isinstance(noise, (int, float)) and noise > 0):
        noise = cards.get("NOISE00")
    if not (isinstance(noise, (int, float)) and noise > 0):
        noise = robust_sigma(_strided_sample(path, (ny, nx)))
    fwhm = (analysis.get("psf") or {}).get("fwhm_arcsec") or psf_fwhm
    if not (isinstance(fwhm, (int, float)) and fwhm > 0):
        raise ValueError(f"{path.name}: no PSF width in its sidecar's analysis and no --psf-fwhm given")
    cx, cy = (nx - 1) / 2, (ny - 1) / 2
    corners = np.array([[cx, cy], [0, 0], [nx - 1, 0], [0, ny - 1], [nx - 1, ny - 1]], np.float64)
    lon, lat = wcs.all_pix2world(corners[:, 0], corners[:, 1], 0)
    v = _vec(np.radians(lon), np.radians(lat))
    radius = float(np.degrees(np.arccos(np.clip(v[1:] @ v[0], -1, 1)).max()))
    return Master(path=Path(path), header=cards, shape=(ny, nx), pixel_arcsec=float(np.sqrt(area)),
                  scale=scale, sigma=float(noise) * scale, fwhm_arcsec=float(fwhm),
                  centre=tuple(v[0]), radius_deg=radius)


def _vec(lon_rad, lat_rad) -> np.ndarray:
    c = np.cos(lat_rad)
    return np.stack([c * np.cos(lon_rad), c * np.sin(lon_rad), np.sin(lat_rad)], axis=-1)


def _strided_sample(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """About 65k pixels spread over the image: enough for a MAD, cheap to read."""
    step = max(1, int(np.sqrt(shape[0] * shape[1] / 65536)))
    if path.suffix.lower() in (".fits", ".fit", ".fts"):
        from astropy.io import fits
        with fits.open(path, memmap=True) as h:
            return np.asarray(h[0].section[::step, ::step], np.float64)
    return _load_pixels(path)[::step, ::step]


def _load_pixels(path: Path) -> np.ndarray:
    from file_io.generic import read_image
    a = np.asarray(read_image(path), np.float32).squeeze()
    if a.ndim == 3:
        a = a[0] if a.shape[0] <= 4 else a[..., 0]
    return a


# --- proper coaddition

def psf_hat(sigma_cells: float, n: int) -> np.ndarray:
    """`P̂(f)` of a unit-sum Gaussian of σ cells on an n × n grid's rfft
    frequencies: exp(−2π²σ²|f|²)."""
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.rfftfreq(n)[None, :]
    return np.exp(-2 * np.pi ** 2 * sigma_cells ** 2 * (fx * fx + fy * fy))


def transfer_shape(spectra: list[tuple[float, float]], n: int) -> np.ndarray:
    """`s(f) = Σ w_j |P̂_j(f)|² / Σ w_j` on an n × n grid's rfft frequencies;
    s(0) = 1. The coadd's PSF is √s."""
    total = sum(w for w, _ in spectra)
    return sum(w * psf_hat(s, n) ** 2 for w, s in spectra) / total


def coadd_fwhm_cells(spectra: list[tuple[float, float]]) -> float:
    """The FWHM, in cells, of the coadd's PSF `P̂_R = √s`: a slice through
    the centre of its 2-D profile (√s is not separable, so a 1-D transform
    would give the PSF's projection, which is wider)."""
    n = 512
    profile = np.fft.irfft2(np.sqrt(transfer_shape(spectra, n)), s=(n, n))[0]
    profile = profile / profile[0]
    half = int(np.argmax(profile < 0.5))             # first cell below half maximum
    x = half - 1 + (profile[half - 1] - 0.5) / (profile[half - 1] - profile[half])
    return float(2 * x)


def whiten(numer_hat: np.ndarray, omega: np.ndarray, spectra: list[tuple[float, float]]) -> np.ndarray:
    """The proper coadd in flux units, `IFFT(N̂ / √s) / Ω`, from the Fourier
    numerator `N̂ = Σ w_j P̂_j M̂_j`, the weight sum Ω and the masters' (w, σ)
    that shaped this tile. Each master's term carries the same analytic
    P̂_j the denominator does, so the ratio is bounded everywhere (by
    √(Ω/w_j)), footprint edges included: no deconvolution ever happens, and
    a single master comes back as itself."""
    covered = omega > 0
    if not covered.any():
        return np.full(omega.shape, np.nan, np.float32)
    s = transfer_shape(spectra, omega.shape[0])
    r = np.fft.irfft2(numer_hat / np.sqrt(np.maximum(s, WHITEN_FLOOR)), s=omega.shape) / np.where(covered, omega, 1.0)
    return np.where(covered, r, np.nan).astype(np.float32)


# --- one tile

_pixels: OrderedDict = OrderedDict()
_wcs_cache: dict = {}


def _master_pixels(m: Master) -> np.ndarray:
    a = _pixels.get(m.path)
    if a is None:
        a = _load_pixels(m.path) * np.float32(m.scale)
        # exact zero is no data, as everywhere in the pipeline (a registered
        # frame's rotated-in corners): never sky
        a = np.where(np.isfinite(a) & (a != 0), a, np.nan).astype(np.float32)
        _pixels[m.path] = a
        while len(_pixels) > MASTER_CACHE:
            _pixels.popitem(last=False)
    else:
        _pixels.move_to_end(m.path)
    return a


def sample_master(m: Master, vectors: np.ndarray) -> np.ndarray:
    """The master's surface brightness at each direction of `vectors`
    (k·k, cells, 3), averaged over the k·k sub-positions; NaN off the image
    or over its holes."""
    from scipy.ndimage import map_coordinates
    wcs = _wcs_cache.get(m.path)
    if wcs is None:
        wcs = _wcs_cache[m.path] = _wcs(m.header)
    data = _master_pixels(m)
    flat = vectors.reshape(-1, 3)
    near = flat @ np.asarray(m.centre) >= np.cos(np.radians(m.radius_deg * 1.02))
    out = np.full(flat.shape[0], np.nan, np.float64)
    if near.any():
        lon, lat = lonlat_of(flat[near])
        px, py = wcs.all_world2pix(lon, lat, 0, quiet=True, maxiter=20, tolerance=1e-3)
        px, py = np.asarray(px, np.float64), np.asarray(py, np.float64)
        inside = np.isfinite(px) & np.isfinite(py) & (px > -1) & (py > -1) \
            & (px < m.shape[1]) & (py < m.shape[0])
        vals = np.full(px.shape, np.nan)
        if inside.any():
            vals[inside] = map_coordinates(data, [py[inside], px[inside]], order=1,
                                           mode="constant", cval=np.nan)
        out[near] = vals
    return _nanmean(out.reshape(vectors.shape[:2]), axis=0) - m.offset


def _nanmean(a: np.ndarray, axis: int) -> np.ndarray:
    """nanmean without the warning for all-NaN slices (they are the answer)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


# --- sky offsets: every master carries its own night's foreground (for Hα the
# geocoronal line itself), which a star-based flux calibration cannot see.
# Where two masters overlap, the median of their difference measures the
# relative offset with the nebulosity cancelled; one least squares over all
# pairs (mosaics/merging/level_solve.py) then places every master, with one
# deterministic reference per connected group pinned at zero so the gauge
# stays put as masters are added. No external zero: the map stands on its
# own data. An absolute scale, where a reference exists, is a later constant.

MIN_OVERLAP_POINTS = 25
REJECT_SIGMA = 3.0
GRID_POINTS = 100           # across the smaller master's diameter: a strip a few arcminutes wide still counts


def master_id(m: Master) -> str:
    st = m.path.stat()
    return f"{m.path.name}:{st.st_size}:{int(st.st_mtime)}"


def overlap_grid(centre: np.ndarray, half_deg: float) -> np.ndarray:
    """Directions (1, N, 3) on a square grid of the sky around `centre`."""
    lon0, lat0 = lonlat_of(centre)
    d = np.linspace(-half_deg, half_deg, GRID_POINTS)
    dlon, dlat = np.meshgrid(d / max(np.cos(np.radians(lat0)), 0.2), d)
    lon = np.radians(lon0 + dlon.ravel())
    lat = np.radians(np.clip(lat0 + dlat.ravel(), -89.9, 89.9))
    return _vec(lon, lat)[None, :, :]


def measure_pairs(masters: list[Master], ids: list[str], known: dict[str, list]) -> tuple[list[tuple], dict[str, list]]:
    """`(i, j, d_ij, n)` for every overlapping pair, d_ij the sigma-clipped
    median of master i minus master j over their shared sky (surface
    brightness), n the points it rests on. Pairs already in `known` (keyed
    by both ids, so a changed file re-measures) are reused."""
    pairs, record = [], {}
    centres = np.array([m.centre for m in masters])
    for i, a in enumerate(masters):
        sep = np.degrees(np.arccos(np.clip(centres[i + 1:] @ centres[i], -1, 1)))
        for j in (np.nonzero(sep < a.radius_deg + np.array([m.radius_deg for m in masters[i + 1:]]))[0] + i + 1):
            b = masters[j]
            key = f"{ids[i]}|{ids[j]}"
            if key in known:
                d, n = known[key]
            else:
                mid = centres[i] + centres[j]
                v = overlap_grid(mid / np.linalg.norm(mid), min(a.radius_deg, b.radius_deg) * 1.05)
                va, vb = sample_master(a, v), sample_master(b, v)
                both = np.isfinite(va) & np.isfinite(vb)
                n = int(both.sum())
                if n < MIN_OVERLAP_POINTS:
                    d = None
                else:
                    diff = va[both] - vb[both]
                    med = np.median(diff)
                    mad = 1.4826 * np.median(np.abs(diff - med))
                    d = float(np.median(diff[np.abs(diff - med) <= REJECT_SIGMA * max(mad, 1e-30)]))
            record[key] = [d, n]
            if d is not None:
                pairs.append((i, int(j), d, n))
    return pairs, record


def solve_offsets(masters: list[Master], ids: list[str], pairs: list[tuple]) -> np.ndarray:
    """One offset per master from the pairs: within each connected group the
    master with the smallest id is the reference at zero (the same one every
    run, so the gauge holds); a master in no pair stays at zero. Offsets are
    quantised to a twentieth of the master's noise so a hair's change does
    not re-reduce its tiles."""
    from mosaics.merging.level_solve import solve_level_offsets
    parent = list(range(len(masters)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, *_ in pairs:
        parent[root(i)] = root(j)
    reference: dict[int, int] = {}
    for i in range(len(masters)):
        r = root(i)
        if r not in reference or ids[i] < ids[reference[r]]:
            reference[r] = i
    offsets = solve_level_offsets(len(masters), pairs, {i: 0.0 for i in reference.values()})
    quantum = np.array([0.05 * m.sigma for m in masters])
    return np.round(offsets / quantum) * quantum


def reduce_tile(order: int, npix: int, masters: list[Master], width: int) -> tuple[dict[str, np.ndarray], float] | None:
    """The three planes of one tile from the masters that may cover it and
    the coadd's PSF FWHM in arcsec, or None when none covers it."""
    cell_arcsec = CELL_ARCSEC_NSIDE1 / (width << order)
    ext = width + 2 * MARGIN
    numer_hat = np.zeros((ext, ext // 2 + 1), np.complex128)
    omega = np.zeros((ext, ext))
    count = np.zeros((ext, ext), np.int32)
    spectra: list[tuple[float, float]] = []
    cache: dict[int, np.ndarray] = {}
    for m in masters:
        k = int(np.clip(round(cell_arcsec / m.pixel_arcsec), 1, MAX_SUBSAMPLE))
        vectors = cache.get(k)
        if vectors is None:
            vectors = cache[k] = tile_cell_vectors(order, npix, width, k, MARGIN)
        values = sample_master(m, vectors).reshape(ext, ext)
        mask = np.isfinite(values)
        if not mask.any():
            continue
        w, sig = m.weight(), m.psf_sigma_cells(cell_arcsec, k)
        # this master's term, `w P̂_j M̂_j`, with M zero outside its footprint
        numer_hat += w * psf_hat(sig, ext) * np.fft.rfft2(np.where(mask, values, 0.0))
        omega += w * mask
        count += mask
        spectra.append((w, sig))
    inner = slice(MARGIN, MARGIN + width)
    if not count[inner, inner].any():
        return None
    mean = whiten(numer_hat, omega, spectra)[inner, inner]
    seen = count[inner, inner] > 0
    noise = np.where(seen, 1.0 / np.sqrt(np.maximum(omega[inner, inner], 1e-300)), np.nan).astype(np.float32)
    return ({"mean": mean, "noise": noise, "count": count[inner, inner].astype(np.float32)},
            coadd_fwhm_cells(spectra) * cell_arcsec)


# --- the planes

def properties(name: str, order: int, width: int, first: Master, title: str) -> dict:
    """A plane's HiPS properties (allsky/hips.py's vocabulary, so a plane
    exported as a tree needs no translation)."""
    lon, lat = lonlat_of(np.asarray(first.centre))
    width = tree_width(name, width)
    cell_deg = CELL_ARCSEC_NSIDE1 / (width << order) / 3600.0
    return {
        "creator_did": f"ivo://astroflow/{name}",
        "obs_title": f"{title} ({name})",
        "dataproduct_type": "image",
        "hips_version": "1.4",
        "hips_release_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "hips_status": "private master clonable",
        "hips_order": str(order),
        "hips_order_min": "0",
        "hips_tile_width": str(width),
        "hips_frame": "equatorial",
        "hips_pixel_bitpix": "16" if PLANES[name][0] is np.uint16 else "-32",
        "hips_pixel_scale": f"{cell_deg:.6g}",
        "hips_initial_ra": f"{float(lon):.5f}",
        "hips_initial_dec": f"{float(lat):.5f}",
        "hips_initial_fov": f"{2 * first.radius_deg:.3f}",
        "hips_creator": "AstroFlow allsky.reduce",
        "astroflow_coadd": "proper (Zackay & Ofek 2017); psffwhm per tile",
    }


def _degrade(raster: np.ndarray) -> np.ndarray:
    n = raster.shape[0]
    blocks = raster.reshape(n // 2, 2, n // 2, 2).transpose(0, 2, 1, 3).reshape(n // 2, n // 2, 4)
    return _nanmean(blocks, axis=-1).astype(np.float32)


def build_parents(plane: Plane, name: str, child_order: int, shard: int, parents: list[int], width: int) -> None:
    """The parent tiles `parents` -- all in `shard`, one order above
    `child_order` -- of plane `name`, each its four children (missing ones
    NaN) averaged 2 × 2: plain means of valid cells, which is the area mean
    on HEALPix's equal-area grid. A parent with no child left is removed.
    One write of the shard."""
    width = tree_width(name, width)
    half = width // 2
    tiles: dict[int, np.ndarray | None] = {}
    for p in parents:
        raster = np.full((width, width), np.nan, np.float32)
        any_child = False
        for i in range(4):
            child = plane.tile(child_order, 4 * p + i)
            if child is not None:
                any_child = True
                x0, y0 = (i & 1) * half, ((i >> 1) & 1) * half
                raster[y0:y0 + half, x0:x0 + half] = _degrade(child.astype(np.float32))
        tiles[p] = as_tile(name, raster) if any_child else None
    plane.write_block(child_order - 1, shard, tiles)


# --- the run

_job_masters: list[Master] = []
_job_order = 0
_job_width = TILE_WIDTH
_job_planes: dict[str, Plane] = {}


def _init_worker(masters, out, order, width, names):
    global _job_masters, _job_order, _job_width, _job_planes
    _job_masters, _job_order, _job_width = masters, order, width
    _job_planes = {name: Plane(Path(out) / name, "r+") for name in names}


def _run_parents(args) -> None:
    name, child_order, shard, parents = args
    build_parents(_job_planes[name], name, child_order, shard, parents, _job_width)


def _run_shard(args) -> list[tuple[int, bool]]:
    """One shard: reduce its dirty tiles, drop its gone ones, one write per
    plane. Returns (npix, landed) for every tile reduced -- not landed when
    no master had data under it."""
    shard, dirty, gone = args
    tiles: dict[str, dict[int, np.ndarray | None]] = {name: {npix: None for npix in gone} for name in _job_planes}
    fwhm: dict[int, float] = {}
    landed = []
    for npix, master_ids in dirty:
        reduced = reduce_tile(_job_order, npix, [_job_masters[i] for i in master_ids], _job_width)
        if reduced is None:
            for name in tiles:
                tiles[name][npix] = None
            landed.append((npix, False))
            continue
        planes, psf = reduced
        for name in tiles:
            raster = _degrade(planes[name]) if PLANES[name][1] else planes[name]
            tiles[name][npix] = as_tile(name, raster)
        fwhm[npix] = round(psf, 3)
        landed.append((npix, True))
    for name, plane in _job_planes.items():
        plane.write_block(_job_order, shard, tiles[name], {"psffwhm": fwhm} if name == "mean" else None)
    return landed


def order_for(pixel_arcsec: float, width: int) -> int:
    """The tile order whose cells are nearest the finest master's pixel."""
    return max(0, int(round(np.log2(CELL_ARCSEC_NSIDE1 / pixel_arcsec))) - int(np.log2(width)))


def tiles_touched(masters: list[Master], order: int, width: int) -> dict[int, list[int]]:
    """npix → the masters whose footprint can reach that tile, in NESTED order."""
    centres = tile_centres(order)
    # A tile's reach from its centre: cells are squares on the equator but
    # stretched rhombi in the polar caps, so allow a full side and a half --
    # of the tile with its margin.
    tile_radius = 1.5 * (1 + 2 * MARGIN / width) * np.degrees(np.sqrt(4 * np.pi / 12) / (1 << order))
    touched: dict[int, list[int]] = {}
    for i, m in enumerate(masters):
        near = np.nonzero(centres @ np.asarray(m.centre) >= np.cos(np.radians(m.radius_deg + tile_radius)))[0]
        for npix in near.tolist():
            touched.setdefault(npix, []).append(i)
    return dict(sorted(touched.items()))


def reduce_folder(masters_dir: Path, out: Path, *, pattern: str = "*.fits", order: int | None = None,
                  width: int = TILE_WIDTH, workers: int | None = None, title: str = "AstroFlow sky",
                  planes: tuple[str, ...] = TREES, level: bool = True, psf_fwhm: float | None = None,
                  log=print) -> dict:
    paths = sorted(p for p in Path(masters_dir).rglob("*") if p.is_file() and fnmatch.fnmatch(p.name, pattern))
    if not paths:
        raise FileNotFoundError(f"no masters matching {pattern!r} under {masters_dir}")
    t0 = time.time()
    masters = [open_master(p, psf_fwhm) for p in paths]
    masters = [m for m in masters if np.isfinite(m.sigma) and m.sigma > 0]
    if order is None:
        order = order_for(min(m.pixel_arcsec for m in masters), width)
    log(f"{len(masters)} masters, finest {min(m.pixel_arcsec for m in masters):.2f}\"/px → "
        f"tile order {order} ({CELL_ARCSEC_NSIDE1 / (width << order):.2f}\" cells), measured in {time.time() - t0:.0f} s")
    out = Path(out)
    # Incremental: `state.json` records the overlap pairs measured, the offsets
    # solved, and which masters (name, size, mtime, offset, PSF, weight) made
    # each finest tile; only tiles whose set changed are reduced again, and
    # only their ancestors rebuilt.
    ids = [master_id(m) for m in masters]
    state_path = out / "state.json"
    prev = json.loads(state_path.read_text()) if state_path.is_file() else {}
    if level:
        t0 = time.time()
        grid = [GRID_POINTS, MIN_OVERLAP_POINTS]
        known = prev.get("pairs", {}) if prev.get("pairs_grid") == grid else {}
        pairs, record = measure_pairs(masters, ids, known)
        offsets = solve_offsets(masters, ids, pairs)
        masters = [replace(m, offset=float(o)) for m, o in zip(masters, offsets)]
        log(f"{len(pairs)} overlapping pairs ({sum(k not in known for k in record)} new), "
            f"offsets span {offsets.max() - offsets.min():.3g} ({(offsets.max() - offsets.min()) / np.median([m.sigma for m in masters]):.0f} σ), "
            f"measured in {time.time() - t0:.0f} s")
    else:
        record = {}
    touched = tiles_touched(masters, order, width)
    same_grid = (prev.get("order"), prev.get("width")) == (order, width)
    prev_tiles = prev.get("tiles", {}) if same_grid else {}
    prev_empty = prev.get("empty", {}) if same_grid else {}      # touched by a footprint, no data under it
    wanted = {npix: sorted(f"{ids[i]}@{masters[i].offset:.6g}~{masters[i].fwhm_arcsec:.4g}/{masters[i].weight():.4g}"
                           for i in idx) for npix, idx in touched.items()}
    dirty = {npix: touched[npix] for npix, made_by in wanted.items()
             if prev_tiles.get(str(npix), prev_empty.get(str(npix))) != made_by}
    gone = {int(n) for n in prev_tiles} - set(touched)
    log(f"{len(touched)} tiles touched, {len(dirty)} to reduce, {len(gone)} to drop")
    tiles = {int(n): made_by for n, made_by in prev_tiles.items() if int(n) in touched}
    empty = {int(n): made_by for n, made_by in prev_empty.items() if int(n) in touched}
    if prev and not same_grid:
        # A new grid: nothing of the old planes would ever be reached again.
        import shutil
        for name in planes:
            shutil.rmtree(out / name, ignore_errors=True)
        log(f"grid was order {prev.get('order')}, width {prev.get('width')}: planes rebuilt")
    create(out, {name: (PLANES[name][0], tree_width(name, width)) for name in planes}, order,
           scalars={"mean": ("psffwhm",)})
    # A worker owns whole shards: the shard is the file it rewrites.
    by_shard: dict[int, tuple[list, list]] = {}
    for npix, idx in dirty.items():
        by_shard.setdefault(shard_of(npix), ([], []))[0].append((npix, idx))
    for npix in gone:
        by_shard.setdefault(shard_of(npix), ([], []))[1].append(npix)
    changed: set[int] = set(gone)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(workers or os.cpu_count(), initializer=_init_worker,
                             initargs=(masters, out, order, width, planes)) as pool:
        for landed in pool.map(_run_shard, [(s, d, g) for s, (d, g) in sorted(by_shard.items())], chunksize=1):
            for npix, ok in landed:
                changed.add(npix)
                if ok:
                    tiles[npix] = wanted[npix]
                    empty.pop(npix, None)
                else:
                    tiles.pop(npix, None)
                    empty[npix] = wanted[npix]
            if done // 500 != (done + len(landed)) // 500:
                i = done + len(landed)
                log(f"{i}/{len(dirty)} tiles, {(time.time() - t0) / i * (len(dirty) - i) / 60:.0f} min left")
            done += len(landed)
        log(f"{len(tiles)} tiles at order {order} ({len(changed)} changed) in {(time.time() - t0) / 60:.1f} min; pyramid...")
        # The coarser orders above the tiles that changed, one level at a
        # time (a level's parents are independent of each other, not of the
        # level below), every plane's parents in the same pool, by shard.
        level = set(changed)
        for k in range(order - 1, -1, -1):
            parents = sorted({c >> 2 for c in level})
            groups: dict[int, list[int]] = {}
            for p in parents:
                groups.setdefault(shard_of(p), []).append(p)
            list(pool.map(_run_parents, [(name, k + 1, s, ps) for name in planes for s, ps in groups.items()],
                          chunksize=4))
            level = set(parents)
    # Written LAST: it moves the map's version in the viewer, which caches
    # tiles as immutable under it, so it must move only once the planes are whole.
    for name in planes:
        Plane(out / name, "r+").write_properties(properties(name, order, width, masters[0], title))
    state_path.write_text(json.dumps({
        "order": order, "width": width, "pairs": record, "pairs_grid": [GRID_POINTS, MIN_OVERLAP_POINTS],
        "offsets": {ids[i]: m.offset for i, m in enumerate(masters)},
        "tiles": {str(n): made_by for n, made_by in sorted(tiles.items())},
        "empty": {str(n): made_by for n, made_by in sorted(empty.items())},
    }))
    log(f"done in {(time.time() - t0) / 60:.1f} min")
    return {"order": order, "tiles": len(tiles), "changed": len(changed), "masters": len(masters),
            "offsets": {ids[i]: m.offset for i, m in enumerate(masters)}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("masters")
    ap.add_argument("out")
    ap.add_argument("--pattern", default="*.fits")
    ap.add_argument("--order", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--title", default="AstroFlow sky")
    ap.add_argument("--planes", default=",".join(TREES), help="which planes to write: mean,noise,count")
    ap.add_argument("--no-level", action="store_true", help="skip the sky-offset solve (masters as they are)")
    ap.add_argument("--fresh", action="store_true", help="delete the output first: every tile from scratch")
    ap.add_argument("--psf-fwhm", type=float, help="PSF FWHM in arcsec for masters whose sidecar carries none")
    a = ap.parse_args(argv)
    planes = tuple(p for p in a.planes.split(",") if p)
    if set(planes) - set(TREES):
        ap.error(f"--planes must be among {', '.join(TREES)}")
    if a.fresh and Path(a.out).is_dir():
        import shutil
        shutil.rmtree(a.out)
    reduce_folder(Path(a.masters), Path(a.out), pattern=a.pattern, order=a.order, workers=a.workers,
                  title=a.title, planes=planes, level=not a.no_level, psf_fwhm=a.psf_fwhm,
                  log=lambda s: print(f"{time.strftime('%H:%M:%S')} {s}", flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
