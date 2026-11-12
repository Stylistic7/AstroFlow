"""Whole-sky HEALPix maps for the sky viewer.

A map is one of two files in `~/AstroFlow/healpix` (the user folder the
finishes and pipelines live under):

* `<name>.fits` -- a flat HEALPix map (a binary table, one row per cell,
  NESTED), read once per process into a pyramid of face rasters and served
  to the viewer as HiPS-shaped tiles;
* `<name>.hips` -- one line, the root of a HiPS tree (a URL or a folder) or
  of a packed plane (allsky/tiles.py, the combiner's output); the browser
  reads a remote tree's tiles itself (allsky/hips.py).

Either way the map carries a gauge and the per-band statistics every finish
seeds from. The gauge scales file units so the sky median sits on the
stacks' background pedestal, or lower when the map's bright anchor would
otherwise overshoot the finishes' declared range, so the stock finishes see
a map as they see a master: a sky level and an anchor inside their ranges.
Tiles stay in file units; the shader applies the gauge.
"""
from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from allsky.healpix_faces import compact_bits, degrade, faces_from_nested
from previews.linear_stats import band_stats
from stacks.post_processing.background_pedestal import DEFAULT_PEDESTAL

MAPS_DIR = Path.home() / "AstroFlow" / "healpix"

# A flat map's tiles: the HiPS tile side, or the whole face of a smaller map.
TILE_WIDTH = 512

# The stock finishes' declared ceiling for the bright anchor they seed from
# (`top<role>` in finishes/stock.py, hi=0.5): a stack's anchor sits a few
# stops above its sky, a whole-sky map's can sit eight. The gauge keeps the
# anchor under this ceiling so the seed survives the finish's own clamp.
TOP_CEILING = 0.5

# HEALPix's "no observation" value (the HEALPix library's UNSEEN).
_UNSEEN = -1.6375e30


@dataclass(frozen=True)
class SkyMap:
    name: str
    path: Path
    mtime: float
    frame: str          # "galactic" | "icrs": the frame the cells are laid out in
    unit: str           # the file's own unit, for the label only
    gauge: float        # file units × gauge = pinned-gauge values (the shader applies it)
    stats: dict         # band_stats of the gauge-scaled map, one band
    tile_width: int     # a tile is tile_width² cells, log2(tile_width) orders below its own
    max_order: int      # the finest tile order
    tiles: str = ""     # a HiPS root the browser reads, or "" when this app serves `tile()`
    pyramid: tuple = () # flat maps: (12, n, n) face rasters per tile order, coarse to fine
    tree: object = None # folder trees: the allsky.hips.HipsTree the tiles are read from

    @property
    def nside(self) -> int:
        return self.tile_width << self.max_order

    def tile(self, order: int, npix: int) -> np.ndarray:
        """The (tile_width, tile_width) raster of cell `npix` at `order`, in
        file units with 0 = no data; IndexError off the tree."""
        if not 0 <= order <= self.max_order or not 0 <= npix < 12 << (2 * order):
            raise IndexError((order, npix))
        if self.pyramid:
            face = npix >> (2 * order)
            within = np.array([npix & ((1 << (2 * order)) - 1)], dtype=np.uint64)
            tx = int(compact_bits(within)[0])
            ty = int(compact_bits(within >> np.uint64(1))[0])
            w = self.tile_width
            return self.pyramid[order][face, ty * w:(ty + 1) * w, tx * w:(tx + 1) * w]
        if self.tree is not None and not self.tree.remote:
            raster = self.tree.tile(order, npix)
            if raster is None:
                raise IndexError((order, npix))
            return np.where(np.isfinite(raster), raster, 0.0).astype(np.float32)
        raise IndexError((order, npix))


def map_path(name: str) -> Path:
    if re.fullmatch(r"[A-Za-z0-9_-]+", name):
        for ext in (".fits", ".hips"):
            p = MAPS_DIR / f"{name}{ext}"
            if p.is_file():
                return p
    raise FileNotFoundError(name)


def available_maps() -> list[str]:
    if not MAPS_DIR.is_dir():
        return []
    return sorted({p.stem for p in MAPS_DIR.iterdir() if p.suffix in (".fits", ".hips")})


def frame_of(coordsys) -> str:
    key = (str(coordsys or "G")).strip().upper()
    if key in ("G", "GAL", "GALACTIC"):
        return "galactic"
    if key in ("C", "Q", "EQU", "EQUATORIAL", "ICRS", "FK5", "J2000"):
        return "icrs"
    raise ValueError(f"unsupported sky frame {coordsys!r}")


def read_healpix_fits(path: Path) -> tuple[np.ndarray, int, str, str]:
    """`(values, nside, frame, unit)` of the first column of a HEALPix binary
    table. NESTED only: that is the order the face rasters come from, and the
    maps this is for (LAMBDA's foreground templates) ship nested."""
    from astropy.io import fits
    with fits.open(path, memmap=True) as h:
        hdu = h[1]
        hd = hdu.header
        nside = int(hd["NSIDE"])
        ordering = str(hd.get("ORDERING", "")).strip().upper()
        if ordering != "NESTED":
            raise ValueError(f"{path.name}: ORDERING={ordering!r}; only NESTED maps are supported")
        values = np.asarray(hdu.data.field(0), dtype=np.float32).ravel()
        frame = frame_of(hd.get("COORDSYS"))
        unit = str(hd.get("TUNIT1") or "")
    if values.size != 12 * nside * nside:
        raise ValueError(f"{path.name}: {values.size} cells is not a nside={nside} map")
    return values, nside, frame, unit


def gauge_and_stats(values: np.ndarray) -> tuple[float, dict]:
    """The gauge for a map's cells (file units, 0 = no data) and the
    statistics of the gauged map that the finishes seed from."""
    raw = band_stats(values[:, None])[0]
    sky, high = raw["skyMedian"], raw["high"]
    gauge = DEFAULT_PEDESTAL / sky if sky > 0 else 1.0
    if high * gauge > TOP_CEILING:
        gauge = TOP_CEILING / high
    return gauge, {k: v * gauge for k, v in raw.items()}


def _build_fits(name: str, path: Path, mtime: float) -> SkyMap:
    values, nside, frame, unit = read_healpix_fits(path)
    # Unobserved cells become the viewer's no-data sentinel (exact zero).
    values = np.where(np.isfinite(values) & (values > _UNSEEN / 2), values, 0.0).astype(np.float32)
    gauge, stats = gauge_and_stats(values)
    width = min(TILE_WIDTH, nside)
    levels = [faces_from_nested(values, nside)]
    while levels[-1].shape[1] > width:
        levels.append(degrade(levels[-1]))
    levels.reverse()
    return SkyMap(name=name, path=path, mtime=mtime, frame=frame, unit=unit, gauge=gauge,
                  stats=stats, tile_width=width, max_order=len(levels) - 1,
                  pyramid=tuple(levels))


_cache: dict[str, SkyMap] = {}
_lock = threading.Lock()


def map_mtime(path: Path) -> float:
    """When the map last changed: its file, or for a local source also its
    properties -- a tree's `properties` file, a packed plane's `zarr.json` --
    which the combiner rewrites last on every run. The viewer caches tiles
    as immutable under this version, so a source that grows under the same
    name must move it."""
    mtime = os.path.getmtime(path)
    if path.suffix == ".hips":
        root = path.read_text().strip()
        if "://" not in root:
            for marker in ("properties", "zarr.json"):
                p = Path(root) / marker
                if p.is_file():
                    mtime = max(mtime, p.stat().st_mtime)
    return mtime


def load_map(name: str) -> SkyMap:
    """The map, built on first use and again when its file or its tree changes."""
    path = map_path(name)
    mtime = map_mtime(path)
    with _lock:
        m = _cache.get(name)
        if m is None or m.mtime != mtime:
            if path.suffix == ".hips":
                from allsky.hips import load_hips
                m = load_hips(name, path, mtime)
            else:
                m = _build_fits(name, path, mtime)
            _cache[name] = m
        return m
