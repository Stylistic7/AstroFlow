"""Whole-sky HEALPix maps for the sky viewer.

A map is a HEALPix FITS file in `~/AstroFlow/healpix/<name>.fits` (the same
user folder the finishes and pipelines live under). It is read once per
process into twelve face rasters on AstroFlow's pinned gauge -- scaled so the
sky median sits on the stacks' background pedestal, or lower when the map's
bright anchor would otherwise overshoot the finishes' declared range -- with
the per-band statistics every finish seeds from, so the stock finishes see a
map as they see a master: a sky level and an anchor inside their ranges.
"""
from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from allsky.healpix_faces import faces_from_nested
from previews.linear_stats import band_stats
from stacks.post_processing.background_pedestal import DEFAULT_PEDESTAL

MAPS_DIR = Path.home() / "AstroFlow" / "healpix"

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
    nside: int
    frame: str          # "galactic" | "icrs": the frame the cells are laid out in
    unit: str           # the file's own unit, for the label only
    gauge: float        # file units × gauge = the served (pinned-gauge) values
    faces: np.ndarray   # (12, nside, nside) float32, gauge units, 0 = no data
    stats: dict         # band_stats of the gauge-scaled map, one band


def map_path(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise FileNotFoundError(name)
    return MAPS_DIR / f"{name}.fits"


def available_maps() -> list[str]:
    if not MAPS_DIR.is_dir():
        return []
    return sorted(p.stem for p in MAPS_DIR.glob("*.fits"))


def _frame(coordsys) -> str:
    key = (str(coordsys or "G")).strip().upper()
    if key in ("G", "GAL", "GALACTIC"):
        return "galactic"
    if key in ("C", "Q", "EQU", "EQUATORIAL", "ICRS", "FK5", "J2000"):
        return "icrs"
    raise ValueError(f"unsupported HEALPix COORDSYS {coordsys!r}")


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
        frame = _frame(hd.get("COORDSYS"))
        unit = str(hd.get("TUNIT1") or "")
    if values.size != 12 * nside * nside:
        raise ValueError(f"{path.name}: {values.size} cells is not a nside={nside} map")
    return values, nside, frame, unit


def _build(name: str, path: Path, mtime: float) -> SkyMap:
    values, nside, frame, unit = read_healpix_fits(path)
    # Unobserved cells become the viewer's no-data sentinel (exact zero).
    values = np.where(np.isfinite(values) & (values > _UNSEEN / 2), values, 0.0).astype(np.float32)
    raw = band_stats(values[:, None])[0]
    sky, high = raw["skyMedian"], raw["high"]
    gauge = DEFAULT_PEDESTAL / sky if sky > 0 else 1.0
    if high * gauge > TOP_CEILING:
        gauge = TOP_CEILING / high
    stats = {k: v * gauge for k, v in raw.items()}
    return SkyMap(name=name, path=path, mtime=mtime, nside=nside, frame=frame, unit=unit,
                  gauge=gauge, faces=faces_from_nested(values * np.float32(gauge), nside),
                  stats=stats)


_cache: dict[str, SkyMap] = {}
_lock = threading.Lock()


def load_map(name: str) -> SkyMap:
    """The map, built on first use and again when its file changes."""
    path = map_path(name)
    mtime = os.path.getmtime(path)          # FileNotFoundError when absent
    with _lock:
        m = _cache.get(name)
        if m is None or m.mtime != mtime:
            m = _build(name, path, mtime)
            _cache[name] = m
        return m
