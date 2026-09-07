"""Per-star Gaia photometry of a plate-solved raw subframe, persisted in its
OWN sidecar section — the raw material of the star flat (`stacks/starflat.py`).
Runs as the step after a successful plate solve on every raw subframe
(`compute_image_metadata`), through `photometry.harvest` (forced photometry
at Gaia positions, no PixInsight; ~0.4 s per sub). Schema history: v1 =
PixInsight SPFC; v2 = the engine before PI's relative saturation rule and
the wide-PSF annulus; v3 = fixed stamp geometry, flat "ideal QE" L passband,
outlier-fraction gate; v4 = FWHM-adaptive geometry, L = R/G/B envelope,
scatter gate; v5 = current (robust FWHM probe, gate over 8-13.5). Older
sections are recomputed by the sweep.

Section `star_photometry`, shaped like `plate_solution`:

    {"schema": SCHEMA, "data": {"n", "columns": {name: [...]}, provenance}}
    {"schema": SCHEMA, "data": None, "reason": ...}     nothing to harvest

`reason` "no_plate_solution" is provisional (a solution may arrive later and
the next call harvests); "no_match" is final for this schema (too few usable
stars, or a wrong WCS: the per-star scatter about the colour term exceeds
`stacks.starflat._MAX_SCATTER`). Bumping SCHEMA recomputes every sub.

Deliberately NOT a field of `ImageMetadata`: metadata values are hashed into
downstream cache keys, so a new field would re-key every stack in the
system (the reason `defect_candidates` was lifted off it too). A separate
section changes no key; the star flat reads it on demand.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import structlog

from file_io.generic import read_header
from file_io.sidecar import get_section, set_section

log = structlog.get_logger()

SCHEMA = "astroflow.starphotometry.v5"
SECTION = "star_photometry"
# Column order of the arrays read back; `scale` is the image/catalogue flux
# ratio (∝ the sensor's response at that pixel, exp(+F)).
COLUMNS = ("x", "y", "mag_G", "colour", "fwhm", "signal", "scale", "snr")
_NO_SOLUTION = "no_plate_solution"
_NO_MATCH = "no_match"
# (decimals, or None → 6 significant figures) per column; keeps a 2,000-star
# section near 100 KB.
_ROUND = {"x": 2, "y": 2, "mag_G": 3, "colour": 3, "fwhm": 2, "signal": None, "scale": None, "snr": 1}


def _to_obj(stars: np.ndarray, info: dict[str, Any]) -> dict[str, Any]:
    """`stars` is harvest_sub_xp's (n, 10) layout: x, y, ra, dec, G, colour,
    fwhm, signal, scale, snr. Rows with a non-finite value are dropped (a NaN
    colour would poison the joint solve)."""
    cols = {
        "x": stars[:, 0], "y": stars[:, 1], "mag_G": stars[:, 4], "colour": stars[:, 5],
        "fwhm": stars[:, 6], "signal": stars[:, 7], "scale": stars[:, 8], "snr": stars[:, 9],
    }
    keep = np.all([np.isfinite(v) for v in cols.values()], axis=0) & (cols["scale"] > 0)
    out = {}
    for name, v in cols.items():
        d = _ROUND[name]
        v = v[keep]
        out[name] = [round(float(x), d) for x in v] if d is not None else [float(f"{x:.6g}") for x in v]
    return {"schema": SCHEMA, "data": {"n": int(keep.sum()), "columns": out, **info}}


def _from_obj(obj: dict[str, Any]) -> np.ndarray:
    c = obj["data"]["columns"]
    return np.column_stack([np.asarray(c[k], dtype=float) for k in COLUMNS]).reshape(-1, len(COLUMNS))


def write_star_photometry(filename: Path, stars: np.ndarray, info: dict[str, Any]) -> None:
    set_section(filename, SECTION, _to_obj(stars, info))


def read_star_photometry(filename: Path) -> np.ndarray | None:
    """(n, 8) array in COLUMNS order, or None when the sub has no harvested
    photometry (never triggers a harvest)."""
    obj = get_section(filename, SECTION)
    if isinstance(obj, dict) and obj.get("schema") == SCHEMA and obj.get("data") is not None:
        return _from_obj(obj)
    return None


def compute_star_photometry(filename: Path) -> np.ndarray | None:
    """Sidecar first; otherwise measure the sub and persist the result.
    Returns the (n, 8) array, or None when there is nothing to harvest.
    Raises on a transient failure (unreadable file, missing catalogue): no
    marker is written, so the next sweep retries."""
    obj = get_section(filename, SECTION)
    if isinstance(obj, dict) and obj.get("schema") == SCHEMA:
        if obj.get("data") is not None:
            return _from_obj(obj)
        if obj.get("reason") != _NO_SOLUTION:
            return None

    ps = get_section(filename, "plate_solution")
    if not (isinstance(ps, dict) and ps.get("data")):
        set_section(filename, SECTION, {"schema": SCHEMA, "data": None, "reason": _NO_SOLUTION})
        return None

    from photometry.harvest import harvest_sub_xp   # loads the Gaia catalogue on first use
    stars, info = harvest_sub_xp(filename, filter=read_header(filename).get("FILTER"))
    if not info.pop("good"):
        log.warning("star photometry: too few usable stars or a wrong WCS; not stored",
                    filename=str(filename), **info)
        set_section(filename, SECTION, {"schema": SCHEMA, "data": None, "reason": _NO_MATCH, **info})
        return None
    write_star_photometry(filename, stars, info)
    log.info("star photometry harvested", filename=str(filename), n=len(stars), **info)
    return read_star_photometry(filename)
