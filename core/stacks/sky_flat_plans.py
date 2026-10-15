"""Night-sky flat plans: where to point, how to name the panes, and how a
stack of them is recognised again.

A night-sky flat is a median of blank-sky frames taken at slightly different
pointings, so stars move between frames and drop out of the median while the
optics' response (vignetting, dust, etalon) stays put. `stacks/sky_flat.py`
builds it from a Stack; this module makes it easy to SHOOT one:

  * FIELDS — `sky_flat_fields.json`, written by
    `scripts/build_sky_flat_fields.py`: blank fields over the whole sky, in
    two tiers by how much clear sky surrounds them (WIDE for short focal
    lengths, NARROW for long ones).
  * RANKING — for a rig, a night and a site: the fields usable longest after
    astronomical dark, high and away from the moon.
  * PANES — five pointings around the field, offset by a few percent of the
    rig's short field of view: far enough that stars move hundreds of
    pixels, close enough that all frames group into ONE Stack (grouping
    admits 10% of the short side from the first frame).
  * NAMES — every pane is named `SkyFlat <filter> <field> <n>`. The ASIAir
    writes that name into each frame's file name, and a Stack whose frames
    carry it is a night-sky flat source without anyone promoting it
    (`is_named_sky_flat`; a user can still demote it).
"""
from __future__ import annotations

import datetime as _dt
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

FIELDS_PATH = Path(__file__).with_name("sky_flat_fields.json")
NAME_PREFIX = "SkyFlat"
PANES = 5
# Radial pane offset as a fraction of the short side: under half the 10%
# grouping tolerance, so every frame lands in the first frame's Stack.
OFFSET_FRACTION = 0.04
# Wall-clock cost per frame beyond its exposure (download, dither, settle).
OVERHEAD_S = 60.0
MIN_ALT_DEG = 40.0
MIN_MOON_SEP_DEG = 60.0
DARK_SUN_ALT_DEG = -18.0


@lru_cache(maxsize=1)
def fields() -> tuple[dict, ...]:
    return tuple(json.loads(FIELDS_PATH.read_text())["fields"])


def field_by_id(field_id: str) -> dict:
    for f in fields():
        if f["id"] == field_id:
            return f
    raise KeyError(f"no sky-flat field {field_id!r}")


def field_of_view_deg(focal_mm: float, pixel_um: float, sensor: tuple[int, int]) -> tuple[float, float]:
    """(long side, short side) in degrees."""
    scale = 206.265 * pixel_um / focal_mm / 3600.0
    w, h = sensor
    return max(w, h) * scale, min(w, h) * scale


def pane_names(filt: str, field_id: str) -> list[str]:
    return [f"{NAME_PREFIX} {filt} {field_id} {i}" for i in range(1, PANES + 1)]


def is_sky_flat_name(name: str) -> bool:
    return str(name).strip().lower().startswith(NAME_PREFIX.lower() + " ")


def frame_target_name(filename: str) -> str:
    """The ASIAir target name in a frame's file name: `Light_<name>_<exp>s_...`."""
    base = Path(str(filename)).name
    if not base.startswith("Light_"):
        return ""
    rest = base[len("Light_"):]
    parts = rest.split("_")
    for i, p in enumerate(parts):
        if p.endswith("s") and p[:-1].replace(".", "", 1).isdigit():
            return "_".join(parts[:i])
    return ""


def is_named_sky_flat(filenames) -> bool:
    """True when most of a Stack's frames were shot as SkyFlat panes."""
    names = [frame_target_name(f) for f in filenames]
    return bool(names) and sum(is_sky_flat_name(n) for n in names) * 2 > len(names)


def pane_centres(field: dict, short_side_deg: float) -> list[tuple[float, float]]:
    """J2000 (ra, dec) of the five panes: the field centre, then the four
    diagonals at OFFSET_FRACTION of the short side, in the field's own
    tangent plane."""
    d = OFFSET_FRACTION * short_side_deg / np.sqrt(2.0)
    ra0, de0 = np.radians(field["ra_deg"]), np.radians(field["dec_deg"])
    c = np.array([np.cos(de0) * np.cos(ra0), np.cos(de0) * np.sin(ra0), np.sin(de0)])
    e = np.array([-np.sin(ra0), np.cos(ra0), 0.0])
    n = np.cross(c, e)
    out = []
    for dx, dy in ((0, 0), (d, d), (-d, d), (-d, -d), (d, -d)):
        v = c + np.radians(dx) * e + np.radians(dy) * n
        v /= np.linalg.norm(v)
        out.append((float(np.degrees(np.arctan2(v[1], v[0])) % 360.0), float(np.degrees(np.arcsin(v[2])))))
    return out


def plan_minutes(exposure_s: float, frames_per_pane: int) -> float:
    return PANES * frames_per_pane * (exposure_s + OVERHEAD_S) / 60.0


def _night_grid(date: _dt.date, lon: float, lat: float, step_min: int = 10):
    """Times (UTC) of astronomical darkness the night beginning on `date` at
    the site, with the moon's altitude and position at each."""
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, get_body, get_sun
    from astropy.time import Time
    site = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)
    # local noon -> next local noon, from the longitude
    noon = Time(_dt.datetime(date.year, date.month, date.day, 12)) - (lon / 15.0) * u.hour
    t = noon + np.arange(0, 24 * 60, step_min) * u.min
    frame = AltAz(obstime=t, location=site)
    dark = get_sun(t).transform_to(frame).alt.deg < DARK_SUN_ALT_DEG
    moon = get_body("moon", t, site)
    return t[dark], frame[dark], moon[dark], moon[dark].transform_to(frame[dark]).alt.deg


def rank_fields(focal_mm: float, pixel_um: float, sensor: tuple[int, int],
                date: _dt.date, lon: float, lat: float, minutes_needed: float,
                limit: int = 6) -> list[dict]:
    """The fields best placed on `date`'s night for this rig, best first.

    A field is usable at a moment when it is above MIN_ALT_DEG in
    astronomical darkness and the moon is down or at least MIN_MOON_SEP_DEG
    away. Fields whose clear radius covers the rig's half-diagonal come
    first (`fits`); a narrow field on a wide rig may have something bright
    near the frame's corners."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    t, frame, moon, moon_alt = _night_grid(date, lon, lat)
    if len(t) == 0:
        return []
    moon = moon.icrs        # separations in one frame, without astropy's direction warning
    long_side, short_side = field_of_view_deg(focal_mm, pixel_um, sensor)
    half_diag = 0.5 * float(np.hypot(long_side, short_side))
    step_h = (t[1] - t[0]).to(u.hour).value if len(t) > 1 else 1 / 6
    out = []
    for f in fields():
        c = SkyCoord(f["ra_deg"] * u.deg, f["dec_deg"] * u.deg)
        alt = c.transform_to(frame).alt.deg
        sep = moon.separation(c).deg
        ok = (alt >= MIN_ALT_DEG) & ((moon_alt < 0) | (sep >= MIN_MOON_SEP_DEG))
        if not ok.any():
            continue
        # longest contiguous usable run
        best_len, best_start, run, start = 0, 0, 0, 0
        for i, v in enumerate(ok):
            if v:
                if run == 0:
                    start = i
                run += 1
                if run > best_len:
                    best_len, best_start = run, start
            else:
                run = 0
        window_h = best_len * step_h
        out.append({
            **f,
            "fits": bool(f["clear_radius_deg"] >= half_diag),
            "window_h": round(float(window_h), 2),
            "window_start_utc": t[best_start].isot[:16] + "Z",
            "window_end_utc": (t[best_start + best_len - 1] + step_h * u.hour).isot[:16] + "Z",
            "max_alt_deg": round(float(alt[ok].max()), 1),
            "long_enough": bool(window_h * 60 >= minutes_needed),
        })
    out.sort(key=lambda r: (not r["fits"], not r["long_enough"], -r["window_h"], r["stars_brighter_than_g11"]))
    return out[:limit]
