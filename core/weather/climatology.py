"""The site's clear fraction by week of the year, from the ERA5 reanalysis.

First cut (design doc, 2026-10-12): ERA5 total cloud over the dark hours of
the past `years`, calibrated like a forecast (`probability.CALIBRATION`),
averaged per ISO week. Blending toward the user's own shot/not-shot history
comes later, behind the same `Callable[[datetime], float]` the planner reads.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta

import numpy as np

from weather.probability import calibrate
from weather.sources import SkyHour, era5

DARK_SUN_ALT_DEG = -12.0     # nautical dark: when a night's shooting starts
DEFAULT_YEARS = 10


def dark_mask(hours: list[SkyHour], lat: float, lon: float) -> np.ndarray:
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, get_sun
    from astropy.time import Time
    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)
    t = Time([h.time for h in hours])
    return get_sun(t).transform_to(AltAz(obstime=t, location=loc)).alt.deg < DARK_SUN_ALT_DEG


def by_week(hours: list[SkyHour], dark: np.ndarray) -> dict[int, float]:
    """ISO week -> mean calibrated p(clear) over the dark hours in it."""
    acc: dict[int, list[float]] = defaultdict(list)
    for h, is_dark in zip(hours, dark):
        if is_dark and h.cloud is not None:
            acc[h.time.isocalendar()[1]].append(calibrate("era5", h.cloud))
    return {week: float(np.mean(v)) for week, v in acc.items()}


def fetch(lat: float, lon: float, years: int = DEFAULT_YEARS, today: date | None = None) -> dict[int, float]:
    """ERA5 over the past `years` (it runs ~5 days behind real time)."""
    end = (today or date.today()) - timedelta(days=7)
    start = end.replace(year=end.year - years)
    hours = era5(lat, lon, start, end)
    return by_week(hours, dark_mask(hours, lat, lon))


def lookup(table: dict[int, float], fallback: float = 0.3):
    """`climatology(t)` for `probability.p_clear`: the week's value, a
    neighbour's when the week is missing, `fallback` when the table is empty."""
    def f(t: datetime) -> float:
        week = t.isocalendar()[1]
        for w in (week, week - 1, week + 1, 1, 52):
            if w in table:
                return table[w]
        return fallback
    return f
