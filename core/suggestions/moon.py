"""Moon-phase helpers for the Suggestions feature.

The philosophy: advisory only. Compute illumination + altitude at local
midnight, recommend a filter set, but let the user override anything.
"""
from dataclasses import dataclass
from datetime import date, datetime, time, timezone

import astropy.units as u  # type: ignore
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body  # type: ignore
from astropy.time import Time  # type: ignore


@dataclass(frozen=True)
class MoonConditions:
    illumination: float
    altitude_at_midnight_deg: float
    recommended_filters: frozenset[str]


def _midnight_utc(for_date: date, observer: EarthLocation) -> Time:
    """UTC instant of local midnight at the observer's longitude.

    Approximates local midnight using the observer's longitude (no DST / tz
    data needed — good enough for the moon, which moves ~0.5 deg/hour)."""
    lon_deg = float(observer.lon.to_value(u.deg))
    naive = datetime.combine(for_date, time(0, 0)).replace(tzinfo=timezone.utc)
    # Subtract the longitude offset: local midnight = UTC midnight - lon/15 hours.
    offset_hours = lon_deg / 15.0
    midnight_utc = naive.timestamp() - offset_hours * 3600.0
    return Time(midnight_utc, format="unix")


def moon_illumination(for_date: date, observer: EarthLocation) -> float:
    """Illuminated fraction [0, 1] of the moon at local midnight."""
    t = _midnight_utc(for_date, observer)
    moon = get_body("moon", t, observer)
    sun = get_body("sun", t, observer)
    elong = sun.separation(moon).rad
    import math
    # Phase angle: angle sun-moon-earth. cos(phase) ≈ -cos(elong) for far sun.
    phase_angle = math.pi - elong
    # Illuminated fraction = (1 + cos(phase_angle)) / 2.
    return float((1.0 + math.cos(phase_angle)) / 2.0)


def moon_altitude_at_midnight(for_date: date, observer: EarthLocation) -> float:
    """Moon altitude (degrees above horizon) at local midnight."""
    t = _midnight_utc(for_date, observer)
    moon = get_body("moon", t, observer)
    altaz = moon.transform_to(AltAz(obstime=t, location=observer))
    return float(altaz.alt.to_value(u.deg))


def recommend_filters(illum: float, altitude_deg: float) -> frozenset[str]:
    """Suggest filters to shoot tonight given moon conditions.

    Moon below horizon OR dim → all filters.
    Moon up AND moderate (0.3 < illum < 0.7) → narrowband only (H, S, O).
    Moon up AND bright (illum > 0.7) → only H, S — OIII is too close to
        the moon's blue-green light to come out clean."""
    ALL = frozenset({"H", "S", "O", "L", "R", "G", "B"})
    if altitude_deg < 0 or illum < 0.3:
        return ALL
    if illum > 0.7:
        return frozenset({"H", "S"})
    return frozenset({"H", "S", "O"})


def compute_conditions(for_date: date, observer: EarthLocation) -> MoonConditions:
    illum = moon_illumination(for_date, observer)
    alt = moon_altitude_at_midnight(for_date, observer)
    return MoonConditions(
        illumination=illum,
        altitude_at_midnight_deg=alt,
        recommended_filters=recommend_filters(illum, alt),
    )
