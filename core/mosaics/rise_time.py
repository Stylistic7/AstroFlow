"""Panel ordering and rise-time computation for mosaics.

`panel_order` is observer-independent — sort by RA, then rotate so the
sequence starts right after the largest RA gap. This means the same
mosaic gets the same panel numbers anywhere on Earth, and panels in a
strip that crosses the 0h/24h RA boundary are still numbered
contiguously instead of split at zero. The numbering is stored on
`Mosaic.panel_target_ids` and used by both the polygon labels in the
mosaic viewer and the "Panel N" prefix in the Telescopius / text
export.

`next_rise_utc` is observer-dependent (longitude + latitude + time)
and is computed on-demand for display in the mosaic viewer's panel
labels — it isn't stored anywhere.
"""
import datetime
import numpy as np


# Sidereal vs solar day ratio — divide sidereal hours by this to get
# wall-clock hours.
_SIDEREAL_PER_SOLAR = 1.00273790935


def panel_order(
    items: list[tuple[str, float]],
) -> tuple[str, ...]:
    """Order targets by RA, rotated so the sequence starts after the
    largest RA gap. Input is a list of (target_id, ra_deg) pairs;
    output is a tuple of target_ids in panel order."""
    if len(items) <= 1:
        return tuple(t for t, _ in items)

    by_ra = sorted(items, key=lambda x: x[1])
    n = len(by_ra)
    ras = [r for _, r in by_ra]
    # Forward gap from each RA to the next (with wraparound).
    gaps = [(ras[(i + 1) % n] - ras[i]) % 360.0 for i in range(n)]
    largest_gap = max(range(n), key=lambda i: gaps[i])
    start = (largest_gap + 1) % n
    return tuple(t for t, _ in by_ra[start:] + by_ra[:start])


def next_rise_utc(
    ra_deg: float,
    dec_deg: float,
    observer_lat_rad: float,
    lst_at_ref_hours: float,
    reference_utc: datetime.datetime,
    altitude_threshold_deg: float = 0.0,
) -> datetime.datetime | None:
    """Next UTC moment after reference_utc when the target crosses
    `altitude_threshold_deg` ascending. Defaults to the mathematical
    horizon (0°); pass e.g. 15° to get the "usable altitude" rise that
    matches what astrophotography planning tools typically report.

    Returns None if the target never reaches that altitude (or is
    always above it) at this observer's latitude.
    """
    alt_rad = np.deg2rad(altitude_threshold_deg)
    dec_rad = np.deg2rad(dec_deg)
    cos_ha = (
        np.sin(alt_rad) - np.sin(observer_lat_rad) * np.sin(dec_rad)
    ) / (np.cos(observer_lat_rad) * np.cos(dec_rad))
    if cos_ha <= -1 or cos_ha >= 1:
        return None
    ha_rise_hours = np.degrees(np.arccos(cos_ha)) / 15.0
    lst_at_rise = (ra_deg / 15.0 - ha_rise_hours) % 24
    hours_sidereal = (lst_at_rise - lst_at_ref_hours) % 24
    hours_solar = hours_sidereal / _SIDEREAL_PER_SOLAR
    return reference_utc + datetime.timedelta(hours=hours_solar)
