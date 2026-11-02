"""Where a frame was shot, and the sites of a library.

A frame's position is its header's SITELAT/SITELONG (the ASIAir writes decimal
degrees; other software writes sexagesimal text such as `+37 47 00.00`). A
frame without one, or with a placeholder, belongs to no site and is never
given one — Stylistic7, 2026-11-02: data from unknown locations must not corrupt
known-good data, so nothing is backfilled or assumed.

Positions within SITE_RADIUS_DEG of each other are one site (single linkage):
the same roof reports 37.73 one night and 37.75 another. A site's key is its
cluster's median position rounded to 0.01°, e.g. `37.75,-122.42`.
"""
import math
from dataclasses import dataclass

import numpy as np

SITE_RADIUS_DEG = 0.05   # ~5 km


@dataclass(frozen=True)
class Site:
    key: str
    lat: float
    lon: float


def _degrees(value) -> float | None:
    """A header coordinate as degrees: a number, or sexagesimal text."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    parts = str(value).replace(":", " ").split()
    try:
        if len(parts) == 1:
            return float(parts[0])
        sign = -1.0 if parts[0].startswith("-") else 1.0
        d, m = abs(float(parts[0])), float(parts[1])
        s = float(parts[2]) if len(parts) > 2 else 0.0
        return sign * (d + m / 60 + s / 3600)
    except (ValueError, IndexError):
        return None


def site_of(header) -> tuple[float, float] | None:
    """The frame's (lat, lon) in degrees, or None when the header does not say."""
    lat, lon = _degrees(header.get("SITELAT")), _degrees(header.get("SITELONG"))
    if lat is None or lon is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return None
    if lat == 0 and lon == 0:   # the placeholder some software writes
        return None
    return (round(lat, 4), round(lon, 4))


def same_site(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return math.hypot(a[0] - b[0], (a[1] - b[1]) * math.cos(math.radians(a[0]))) < SITE_RADIUS_DEG


def site_key(lat: float, lon: float) -> str:
    return f"{lat:.2f},{lon:.2f}"


def cluster_sites(positions) -> dict[tuple[float, float], Site]:
    """Each distinct position -> its Site: single linkage within SITE_RADIUS_DEG."""
    points = sorted(set(positions))
    if not points:
        return {}
    parent = list(range(len(points)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, p in enumerate(points):
        for j in range(i + 1, len(points)):
            if same_site(p, points[j]):
                ri, rj = root(i), root(j)
                if ri != rj:
                    parent[rj] = ri
    members: dict[int, list[int]] = {}
    for i in range(len(points)):
        members.setdefault(root(i), []).append(i)
    out = {}
    for idx in members.values():
        lat = round(float(np.median([points[i][0] for i in idx])), 2)
        lon = round(float(np.median([points[i][1] for i in idx])), 2)
        site = Site(site_key(lat, lon), lat, lon)
        for i in idx:
            out[points[i]] = site
    return out
