"""Supply for Layer 1: the dark sky of the coming season in blocks, per rig,
and which (panel, filter) each block can serve.

Fine blocks (BLOCK_MIN) for the next FINE_NIGHTS, so a night's moon-down
and altitude windows are resolved; coarse blocks (COARSE_BLOCK_MIN) from
there to SEASON_NIGHTS, enough to see a target's hours run out for the
season without anyone typing a date. One astropy pass covers the whole
horizon: the sun and moon on the block grid, then one track per target
over every dark block. `efficiency` (the v1 altitude and moon factors)
decides whether a block can serve a need and how much an hour there is
worth; `weather.probability` decides how much of the block is expected to
exist.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
from astropy.time import Time

from planner.allocate import Edge
from planner.efficiency import efficiency
from planner.model import Candidate, Rig
from planner.sky import Track

FINE_NIGHTS = 30          # a lunar cycle at full resolution: moonlit nights are in view
SEASON_NIGHTS = 180       # half a year: every cart target's season ends inside it
BLOCK_MIN = 30
COARSE_BLOCK_MIN = 120
DARK_SUN_ALT_DEG = -18.0


@dataclass(frozen=True)
class Horizon:
    location: EarthLocation
    times: Time               # dark block centres across the horizon
    hours: np.ndarray         # per block: its length in hours (fine or coarse)
    night: np.ndarray         # per block: 0 = the first night
    moon_alt: np.ndarray
    moon_illum: np.ndarray
    moon: SkyCoord

    @property
    def n(self) -> int:
        return len(self.times)

    def track(self, ra_deg: float, dec_deg: float) -> Track:
        c = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
        alt = c.transform_to(AltAz(obstime=self.times, location=self.location)).alt.deg
        lst = self.times.sidereal_time("apparent", longitude=self.location.lon).hour
        ha = (lst - ra_deg / 15.0 + 12.0) % 24.0 - 12.0
        return Track(alt=alt, hour_angle=ha, moon_sep=self.moon.separation(c).deg)

    def utc(self) -> list[datetime]:
        from datetime import timezone
        return [t.replace(microsecond=0) for t in self.times.to_datetime(timezone=timezone.utc)]


def dark_blocks(lon_deg: float, lat_deg: float, noon_utc: datetime,
                nights: int = SEASON_NIGHTS, fine_nights: int = FINE_NIGHTS,
                block_min: int = BLOCK_MIN, coarse_block_min: int = COARSE_BLOCK_MIN) -> Horizon:
    """Every block of astronomical darkness in the `nights` after `noon_utc`:
    `block_min` long for the first `fine_nights`, `coarse_block_min` after."""
    loc = EarthLocation(lon=lon_deg * u.deg, lat=lat_deg * u.deg)
    fine = np.arange(0, min(nights, fine_nights) * 24 * 60, block_min) + block_min / 2
    coarse = np.arange(fine_nights * 24 * 60, nights * 24 * 60, coarse_block_min) + coarse_block_min / 2
    offsets = np.concatenate([fine, coarse])
    length_h = np.concatenate([np.full(len(fine), block_min / 60.0), np.full(len(coarse), coarse_block_min / 60.0)])
    grid = Time(noon_utc) + offsets * u.min
    sun = get_sun(grid)
    dark = sun.transform_to(AltAz(obstime=grid, location=loc)).alt.deg < DARK_SUN_ALT_DEG
    times = grid[dark]
    moon = get_body("moon", times, loc)
    return Horizon(location=loc, times=times, hours=length_h[dark],
                   night=(offsets[dark] // (24 * 60)).astype(int),
                   moon_alt=moon.transform_to(AltAz(obstime=times, location=loc)).alt.deg,
                   moon_illum=(1 - np.cos(sun[dark].separation(moon).rad)) / 2,
                   moon=moon)


def night_dates(noon_utc: datetime, lon_deg: float, nights: int) -> list[date]:
    """The local calendar date each night index starts on (the evening's date)."""
    local_noon = noon_utc + timedelta(hours=lon_deg / 15.0)
    return [(local_noon + timedelta(days=k)).date() for k in range(nights)]


@dataclass(frozen=True)
class Slot:
    rig: str
    block: int


def supply(rigs: Sequence[Rig], horizon: Horizon, p_clear: Sequence[float]) -> tuple[list[Slot], list[float]]:
    """One slot per rig × block; its capacity in expected hours."""
    slots, capacity = [], []
    for rig in rigs:
        for b in range(horizon.n):
            slots.append(Slot(rig.key, b))
            capacity.append(float(horizon.hours[b]) * float(p_clear[b]))
    return slots, capacity


def edges(slots: Sequence[Slot], horizon: Horizon, cands: Sequence[tuple[str, Candidate]],
          last_night: dict[int, int] | None = None) -> list[Edge]:
    """Which slot can serve which candidate, at what efficiency. `cands` pairs
    each candidate with the rig key that shoots it; `last_night` caps a
    candidate's edges at a night index (tests only)."""
    by_rig: dict[str, list[tuple[int, int]]] = {}
    for si, s in enumerate(slots):
        by_rig.setdefault(s.rig, []).append((si, s.block))
    tracks: dict[tuple[float, float], Track] = {}
    out: list[Edge] = []
    for ci, (rig_key, c) in enumerate(cands):
        key = (c.ra_deg, c.dec_deg)
        if key not in tracks:
            tracks[key] = horizon.track(*key)
        tr = tracks[key]
        eff = efficiency(c.filter, tr.alt, horizon.moon_alt, horizon.moon_illum, tr.moon_sep)
        stop = (last_night or {}).get(ci)
        for si, b in by_rig.get(rig_key, []):
            if eff[b] > 0 and (stop is None or horizon.night[b] <= stop):
                out.append(Edge(si, ci, float(eff[b])))
    return out
