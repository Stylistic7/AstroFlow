"""Tonight, sliced into equal time slots: sun, moon, and every target's track."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
from astropy.time import Time

SLOT_MIN = 10
# Plan past the end of astronomical darkness: the hour before dawn is often
# the stillest and clearest, and the ASIAir stops the plan at its own dawn
# setting anyway (Stylistic7, 2026-10-12).
DAWN_OVERRUN_MIN = 30


@dataclass(frozen=True)
class Night:
    location: EarthLocation
    times: Time               # slot centres across the astronomically dark night
    moon_alt: np.ndarray      # degrees, per slot
    moon_illum: np.ndarray    # 0..1, per slot
    moon: SkyCoord            # per slot, for separations

    @property
    def n(self) -> int:
        return len(self.times)

    def local(self, slot: int, utc_offset_h: float) -> str:
        t = self.times[min(slot, self.n - 1)].to_datetime(timezone=timezone.utc) + timedelta(hours=utc_offset_h)
        return t.strftime("%H:%M")


def tonight(lon_deg: float, lat_deg: float, local_noon_utc: datetime) -> Night:
    """Slots of SLOT_MIN minutes from the sun reaching −18° in the evening to
    DAWN_OVERRUN_MIN past −18° in the morning following `local_noon_utc`."""
    loc = EarthLocation(lon=lon_deg * u.deg, lat=lat_deg * u.deg)
    grid = Time(local_noon_utc) + np.arange(0, 24 * 60, SLOT_MIN) * u.min
    sun_alt = get_sun(grid).transform_to(AltAz(obstime=grid, location=loc)).alt.deg
    dark = np.where(sun_alt < -18)[0]
    end = min(len(grid), dark[-1] + 1 + DAWN_OVERRUN_MIN // SLOT_MIN)
    times = grid[dark[0]:end] + (SLOT_MIN / 2) * u.min
    frame = AltAz(obstime=times, location=loc)
    moon = get_body("moon", times, loc)
    sun = get_sun(times)
    elongation = sun.separation(moon).rad
    return Night(location=loc, times=times,
                 moon_alt=moon.transform_to(frame).alt.deg,
                 moon_illum=(1 - np.cos(elongation)) / 2,
                 moon=moon)


@dataclass(frozen=True)
class Track:
    alt: np.ndarray           # degrees, per slot
    hour_angle: np.ndarray    # hours, −12..12; negative = east of the meridian
    moon_sep: np.ndarray      # degrees, per slot


def track(night: Night, ra_deg: float, dec_deg: float) -> Track:
    c = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    alt = c.transform_to(AltAz(obstime=night.times, location=night.location)).alt.deg
    lst = night.times.sidereal_time("apparent", longitude=night.location.lon).hour
    ha = (lst - ra_deg / 15.0 + 12.0) % 24.0 - 12.0
    return Track(alt=alt, hour_angle=ha, moon_sep=night.moon.separation(c).deg)

