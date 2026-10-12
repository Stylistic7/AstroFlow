"""The night's slot grid: astronomical darkness plus the dawn overrun."""
from datetime import datetime, timezone

import astropy.units as u
from astropy.coordinates import AltAz, get_sun

from planner import sky

LON, LAT = -122.42, 37.77
NOON_UTC = datetime(2026, 10, 12, 19, tzinfo=timezone.utc)


def test_night_runs_from_dusk_to_thirty_minutes_past_dawn():
    night = sky.tonight(LON, LAT, NOON_UTC)
    sun = get_sun(night.times).transform_to(AltAz(obstime=night.times, location=night.location)).alt.deg
    assert sun[0] < -18
    overrun = sky.DAWN_OVERRUN_MIN // sky.SLOT_MIN
    assert (sun[-overrun:] > -18).all() and sun[-overrun - 1] < -18
    assert 9.5 * 60 <= night.n * sky.SLOT_MIN <= 11.5 * 60          # an October night in San Francisco
