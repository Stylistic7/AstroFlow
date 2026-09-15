"""The download poller drains by day and at the dawn kick; night-time
arrivals wait on disk for the morning bake (2026-09-15)."""
from state import scheduler_actor as sa


def test_night_arrivals_wait_for_the_dawn_kick():
    assert not sa._drain_due(new_files=24, kick=False, sun_alt=-40.0)


def test_the_dawn_kick_drains_whatever_landed():
    assert sa._drain_due(new_files=0, kick=True, sun_alt=-17.0)
    assert sa._drain_due(new_files=14, kick=True, sun_alt=-17.0)


def test_daytime_arrivals_drain_as_they_land():
    assert sa._drain_due(new_files=3, kick=False, sun_alt=35.0)
    assert not sa._drain_due(new_files=0, kick=False, sun_alt=35.0)


def test_an_unknown_sun_altitude_drains_as_before():
    assert sa._drain_due(new_files=1, kick=False, sun_alt=None)


def test_the_kick_fires_once_as_the_sun_rises_through_twilight(monkeypatch):
    monkeypatch.setattr(sa, "_last_sun_alt", None)
    assert not sa._morning_kick_due(-12.0)  # booted after dawn: nothing to kick
    assert not sa._morning_kick_due(-30.0)  # dusk: the downward crossing
    assert not sa._morning_kick_due(-20.0)
    assert sa._morning_kick_due(-17.0)      # dawn
    assert not sa._morning_kick_due(10.0)   # once
    assert not sa._morning_kick_due(None)   # unknown sample: no kick, no reset
    assert sa._last_sun_alt == 10.0
