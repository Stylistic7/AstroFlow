"""The download poller drains by day and at the dawn kick; night-time
arrivals wait on disk for the morning bake (2026-09-15)."""
from state.scheduler_actor import _drain_due


def test_night_arrivals_wait_for_the_dawn_kick():
    assert not _drain_due(new_files=24, kick=False, sun_alt=-40.0)


def test_the_dawn_kick_drains_whatever_landed():
    assert _drain_due(new_files=0, kick=True, sun_alt=-17.0)
    assert _drain_due(new_files=14, kick=True, sun_alt=-17.0)


def test_daytime_arrivals_drain_as_they_land():
    assert _drain_due(new_files=3, kick=False, sun_alt=35.0)
    assert not _drain_due(new_files=0, kick=False, sun_alt=35.0)


def test_an_unknown_sun_altitude_drains_as_before():
    assert _drain_due(new_files=1, kick=False, sun_alt=None)
