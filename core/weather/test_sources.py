"""Normalisation of each source's response, on canned bodies (no network)."""
from datetime import datetime, timezone

import pytest

from weather.sources import SkyHour, parse_open_meteo, parse_seven_timer, open_meteo_archive


def _om(suffix=""):
    return {"hourly": {
        "time": ["2026-10-12T04:00", "2026-10-12T05:00"],
        f"cloud_cover{suffix}": [6, 0], f"cloud_cover_low{suffix}": [6, None],
        f"cloud_cover_mid{suffix}": [0, 0], f"cloud_cover_high{suffix}": [1, 0]}}


def test_open_meteo_hours_are_utc_percent_with_missing_kept_none():
    hours = parse_open_meteo(_om(), "open-meteo:ncep_hrrr_conus")
    assert hours[0] == SkyHour(datetime(2026, 10, 12, 4, tzinfo=timezone.utc),
                               "open-meteo:ncep_hrrr_conus", 6, 6, 0, 1)
    assert hours[1].cloud_low is None and hours[1].lead_days is None


def test_open_meteo_day_ahead_reads_the_suffixed_family():
    hours = parse_open_meteo(_om("_previous_day1"), "open-meteo:ecmwf_ifs025", "_previous_day1", 1)
    assert [h.cloud for h in hours] == [6, 0] and hours[0].lead_days == 1


def test_seven_timer_classes_become_midpoints_on_3h_steps():
    body = {"product": "astro", "init": "2026101212", "dataseries": [
        {"timepoint": 3, "cloudcover": 1, "seeing": 4, "transparency": 3},
        {"timepoint": 6, "cloudcover": 9, "seeing": -9999, "transparency": 8}]}
    first, second = parse_seven_timer(body)
    assert first.time == datetime(2026, 10, 12, 15, tzinfo=timezone.utc)
    assert (first.cloud, first.seeing_arcsec, first.transparency_mag) == (3.0, 1.125, 0.45)
    assert second.time == datetime(2026, 10, 12, 18, tzinfo=timezone.utc)
    assert (second.cloud, second.seeing_arcsec, second.transparency_mag) == (97.0, None, 1.1)
    assert first.cloud_low is None and first.source == "7timer:astro"


def test_archive_refuses_a_lead_it_cannot_serve():
    with pytest.raises(ValueError):
        open_meteo_archive(37.77, -122.42, "ecmwf_ifs025", None, None, lead_days=9)
