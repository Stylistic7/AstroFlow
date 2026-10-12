"""The ladder, the calibration and the override, on canned forecasts."""
from datetime import datetime, timedelta, timezone

from weather.climatology import by_week, lookup
from weather.probability import CLIMATOLOGY, OVERRIDE, calibrate, forecast, p_clear
from weather.sources import SkyHour

NOW = datetime(2026, 10, 12, 20, tzinfo=timezone.utc)


def _hours(model, start, n, cloud):
    return [SkyHour(start + timedelta(hours=i), f"open-meteo:{model}", cloud) for i in range(n)]


def test_calibration_maps_cloud_bins_to_measured_clear_fractions():
    assert calibrate("ncep_hrrr_conus", 0) == 0.50
    assert calibrate("ncep_hrrr_conus", 15) == 0.26
    assert calibrate("ncep_hrrr_conus", 100) == 0.02


def test_ladder_hrrr_first_then_ecmwf_then_climatology():
    t0 = NOW + timedelta(hours=4)                      # tonight: HRRR
    t3 = NOW + timedelta(days=3, hours=4)              # day 3: ECMWF
    t20 = NOW + timedelta(days=20, hours=4)            # day 20: climatology
    forecasts = {"ncep_hrrr_conus": _hours("ncep_hrrr_conus", NOW, 48, 2),
                 "ecmwf_ifs025": _hours("ecmwf_ifs025", NOW, 24 * 8, 25)}
    p = p_clear([t0, t3, t20], NOW, forecasts, climatology=lambda t: 0.3)
    assert p == [0.50, 0.22, 0.3]
    rungs = [r for _, r in forecast([t0, t3, t20], NOW, forecasts, climatology=lambda t: 0.3)]
    assert rungs == ["ncep_hrrr_conus", "ecmwf_ifs025", CLIMATOLOGY]


def test_missing_rung_falls_to_the_next():
    t1 = NOW + timedelta(days=1, hours=4)
    forecasts = {"ecmwf_ifs025": _hours("ecmwf_ifs025", NOW, 72, 2)}   # no HRRR at all
    assert p_clear([t1], NOW, forecasts, climatology=lambda t: 0.3) == [0.48]


def test_override_wins_over_every_source():
    t0 = NOW + timedelta(hours=4)
    forecasts = {"ncep_hrrr_conus": _hours("ncep_hrrr_conus", NOW, 48, 100)}
    assert p_clear([t0], NOW, forecasts, lambda t: 0.3, override=lambda t: "clear") == [1.0]
    assert p_clear([t0], NOW, forecasts, lambda t: 0.3, override=lambda t: "away") == [0.0]
    assert forecast([t0], NOW, forecasts, lambda t: 0.3, override=lambda t: "clear") == [(1.0, OVERRIDE)]


def test_climatology_by_week_averages_calibrated_dark_hours_only():
    base = datetime(2025, 10, 6, 0, tzinfo=timezone.utc)                 # ISO week 41
    hours = [SkyHour(base + timedelta(hours=i), "era5", 2 if i % 2 else 100) for i in range(24)]
    dark = [i % 2 == 1 for i in range(24)]                               # only the clear ones are dark
    table = by_week(hours, dark)
    assert table == {41: 0.45}
    f = lookup(table)
    assert f(base) == 0.45 and f(base + timedelta(days=7)) == 0.45      # neighbour week borrows
    assert lookup({})(base) == 0.3
