"""Fetch + normalise: each source's response becomes `list[SkyHour]`.

The fetchers are thin (one HTTP GET each); the `parse_*` functions are pure,
so the unit conversions are tested on canned responses with no network.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import requests

TIMEOUT_S = 60

# Open-Meteo model ids, best first on Stylistic7's 2026 nights (see __init__).
MODELS = ("ncep_hrrr_conus", "ecmwf_ifs025", "gfs_global", "icon_seamless")
_CLOUD = ("cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high")

_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
_ERA5_URL = "https://archive-api.open-meteo.com/v1/archive"
_SEVEN_TIMER_URL = "https://www.7timer.info/bin/api.pl"

# 7Timer! ASTRO class -> value, the midpoint of each documented class.
# cloudcover 1-9: 0-6 %, 6-19, 19-31, 31-44, 44-56, 56-69, 69-81, 81-94, 94-100.
_7T_CLOUD = {1: 3.0, 2: 12.5, 3: 25.0, 4: 37.5, 5: 50.0, 6: 62.5, 7: 75.0, 8: 87.5, 9: 97.0}
# seeing 1-8: <0.5", 0.5-0.75, 0.75-1, 1-1.25, 1.25-1.5, 1.5-2, 2-2.5, >2.5.
_7T_SEEING = {1: 0.4, 2: 0.625, 3: 0.875, 4: 1.125, 5: 1.375, 6: 1.75, 7: 2.25, 8: 2.75}
# transparency 1-8 (mag per airmass): <0.3, 0.3-0.4, ..., 0.8-1, >1.
_7T_TRANSPARENCY = {1: 0.25, 2: 0.35, 3: 0.45, 4: 0.55, 5: 0.65, 6: 0.75, 7: 0.9, 8: 1.1}


@dataclass(frozen=True)
class SkyHour:
    """One source's sky at one time. Cloud cover in percent (0-100); a
    field a source does not provide is None, never a guess."""
    time: datetime                       # UTC, the forecast's valid time
    source: str                          # e.g. "open-meteo:ncep_hrrr_conus", "7timer:astro"
    cloud: float | None
    cloud_low: float | None = None
    cloud_mid: float | None = None
    cloud_high: float | None = None
    seeing_arcsec: float | None = None
    transparency_mag: float | None = None  # extinction, magnitudes per airmass
    lead_days: int | None = None         # how far ahead it was forecast; None for a live forecast or a reanalysis


def _get(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=TIMEOUT_S)
    r.raise_for_status()
    body = r.json()
    if body.get("error"):
        raise RuntimeError(f"{url}: {body.get('reason')}")
    return body


def _utc(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc)


def parse_open_meteo(body: dict, source: str, suffix: str = "", lead_days: int | None = None) -> list[SkyHour]:
    """An Open-Meteo `hourly` block (requested with timezone=UTC) as SkyHours.
    `suffix` selects a variable family (`_previous_day1` for a day-ahead run)."""
    h = body["hourly"]
    total, low, mid, high = (h[f"{v}{suffix}"] for v in _CLOUD)
    return [SkyHour(_utc(t), source, total[i], low[i], mid[i], high[i], lead_days=lead_days)
            for i, t in enumerate(h["time"])]


def parse_seven_timer(body: dict) -> list[SkyHour]:
    """7Timer! ASTRO JSON: `init` (YYYYMMDDHH, UTC) plus 3-hourly `timepoint`
    offsets; class codes become their class midpoints, -9999 becomes None."""
    init = datetime.strptime(body["init"], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    return [SkyHour(init + timedelta(hours=p["timepoint"]), "7timer:astro",
                    _7T_CLOUD.get(p.get("cloudcover")),
                    seeing_arcsec=_7T_SEEING.get(p.get("seeing")),
                    transparency_mag=_7T_TRANSPARENCY.get(p.get("transparency")))
            for p in body["dataseries"]]


def open_meteo(lat: float, lon: float, model: str, days: int = 3) -> list[SkyHour]:
    """The model's current forecast, `days` ahead from today (UTC)."""
    body = _get(_FORECAST_URL, {"latitude": lat, "longitude": lon, "hourly": ",".join(_CLOUD),
                                "models": model, "forecast_days": days, "timezone": "UTC"})
    return parse_open_meteo(body, f"open-meteo:{model}")


def open_meteo_archive(lat: float, lon: float, model: str, start: date, end: date,
                       lead_days: int = 0) -> list[SkyHour]:
    """What `model` forecast for [start, end], `lead_days` ahead: 0 is the
    short-lead archive (each hour from the newest run before it, the best a
    model ever did), 1-7 the runs made that many days earlier."""
    if not 0 <= lead_days <= 7:
        raise ValueError(f"lead_days must be 0-7, got {lead_days}")
    common = {"latitude": lat, "longitude": lon, "models": model, "timezone": "UTC",
              "start_date": start.isoformat(), "end_date": end.isoformat()}
    if lead_days == 0:
        body = _get(_HISTORICAL_URL, {**common, "hourly": ",".join(_CLOUD)})
        return parse_open_meteo(body, f"open-meteo:{model}", lead_days=0)
    suffix = f"_previous_day{lead_days}"
    body = _get(_PREVIOUS_RUNS_URL, {**common, "hourly": ",".join(v + suffix for v in _CLOUD)})
    return parse_open_meteo(body, f"open-meteo:{model}", suffix, lead_days)


def era5(lat: float, lon: float, start: date, end: date) -> list[SkyHour]:
    """The ERA5 reanalysis (0.25°, ~5 days behind real time)."""
    body = _get(_ERA5_URL, {"latitude": lat, "longitude": lon, "hourly": ",".join(_CLOUD),
                            "start_date": start.isoformat(), "end_date": end.isoformat(),
                            "timezone": "UTC"})
    return parse_open_meteo(body, "era5")


def seven_timer(lat: float, lon: float) -> list[SkyHour]:
    """7Timer! ASTRO: ~3 days, 3-hourly, with seeing and transparency."""
    return parse_seven_timer(_get(_SEVEN_TIMER_URL, {"lat": round(lat, 3), "lon": round(lon, 3),
                                                     "product": "astro", "output": "json"}))
