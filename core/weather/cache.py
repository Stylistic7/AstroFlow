"""A file cache for fetched sky data, so an hourly planner run does not
re-download a forecast every time and never hits Open-Meteo for the same
decade of ERA5 twice. One JSON file per key under the app-support
directory; a stale or unreadable file is simply refetched."""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from weather.sources import SkyHour

FORECAST_TTL_S = 3600           # forecasts change hourly at most
CLIMATOLOGY_TTL_S = 30 * 86400  # a decade of reanalysis changes by a week a week


def _dir() -> Path:
    from base_path import app_support_dir
    d = app_support_dir() / "weather_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cached(key: str, ttl_s: float, make: Callable[[], Any],
           encode: Callable[[Any], Any] = lambda v: v,
           decode: Callable[[Any], Any] = lambda v: v) -> Any:
    path = _dir() / f"{key}.json"
    try:
        stored = json.loads(path.read_text())
        if time.time() - stored["at"] < ttl_s:
            return decode(stored["value"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    value = make()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"at": time.time(), "value": encode(value)}))
    tmp.replace(path)
    return value


def hours_to_json(hours: list[SkyHour]) -> list[dict]:
    return [{**h.__dict__, "time": h.time.isoformat()} for h in hours]


def hours_from_json(rows: list[dict]) -> list[SkyHour]:
    return [SkyHour(**{**r, "time": datetime.fromisoformat(r["time"])}) for r in rows]


def week_table_to_json(table: dict[int, float]) -> dict[str, float]:
    return {str(k): v for k, v in table.items()}


def week_table_from_json(obj: dict[str, float]) -> dict[int, float]:
    return {int(k): float(v) for k, v in obj.items()}
