"""p(clear) for a moment of sky time: the source ladder, calibrated.

The ladder is fixed — its horizons are hard-coded (Stylistic7, 2026-10-12; a UI
can come later): the local high-resolution model for the next 2 days, ECMWF
to day 7, the site's climatology beyond. The user's per-night override wins
over all of them.

A forecast's cloud cover is not a probability. On Stylistic7's 2026 nights an
HRRR "under 5% cloud" hour was clear only half the time: the marine layer,
plus nights not shot for other reasons. `CALIBRATION` maps each source's
cloud cover to the fraction of such hours that were actually clear,
measured in San Francisco over 1,681 dark hours (Mar–Oct 2026). Fitting it
per site from the user's own subs is a later step; it replaces this table
and nothing else.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta

from weather.sources import SkyHour

# (longest lead in days, Open-Meteo model): the first rung whose lead covers
# the block and whose forecast has a value there is used.
LADDER: tuple[tuple[float, str], ...] = ((2.0, "ncep_hrrr_conus"), (7.0, "ecmwf_ifs025"))

# Per source: (upper edge of the cloud-cover bin in %, P(clear) for that bin).
# ECMWF's curve is the mean over leads 2–7 (they differed by a few points).
CALIBRATION: dict[str, tuple[tuple[float, float], ...]] = {
    "ncep_hrrr_conus": ((5, 0.50), (10, 0.44), (20, 0.26), (30, 0.13), (50, 0.13), (70, 0.10), (101, 0.02)),
    "ecmwf_ifs025": ((5, 0.48), (10, 0.36), (20, 0.25), (30, 0.22), (50, 0.13), (70, 0.10), (101, 0.04)),
    "era5": ((5, 0.45), (10, 0.21), (20, 0.20), (30, 0.15), (50, 0.22), (70, 0.15), (101, 0.03)),
}

OVERRIDE_P = {"clear": 1.0, "cloudy": 0.0, "away": 0.0}


def calibrate(source: str, cloud: float) -> float:
    """The fraction of hours with this forecast cloud cover that were clear."""
    for edge, p in CALIBRATION[source]:
        if cloud < edge:
            return p
    return CALIBRATION[source][-1][1]


def _by_hour(hours: Sequence[SkyHour]) -> dict[datetime, float]:
    return {h.time.replace(minute=0, second=0, microsecond=0): h.cloud
            for h in hours if h.cloud is not None}


OVERRIDE = "override"
CLIMATOLOGY = "climatology"


def forecast(times: Sequence[datetime], now: datetime,
             forecasts: dict[str, Sequence[SkyHour]],
             climatology: Callable[[datetime], float],
             override: Callable[[datetime], str | None] = lambda t: None) -> list[tuple[float, str]]:
    """Per time (UTC): (p(clear), the rung that answered) by the ladder.

    `forecasts` maps a model id to its hours; `climatology(t)` is the site's
    clear fraction for that week of the year; `override(t)` returns the
    user's word for that night ("clear" / "cloudy" / "away") or None. The
    rung is the model id, OVERRIDE or CLIMATOLOGY: a planner may trust an
    expected hour from a forecast more than one from the climate."""
    tables = {model: _by_hour(hours) for model, hours in forecasts.items()}
    out = []
    for t in times:
        word = override(t)
        if word in OVERRIDE_P:
            out.append((OVERRIDE_P[word], OVERRIDE))
            continue
        lead_days = (t - now) / timedelta(days=1)
        key = t.replace(minute=0, second=0, microsecond=0)
        answer = None
        for max_lead, model in LADDER:
            if lead_days <= max_lead and key in tables.get(model, {}):
                answer = (calibrate(model, tables[model][key]), model)
                break
        out.append(answer or (climatology(t), CLIMATOLOGY))
    return out


def p_clear(times: Sequence[datetime], now: datetime,
            forecasts: dict[str, Sequence[SkyHour]],
            climatology: Callable[[datetime], float],
            override: Callable[[datetime], str | None] = lambda t: None) -> list[float]:
    """`forecast` without the rungs."""
    return [p for p, _ in forecast(times, now, forecasts, climatology, override)]
