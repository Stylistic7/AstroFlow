"""Sky forecasts from every source, as one shape: `SkyHour`.

Each source answers "what will the sky over (lat, lon) be at time t" in its
own units and cadence; this package turns every answer into the same record
so the planner can compare, grade and combine them without knowing where a
number came from. Only fetching and unit conversion live here — no caching,
no blending, no context reads (the caller passes the site).

Sources (all free, no key):

* `open_meteo` — the live forecast of one weather model. `MODELS` names the
  ones graded on Stylistic7's nights (2026-10-12): HRRR (3 km, hourly runs,
  ~18-48 h) caught the most good nights; ECMWF next; GFS and ICON weaker.
* `open_meteo_archive` — what a model forecast in the past, `lead_days`
  ahead (0 = the short-lead stitched archive, 1-7 = day-N-ahead runs), for
  grading a model against the nights actually shot.
* `era5` — the reanalysis: the best estimate of what the sky WAS, for
  climatology.
* `seven_timer` — 7Timer! ASTRO (GFS-based, 3-hourly): the only source with
  seeing and transparency.

`probability.p_clear` turns these into one number per moment through a
fixed source ladder and a measured calibration; `climatology` supplies the
fallback beyond the forecasts; `cache` keeps fetches off the hourly path.

Every fetch raises on an HTTP or format error (fail-fast: a planner fed a
silent empty forecast would treat the night as unknown, not as broken).
"""
from weather.probability import LADDER, p_clear
from weather.sources import (
    MODELS, SkyHour, era5, open_meteo, open_meteo_archive, seven_timer,
)

__all__ = ["LADDER", "MODELS", "SkyHour", "era5", "open_meteo", "open_meteo_archive",
           "p_clear", "seven_timer"]
