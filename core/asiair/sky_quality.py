"""Estimate sky transparency from PHD2 guide history.

Inputs: the rolling buffer of normalized GuideStep records produced by
`asiair.guide_event_stream` — each record carries `snr` and `star_mass`,
the two signals that drop with cloud cover.

The chip header surfaces ONE level (good / thin / heavy / lost) plus a
short caption; the dropdown shows the actual ratios so the user can
sanity-check.

Strategy — baseline / current ratio:

  * Baseline = median(snr, star_mass) over the OLDER half of the buffer
    when the buffer is long enough (≥ 60 samples). Older half so a recent
    cloud event doesn't lift its own baseline.
  * Current = median(snr, star_mass) over the most recent 10 samples.
  * Levels:
      - StarMass == 0 for the latest sample → "lost" (red)
      - either ratio < 0.4 → "heavy" (red-orange)
      - either ratio < 0.7 → "thin" (yellow)
      - else → "good" (green)

The window picks (older-half vs. most-recent 10) are starting points —
once we see real cloud events in the wild we'll tune.
"""
from __future__ import annotations

import statistics
from typing import Any


_MIN_HISTORY_FOR_BASELINE = 60   # ~1 minute at 1 Hz
_CURRENT_WINDOW = 10             # most recent 10 samples


def _median_of(records: list[dict], field: str) -> float | None:
    vals = [r.get(field) for r in records]
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None
    return float(statistics.median(vals))


def estimate(history: list[dict]) -> dict[str, Any]:
    """Pure function: history -> dict with keys `level`, `caption`,
    `snr_ratio`, `star_mass_ratio`, `samples`.

    `level` is one of: "unknown" | "good" | "thin" | "heavy" | "lost".
    """
    n = len(history)
    if n == 0:
        return {
            "level": "unknown", "caption": "no guide data yet",
            "snr_ratio": None, "star_mass_ratio": None, "samples": 0,
        }

    # Star-lost trumps everything else.
    latest_mass = history[-1].get("star_mass")
    if latest_mass is not None and float(latest_mass) <= 1.0:
        return {
            "level": "lost", "caption": "guide star lost",
            "snr_ratio": 0.0, "star_mass_ratio": 0.0, "samples": n,
        }

    if n < _MIN_HISTORY_FOR_BASELINE:
        return {
            "level": "unknown",
            "caption": f"warming up ({n}/{_MIN_HISTORY_FOR_BASELINE})",
            "snr_ratio": None, "star_mass_ratio": None, "samples": n,
        }

    # Baseline from older half; current from last _CURRENT_WINDOW.
    older = history[: n // 2]
    current = history[-_CURRENT_WINDOW:]
    snr_baseline = _median_of(older, "snr")
    snr_current = _median_of(current, "snr")
    mass_baseline = _median_of(older, "star_mass")
    mass_current = _median_of(current, "star_mass")

    def _ratio(curr, base):
        if curr is None or base is None or base <= 0:
            return None
        return curr / base

    snr_ratio = _ratio(snr_current, snr_baseline)
    mass_ratio = _ratio(mass_current, mass_baseline)

    # The weaker of the two ratios drives the level — clouds dim BOTH
    # signals; using the minimum errs on the side of warning.
    weakest = min(
        r for r in (snr_ratio, mass_ratio) if r is not None
    ) if any(r is not None for r in (snr_ratio, mass_ratio)) else None

    if weakest is None:
        level, caption = "unknown", "insufficient signal"
    elif weakest < 0.4:
        level, caption = "heavy", "heavy clouds — guide star dim"
    elif weakest < 0.7:
        level, caption = "thin", "thin clouds — transparency dropping"
    else:
        level, caption = "good", "transparency normal"

    return {
        "level": level,
        "caption": caption,
        "snr_ratio": round(snr_ratio, 3) if snr_ratio is not None else None,
        "star_mass_ratio": round(mass_ratio, 3) if mass_ratio is not None else None,
        "samples": n,
    }
