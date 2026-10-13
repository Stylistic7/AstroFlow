"""What every plan JSON carries to explain itself (`planner.v2.plan` builds
the plan): the upcoming dark window — tonight's if it has not ended yet,
otherwise tomorrow's — slot times (UTC ISO), the moon, each visit with the
reason it was chosen and its altitude track, and a seam map for mosaic
panels. A plan takes no date: it is only valid for the next night, because
whatever the weather does to it changes every later plan.
"""
from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta, timezone

import numpy as np

from planner import inventory, schedule, sky
from planner.efficiency import DEFAULT_FLOOR_DEG, FLOOR_DEG


def _floor(f: str) -> float:
    return FLOOR_DEG.get(f, DEFAULT_FLOOR_DEG)


def _upcoming(lon: float, lat: float, now: datetime) -> tuple[sky.Night, datetime]:
    """The first dark window that has not ended by `now`."""
    for days in (-1, 0, 1):
        local_noon = (now + timedelta(hours=lon / 15.0)).date() + timedelta(days=days)
        noon_utc = datetime.combine(local_noon, datetime.min.time(), timezone.utc) + timedelta(hours=12 - lon / 15.0)
        night = sky.tonight(lon, lat, noon_utc)
        if night.times[-1].to_datetime(timezone=timezone.utc) > now:
            return night, noon_utc
    raise RuntimeError("no dark window in the next two days")


def _seam_map(context, mosaic_id: str, filt: str, depths) -> dict:
    """Panels projected around the mosaic centre (degrees, east left), each
    with its noise over its neighbours' median in `filt`."""
    from spherical_geometry.vector import vector_to_lonlat
    m = context.mosaics[mosaic_id].get()
    nums = m.panel_numbers()
    targets = {t: context.targets[t].get() for t in m.panel_target_ids}
    sigma = {}
    for tid, t in targets.items():
        stacks = [context.stacks[s].get() for s in t.stack_ids if s in context.stacks]
        stacks = [s for s in stacks if s.filter == filt]
        if stacks:
            sigma[tid] = inventory._sigma_now(max(stacks, key=lambda s: s.integration), depths)
    cra, cdec = map(math.radians, m.center)
    panels = []
    for tid, t in targets.items():
        ref = [sigma[n] for n in targets if n != tid and sigma.get(n) and inventory._overlaps(t, targets[n])]
        ratio = sigma[tid] / statistics.median(ref) if sigma.get(tid) and ref else None
        pts = []
        for v in np.asarray(t.footprint.points)[:-1]:
            ra, dec = map(math.radians, vector_to_lonlat(*v))
            cosc = math.sin(cdec) * math.sin(dec) + math.cos(cdec) * math.cos(dec) * math.cos(ra - cra)
            x = math.cos(dec) * math.sin(ra - cra) / cosc
            y = (math.cos(cdec) * math.sin(dec) - math.sin(cdec) * math.cos(dec) * math.cos(ra - cra)) / cosc
            pts.append([round(-math.degrees(x), 3), round(math.degrees(y), 3)])
        panels.append({"n": nums.get(tid), "target": tid, "has": tid in sigma,
                       "ratio": None if ratio is None else round(ratio, 2), "pts": pts})
    return {"name": m.name, "short": mosaic_id[:4], "panels": panels}


def night_header(now: datetime, lon: float, lat: float, night: sky.Night) -> dict:
    """The keys every plan JSON starts with: when, where, the slot grid, the moon."""
    step = timedelta(minutes=sky.SLOT_MIN)
    start = night.times[0].to_datetime(timezone=timezone.utc) - step / 2
    return {"computed_at": now.isoformat(), "site": [lon, lat], "slot_min": sky.SLOT_MIN,
            "night_start": night.times[0].to_datetime(timezone=timezone.utc).replace(microsecond=0).isoformat(),
            "times": [(start + i * step).isoformat() for i in range(night.n + 1)],
            "moon_alt": [round(float(a), 1) for a in night.moon_alt],
            "moon_illum": round(float(np.mean(night.moon_illum)), 3)}


def visit_json(b: schedule.Block, tracks: dict) -> dict:
    """One visit as the page shows it; the caller adds what its planner knows."""
    c = b.candidate
    return {"start": b.start_slot, "end": b.end_slot, "label": c.label, "filter": c.filter,
            "mosaic": c.mosaic_id, "target": c.target_id, "side": b.side,
            "subs": max(1, int(((b.end_slot - b.start_slot) * sky.SLOT_MIN * 60 - 180) // (c.exposure_s + 15))),
            "exposure": c.exposure_s, "hours_so_far": round(c.hours, 1), "why": c.why,
            "value": round(b.value, 2), "floor": _floor(c.filter),
            "alt": [round(float(a), 1) for a in tracks[c.target_id].alt]}
