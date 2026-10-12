"""The next night's plan, with everything needed to explain it.

`next_night(context, now)` plans the upcoming dark window — tonight's if it
has not ended yet, otherwise tomorrow's — from what is on disk right now.
It takes no date: a plan is only valid for the next night, because whatever
the weather does to it changes every later plan, and the planner re-runs
from the masters after each night anyway.

The result is plain JSON for `/api/planner/next` and the planner page:
slot times (UTC ISO), the moon, per-filter scarcity, and per rig its visits,
each with the reason it was chosen, its altitude track, the runners-up with
the one reason each lost, and a seam map for mosaic panels.
"""
from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta, timezone

import numpy as np

from planner import efficiency, inventory, schedule, sky
from planner.efficiency import DEFAULT_FLOOR_DEG, FLOOR_DEG, moon_factor


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


def _runners_up(night, plan, cands, tracks, rarity, block, k=4) -> list[dict]:
    t, out, seen = block.start_slot, [], set()
    chosen = (block.candidate.target_id, block.candidate.filter)
    scored = []
    for o in cands:
        if (o.target_id, o.filter) == chosen or o.need_h <= 0:
            continue
        tr = tracks[o.target_id]
        alt, ha = float(tr.alt[t]), float(tr.hour_angle[t])
        moon = float(moon_factor(o.filter, night.moon_alt[t:t + 1], night.moon_illum[t:t + 1], tr.moon_sep[t:t + 1])[0])
        side_ok = (ha < 0) if block.side == "east" else (ha >= 0)
        if alt < _floor(o.filter):
            reason = f"below its {_floor(o.filter):.0f}° floor ({alt:.0f}°)"
        elif moon == 0:
            reason = "moon up"
        elif not side_ok:
            reason = "other side of the meridian: would need a flip"
        else:
            reason = "smaller payoff per hour"
        scored.append((o.worth(1.0) * rarity.get(o.filter, 1.0) * o.urgency, o, reason))
    for score, o, reason in sorted(scored, key=lambda x: -x[0]):
        if o.label in seen:
            continue
        seen.add(o.label)
        out.append({"label": o.label, "filter": o.filter, "reason": reason, "score": round(score, 2), "why": o.why})
        if len(out) == k:
            break
    return out


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


def next_night(context, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    lon, lat = inventory.site(context)
    night, noon_utc = _upcoming(lon, lat, now)
    rigs = inventory.rigs(context)
    cands, unassignable = inventory.candidates(context, rigs)
    rarity = efficiency.scarcity(efficiency.availability(lon, lat, noon_utc))
    depths = inventory.master_depths(context.output_dir)
    out = {**night_header(now, lon, lat, night),
           "scarcity": {f: round(v, 2) for f, v in rarity.items()},
           "rigs": [], "mosaics": {}, "unassignable": unassignable}
    for rig in rigs:
        cs = cands.get(rig.key, [])
        inventory.set_urgency(cs, lon, lat, noon_utc)
        plan = schedule.plan_rig(night, rig, cs, rarity)
        tracks = {c.target_id: sky.track(night, c.ra_deg, c.dec_deg) for c in cs}
        visits = []
        for b in plan.blocks:
            c = b.candidate
            visits.append({**visit_json(b, tracks),
                           "urgency": round(c.urgency, 2), "scarcity": rarity.get(c.filter, 1.0),
                           "runners_up": _runners_up(night, plan, cs, tracks, rarity, b)})
            key = f"{c.mosaic_id}|{c.filter}"
            if c.mosaic_id and key not in out["mosaics"]:
                out["mosaics"][key] = _seam_map(context, c.mosaic_id, c.filter, depths)
        out["rigs"].append({"key": rig.key, "name": rig.name, "focal": rig.focal_mm, "scale": round(rig.image_scale, 2),
                            "flip": plan.flip_slot, "visits": visits, "score": round(plan.score, 2)})
    return out
