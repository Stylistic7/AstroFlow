"""Planner v2: the month decides tonight.

Layer 1 (`allocate`) spreads every need's hours over the coming season of
weather-weighted sky (`horizon`: a month in fine blocks, then coarse ones); Layer 2 (`schedule`, v1's greedy)
orders tonight's share into visits. Only tonight's quotas cross between
them. The month's plan is kept for the explanation and thrown away: it is
re-solved tomorrow from whatever was actually shot.

Tonight's quota is a PRIORITY, not a cap: Layer 2 sees every candidate with
its full need, and the hours Layer 1 reserved for tonight count
`QUOTA_BOOST` times their value. The greedy therefore places the month's
choices first and fills the rest of the night with the best of the rest —
an LP spreads hours thinly, and a night with gaps is no plan. When Layer 1
reserved nothing for a rig (the forecast says cloudy, or no weather at all)
the night is planned on v1's values alone and says so (`fallback`): v2
degrades to v1, never to nothing.

Shadow mode (design doc, 2026-10-12): `write_shadow` puts this plan beside
v1's as planner_next_v2.json, served by /api/planner/next_v2. Nothing is
pushed to a rig.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from planner import allocate, explain, horizon, inventory, schedule, sky
from planner.horizon import BLOCK_MIN, COARSE_BLOCK_MIN, FINE_NIGHTS, SEASON_NIGHTS
from planner.model import Candidate, Rig
from weather import cache, climatology, probability, sources

MIN_QUOTA_H = 0.5          # a smaller share of tonight is a fragment: no priority, no slew
QUOTA_BOOST = 3.0          # tonight's reserved hours outrank everything v1 values alone
# Time preference: an hour's value falls 3% per night into the future. A
# finished image sooner is worth more, and without this an LP with a season
# of spare sky is indifferent to WHEN — it spreads every need thinly and
# tonight gets fragments. With it, tonight fills with the best of what is
# usable tonight, then tomorrow, and so on: "shoot the best thing now".
# 1% (2026-10-12) was too mild: a sure moonless hour tonight lost to an
# expected one in November.
NIGHT_DISCOUNT = 0.97
# How much of an expected hour the plan may count on, by the rung that
# forecast it. The climate promises ~25% clear nights for six months, which
# made moonless OIII hours look abundant and tonight's unspecial; counting
# the climate at half makes them scarce, as Stylistic7 knows them to be. The
# probability itself is untouched (the month view shows it); only the sky
# the LP is allowed to sell.
CAPACITY_TRUST = {probability.CLIMATOLOGY: 0.5}


def quota_tiers(tiers: list[list[float]], quota_h: float, boost: float = QUOTA_BOOST) -> list[list[float]]:
    """`tiers` with the first `quota_h` hours at `boost` times their value and
    the remainder unchanged: the month's reservation first, then the rest."""
    out = []
    for hours, value in tiers:
        take = min(hours, max(quota_h, 0.0))
        if take > 1e-9:
            out.append([take, value * boost])
        if hours - take > 1e-9:
            out.append([hours - take, value])
        quota_h -= take
    return out


def night_key(t: datetime, lon_deg: float) -> date:
    """The evening's local date a moment belongs to (noon to noon)."""
    return (t + timedelta(hours=lon_deg / 15.0 - 12.0)).date()


def _forecasts(lat: float, lon: float) -> tuple[dict[str, list[sources.SkyHour]], dict[str, str]]:
    out, status = {}, {}
    for max_lead, model in probability.LADDER:
        try:
            out[model] = cache.cached(
                f"forecast_{model}_{lat:.2f}_{lon:.2f}", cache.FORECAST_TTL_S,
                lambda m=model, d=int(max_lead) + 2: sources.open_meteo(lat, lon, m, days=d),
                encode=cache.hours_to_json, decode=cache.hours_from_json)
            status[model] = f"{len(out[model])} hours"
        except Exception as e:                       # one source down: the next rung serves
            status[model] = f"unavailable: {type(e).__name__}: {e}"[:120]
    return out, status


def _climatology(lat: float, lon: float, status: dict[str, str]):
    try:
        table = cache.cached(f"climatology_era5_{lat:.2f}_{lon:.2f}", cache.CLIMATOLOGY_TTL_S,
                             lambda: climatology.fetch(lat, lon),
                             encode=cache.week_table_to_json, decode=cache.week_table_from_json)
        status["era5"] = f"{len(table)} weeks"
    except Exception as e:
        table = {}
        status["era5"] = f"unavailable: {type(e).__name__}: {e}"[:120]
    return climatology.lookup(table)


def plan(context, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    lon, lat = inventory.site(context)
    night, noon_utc = explain._upcoming(lon, lat, now)
    rigs = inventory.rigs(context)
    cands_by_rig, unassignable = inventory.candidates(context, rigs)
    depths = inventory.master_depths(context.output_dir)

    # --- the month: supply, weather, demand, edges, one solve -------------
    hz = horizon.dark_blocks(lon, lat, noon_utc)
    dates = horizon.night_dates(noon_utc, lon, SEASON_NIGHTS)
    forecasts, weather_status = _forecasts(lat, lon)
    clim = _climatology(lat, lon, weather_status)
    overrides = {str(k): v for k, v in (context.get("planner_overrides") or {}).items()}
    answers = probability.forecast(hz.utc(), now, forecasts, clim,
                                   override=lambda t: overrides.get(night_key(t, lon).isoformat()))
    p = [pr for pr, _ in answers]
    trusted = [pr * CAPACITY_TRUST.get(rung, 1.0) for pr, rung in answers]
    slots, capacity = horizon.supply(rigs, hz, trusted)
    pairs: list[tuple[str, Candidate]] = [(rig.key, c) for rig in rigs for c in cands_by_rig.get(rig.key, [])]
    needs = [allocate.Need(f"{rig_key}|{c.target_id}|{c.filter}", tuple((h, w) for h, w in c.tiers))
             for rig_key, c in pairs]
    es = [replace(e, efficiency=e.efficiency * NIGHT_DISCOUNT ** int(hz.night[slots[e.slot].block]))
          for e in horizon.edges(slots, hz, pairs)]
    alloc = allocate.solve(capacity, needs, es)

    # --- tonight's quotas: expected hours back to real hours ---------------
    slot_of = {(s.rig, s.block): i for i, s in enumerate(slots)}
    tonight = [b for b in range(hz.n) if hz.night[b] == 0]
    # Layer 1 decides how much of each FILTER tonight — the moon-dependent
    # part; which need gets it is value's call. An LP is indifferent among
    # equal needs and splits a filter's hours into fragments under the
    # minimum, which then vanish; consolidating them by value keeps the
    # filter mix and loses nothing.
    by_filter: dict[tuple[str, str], float] = {}
    for i, (rig_key, c) in enumerate(pairs):
        real = sum(alloc.hours.get((slot_of[(rig_key, b)], i), 0.0) / trusted[b] for b in tonight if trusted[b] > 0)
        by_filter[(rig_key, c.filter)] = by_filter.get((rig_key, c.filter), 0.0) + real
    quota = {}
    for (rig_key, f), hours in by_filter.items():
        best_first = sorted((i for i, (rk, c) in enumerate(pairs) if rk == rig_key and c.filter == f),
                            key=lambda i: -max(w for _, w in pairs[i][1].tiers))
        for i in best_first:
            take = min(hours, pairs[i][1].need_h)
            if take < MIN_QUOTA_H:
                break
            quota[i] = take
            hours -= take

    # --- what each need has left in the month, for the explanation --------
    future = {}
    for e in es:
        b = slots[e.slot].block
        if hz.night[b] > 0:
            f = future.setdefault(e.need, {"last_night": 0, "nights": set(), "expected_h": 0.0})
            f["last_night"] = max(f["last_night"], int(hz.night[b]))
            f["nights"].add(int(hz.night[b]))
            f["expected_h"] += capacity[e.slot] * e.efficiency

    out = {**explain.night_header(now, lon, lat, night), "version": 2,
           "weather": {"sources": weather_status,
                       "tonight_p_clear": round(float(np.mean([p[b] for b in tonight])), 2) if tonight else None},
           "horizon": {"block_min": BLOCK_MIN, "coarse_block_min": COARSE_BLOCK_MIN, "fine_nights": FINE_NIGHTS,
                       "nights": _month(hz, dates, p, trusted, rigs, slots, alloc, pairs)},
           "layer1": {"objective": round(alloc.objective, 2), "needs": len(needs), "edges": len(es),
                      "slots": len(slots)},
           "rigs": [], "mosaics": {}, "unassignable": unassignable}

    # --- tonight: Layer 2 on the quotas ----------------------------------
    for rig in rigs:
        cs = cands_by_rig.get(rig.key, [])
        mine = [(i, c) for i, (rk, c) in enumerate(pairs) if rk == rig.key]
        quoted = [replace(c, tiers=quota_tiers(c.tiers, quota.get(i, 0.0)), urgency=1.0) for i, c in mine]
        fallback = None
        if not any(i in quota for i, _ in mine):
            fallback = ("Layer 1 reserved nothing for this rig tonight "
                        f"(p(clear) {out['weather']['tonight_p_clear']}); planned on v1's values alone")
        rig_plan = schedule.plan_rig(night, rig, quoted, rarity={})
        rig_plan.blocks = _merge_contiguous(rig_plan.blocks)
        tracks = {c.target_id: sky.track(night, c.ra_deg, c.dec_deg) for c in quoted}
        by_need = {(c.target_id, c.filter): i for i, c in mine}
        visits = []
        for b in rig_plan.blocks:
            c = b.candidate
            i = by_need.get((c.target_id, c.filter))
            visits.append({**explain.visit_json(b, tracks), "kind": c.kind,
                           "quota_h": round(quota.get(i, 0.0), 2) if i is not None else None,
                           "why_tonight": _why_tonight(c, i, future.get(i), dates, alloc, needs, p, tonight)})
            key = f"{c.mosaic_id}|{c.filter}"
            if c.mosaic_id and key not in out["mosaics"]:
                out["mosaics"][key] = explain._seam_map(context, c.mosaic_id, c.filter, depths)
        out["rigs"].append({
            "key": rig.key, "name": rig.name, "focal": rig.focal_mm, "scale": round(rig.image_scale, 2),
            "flip": rig_plan.flip_slot, "score": round(rig_plan.score, 2), "fallback": fallback,
            "quotas": [{"label": c.label, "filter": c.filter, "target": c.target_id, "hours": round(quota[i], 2)}
                       for i, c in mine if i in quota],
            "unplaced_quota_h": round(sum(quota[i] for i, c in mine if i in quota
                                          and (c.target_id, c.filter) not in {(b.candidate.target_id, b.candidate.filter) for b in rig_plan.blocks}), 2),
            "visits": visits})
    return out


def _merge_contiguous(blocks: list[schedule.Block]) -> list[schedule.Block]:
    """Back-to-back visits of one (target, filter) on one pier side, as one:
    the greedy closes a tier and starts the next, which is one visit to the
    rig."""
    out: list[schedule.Block] = []
    for b in blocks:
        last = out[-1] if out else None
        if last and last.end_slot == b.start_slot and last.side == b.side and \
                (last.candidate.target_id, last.candidate.filter) == (b.candidate.target_id, b.candidate.filter):
            out[-1] = replace(last, end_slot=b.end_slot, effective_h=last.effective_h + b.effective_h,
                              value=last.value + b.value)
        else:
            out.append(b)
    return out


def _why_tonight(c: Candidate, i: int | None, fut: dict | None, dates: list[date],
                 alloc: allocate.Allocation, needs, p, tonight) -> str:
    """One line from Layer 1: why this hour is spent now rather than later."""
    p_tonight = float(np.mean([p[b] for b in tonight])) if tonight else 0.0
    if i is None:
        return f"not in the month's allocation; tonight p(clear) {p_tonight:.0%}"
    if not fut:
        return f"tonight is its last usable night this season; p(clear) {p_tonight:.0%}"
    price = max((v for (n, _), v in alloc.need_price.items() if n == i), default=0.0)
    window = ("usable all season" if fut["last_night"] >= len(dates) - 1
              else f"usable until {dates[fut['last_night']]:%b %d}")
    room = ("the month has room for all of it" if price < 1e-6
            else f"wants {price:.2f}/h more than the month can give")
    return (f"{window} ({len(fut['nights'])} more nights, {fut['expected_h']:.1f} expected clear hours "
            f"above its floor); {room}; tonight p(clear) {p_tonight:.0%}")


def _month(hz, dates, p, trusted, rigs, slots, alloc, pairs) -> list[dict]:
    """Per night: the chance of clear sky, the dark hours, and the expected
    hours Layer 1 placed on each rig by filter."""
    rows = []
    slot_of = {(s.rig, s.block): i for i, s in enumerate(slots)}
    for k, day in enumerate(dates):
        blocks = [b for b in range(hz.n) if hz.night[b] == k]
        per_rig = {}
        for rig in rigs:
            hours: dict[str, float] = {}
            for b in blocks:
                for i, h in alloc.by_slot(slot_of[(rig.key, b)]).items():
                    f = pairs[i][1].filter
                    hours[f] = hours.get(f, 0.0) + h
            per_rig[rig.key] = {f: round(h, 2) for f, h in sorted(hours.items())}
        rows.append({"date": day.isoformat(),
                     "p_clear": round(float(np.mean([p[b] for b in blocks])), 2) if blocks else None,
                     "trusted": round(float(np.mean([trusted[b] for b in blocks])), 2) if blocks else None,
                     "dark_h": round(float(np.sum(hz.hours[blocks])), 1) if blocks else 0.0,
                     "moon_up_h": round(float(np.sum(hz.hours[blocks] * (hz.moon_alt[blocks] > 0))), 1) if blocks else 0.0,
                     "rigs": per_rig})
    return rows


def text(plan_json: dict) -> str:
    """A terminal rendering of the v2 plan."""
    t = lambda i: datetime.fromisoformat(plan_json["times"][i]).astimezone().strftime("%H:%M")
    lines = [f"v2 · tonight p(clear) {plan_json['weather']['tonight_p_clear']} · "
             f"sources {plan_json['weather']['sources']}"]
    for rig in plan_json["rigs"]:
        flip = t(rig["flip"]) if rig["flip"] is not None else "none"
        lines.append(f"{rig['name']} · {rig['focal']:.0f} mm   flip: {flip}" + (f"   [{rig['fallback']}]" if rig["fallback"] else ""))
        for v in rig["visits"]:
            lines.append(f"  {t(v['start'])}–{t(v['end'])}  {v['label']:<24} {v['filter']}  {v['subs']:3d} × {v['exposure']:.0f} s"
                         f"   quota {v['quota_h']} h   [{v['why']}]  [{v['why_tonight']}]")
    lines.append("the next 35 nights (expected hours by filter):")
    for n in plan_json["horizon"]["nights"][:35]:
        rigs = "  ".join(f"{k[:4]}:{v}" for k, v in n["rigs"].items() if v)
        lines.append(f"  {n['date']}  p {n['p_clear']} (counted {n['trusted']})  dark {n['dark_h']} h  moon-up {n['moon_up_h']} h  {rigs}")
    return "\n".join(lines)


def write_shadow(context, path: Path, inputs=None) -> None:
    """The v2 plan beside v1's; an error becomes the file, never an exception
    (v1's plan must land whatever v2 does)."""
    try:
        out = plan(context)
    except Exception as e:
        out = {"error": f"{type(e).__name__}: {e}", "computed_at": datetime.now(timezone.utc).isoformat()}
    if inputs is not None:
        out["inputs"] = inputs
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(out))
    tmp.replace(path)
