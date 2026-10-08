"""What could be shot: every (panel, filter) of the cart's objects, with
the hours it still needs. The one impure layer — it reads the context
(cart, mosaics, targets, stacks, rigs) and the masters' sidecars.

Need comes in tiers, spent in order:
  1. equalisation (mosaic panels only, weight EQUALISE): the hours that
     bring this panel's master noise to within EQUAL_TOLERANCE of the MEDIAN
     of its overlapping neighbours', σ² ∝ 1/t; a panel missing a filter its
     neighbours have needs their median hours. The median, not the deepest
     neighbour: a seam is a panel's border with all of them, and one odd
     neighbour (the gain-0 2023 Cygnus Loop panel, 2026-10-08) must not set
     the target for everything around it.
  2. depth (everything): priced by VISIBLE GAIN when the master has been
     analysed (`metadata.master_analysis`: the fraction of the field newly
     revealed per added hour) — VISIBLE_HOURS hours at a weight proportional
     to that gain; until then, the hours up to GOAL_H at a flat weight.
     Mosaic panels' depth counts half a single target's (DEEPEN vs TARGET):
     seams first, depth second.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict

from metadata.master_analysis import visible_gain_per_hour
from planner.model import Candidate, Rig

EQUALISE, TARGET, DEEPEN = 2.0, 1.0, 0.5
VISIBLE_HOURS = 10.0          # how far ahead one night's visible-gain price is trusted
VISIBLE_GAIN_UNIT = 0.02      # 2% of the field per hour ≈ weight 1 (a strong deep target)
EQUAL_TOLERANCE = 1.1
GOAL_H = {"H": 20.0, "S": 20.0, "O": 20.0, "L": 20.0, "R": 4.0, "G": 4.0, "B": 4.0}
DEFAULT_EXPOSURE_S = {"R": 120.0, "G": 120.0, "B": 120.0}   # star colour; everything else 300 s
PIXEL_UM = 3.76               # ASI6200MM; both rigs today
SENSOR = (9576, 6388)


def rigs(context) -> list[Rig]:
    out = []
    for r in context.asiair_devices or []:
        focal = r.get("focal_len_mm")
        if focal:
            out.append(Rig(key=r["key"], name=r.get("name") or r["key"], focal_mm=float(focal),
                           image_scale=206.265 * PIXEL_UM / float(focal), sensor=SENSOR))
    return out


def site(context) -> tuple[float, float]:
    """(lon, lat) as the rigs report it."""
    for r in context.asiair_devices or []:
        loc = (r.get("live") or {}).get("location")
        if loc:
            return float(loc[0]), float(loc[1])
    raise RuntimeError("no rig reports a location")


def master_depths(output_dir) -> dict[str, tuple[float, float, dict | None]]:
    """`{stack stable_hash: (noise, master integration s, analysis section)}` from the newest
    WBPP master on disk per stack: STBLHASH from its header, noise and
    integration from the sidecar `compute_image_metadata` wrote
    (`noise_estimate` = NOISE00, `master_integration_s` = images combined x
    exposure). Not the header's PCL:TotalExposureTime, which is unreliable
    (the Lion OIII master read 5.6 h against 20.3 h, 2026-10-08). Read
    directly so it is right even while the catalog's pipeline snapshots are
    cold after a deploy."""
    from pathlib import Path
    from file_io.generic import read_header
    from file_io.sidecar import get_section
    from metadata.sidecar_io import read_image_metadata
    newest: dict[str, tuple[float, Path]] = {}
    for path in Path(output_dir, "stack_files").glob("*.xisf"):
        try:
            sid = str(read_header(path).get("STBLHASH") or "")
            mtime = path.stat().st_mtime
        except (OSError, ValueError):
            continue
        if sid and (sid not in newest or mtime > newest[sid][0]):
            newest[sid] = (mtime, path)
    out = {}
    for sid, (_, path) in newest.items():
        meta = read_image_metadata(path)
        if meta and meta.noise_estimate and meta.master_integration_s:
            out[sid] = (meta.noise_estimate, meta.master_integration_s, get_section(path, "analysis"))
    return out


def _sigma_now(stack, depths) -> float | None:
    """The stack's master noise projected to the library's integration: a
    master built before the latest subs is σ·sqrt(t_master/t_library)."""
    if stack.stable_hash not in depths:
        return None
    noise, t_master, _ = depths[stack.stable_hash]
    return noise * math.sqrt(t_master / stack.integration)


def _overlaps(a, b) -> bool:
    from spherical_geometry.polygon import MalformedPolygonError
    if not a.footprint.intersects_poly(b.footprint):
        return False
    try:
        inter = a.footprint.intersection(b.footprint)
        polys = getattr(inter, "polygons", [inter])
        return bool(polys) and max(p.area() for p in polys) / min(a.footprint.area(), b.footprint.area()) >= 0.03
    except MalformedPolygonError:
        return True


def candidates(context, rig_list: list[Rig]) -> tuple[dict[str, list[Candidate]], list[str]]:
    """`{rig key: candidates}` for the cart, and the cart objects no rig can shoot."""
    cart = list(context.shopping_cart or [])
    depths = master_depths(context.output_dir)
    groups: dict[str, tuple[str | None, list[str]]] = {}
    for kind, oid, _ in cart:
        if kind == "mosaic" and oid in context.mosaics:
            m = context.mosaics[oid].get()
            groups[oid] = (oid, list(m.panel_target_ids))
        elif kind == "target" and oid in context.targets:
            groups.setdefault(oid, (None, [oid]))
    per_rig: dict[str, list[Candidate]] = defaultdict(list)
    unassignable = []
    for oid, (mosaic_id, panel_ids) in groups.items():
        targets = {t: context.targets[t].get() for t in panel_ids if t in context.targets}
        if not targets:
            continue
        rig = next((r for r in rig_list
                    if all(t.sensor == r.sensor and abs(t.image_scale / r.image_scale - 1) < 0.1 for t in targets.values())), None)
        if rig is None:
            unassignable.append(f"{oid[:8]} {next(iter(targets.values())).name}")
            continue
        numbers = context.mosaics[mosaic_id].get().panel_numbers() if mosaic_id else {}
        depth: dict[tuple[str, str], tuple[float, float | None]] = {}
        exposure: dict[tuple[str, str], float] = {}
        analysis: dict[tuple[str, str], dict | None] = {}
        for tid, t in targets.items():
            by_filter = defaultdict(list)
            for sid in t.stack_ids:
                if sid in context.stacks:
                    s = context.stacks[sid].get()
                    by_filter[s.filter].append(s)
            for f, stacks in by_filter.items():
                hours = sum(s.integration for s in stacks) / 3600.0
                deepest = max(stacks, key=lambda s: s.integration)
                depth[(tid, f)] = (hours, _sigma_now(deepest, depths))
                analysis[(tid, f)] = (depths.get(deepest.stable_hash) or (None, None, None))[2]
                exposure[(tid, f)] = float(deepest.exposure)
        filters = sorted({f for _, f in depth if f in GOAL_H})
        neighbours = {a: [b for b in targets if b != a and _overlaps(targets[a], targets[b])] for a in targets} if mosaic_id else {}
        for tid, t in targets.items():
            label = f"{mosaic_id[:4]} Panel {numbers.get(tid)}" if mosaic_id else t.name
            for f in filters:
                hours, sigma = depth.get((tid, f), (0.0, None))
                tiers, why = [], ""
                if mosaic_id:
                    nbr = [depth[(n, f)] for n in neighbours[tid] if (n, f) in depth]
                    ref = [s for _, s in nbr if s]
                    if hours == 0 and nbr:
                        need = statistics.median(h for h, _ in nbr)
                        why = f"no {f} yet; neighbours median {need:.1f} h"
                    elif sigma and ref and sigma > EQUAL_TOLERANCE * statistics.median(ref):
                        typical = statistics.median(ref)
                        need = hours * ((sigma / (EQUAL_TOLERANCE * typical)) ** 2 - 1)
                        why = f"σ {sigma / typical:.2f}x its neighbours' median"
                    else:
                        need = 0.0
                    if need > 0:
                        tiers.append([need, EQUALISE])
                scale = DEEPEN if mosaic_id else TARGET
                gain = visible_gain_per_hour(analysis.get((tid, f)), hours) if hours > 0 else None
                if gain is not None:
                    tiers.append([VISIBLE_HOURS, scale * min(3.0, max(0.0, gain / VISIBLE_GAIN_UNIT))])
                    why = why or f"+{100 * gain:.2f}% of field visible per hour"
                else:
                    deepen = max(0.0, GOAL_H[f] - hours - sum(h for h, _ in tiers))
                    if deepen > 0:
                        tiers.append([deepen, scale])
                tiers = [t for t in tiers if t[1] > 0]
                if not tiers:
                    continue
                per_rig[rig.key].append(Candidate(
                    target_id=tid, label=label, mosaic_id=mosaic_id, filter=f,
                    ra_deg=float(t.center[0]), dec_deg=float(t.center[1]), hours=hours,
                    tiers=tiers, why=why or f"{hours:.1f} h of {GOAL_H[f]:.0f} h goal",
                    exposure_s=exposure.get((tid, f), DEFAULT_EXPOSURE_S.get(f, 300.0))))
    return dict(per_rig), unassignable


def set_urgency(cands: list[Candidate], lon: float, lat: float, start_utc, cap: float = 3.0) -> None:
    """Scale each candidate by how little sky its target has left this
    season relative to the others on the same rig: median remaining hours /
    its remaining hours, between 1 and `cap`. Floors as in `efficiency`."""
    from planner.efficiency import DEFAULT_FLOOR_DEG, FLOOR_DEG
    from planner.sky import season_hours
    remaining = {}
    for c in cands:
        key = (c.target_id, FLOOR_DEG.get(c.filter, DEFAULT_FLOOR_DEG))
        if key not in remaining:
            remaining[key] = season_hours(lon, lat, start_utc, c.ra_deg, c.dec_deg, key[1])
    if not remaining:
        return
    median = statistics.median(remaining.values())
    for c in cands:
        left = remaining[(c.target_id, FLOOR_DEG.get(c.filter, DEFAULT_FLOOR_DEG))]
        c.urgency = min(cap, max(1.0, median / max(left, 1.0)))
