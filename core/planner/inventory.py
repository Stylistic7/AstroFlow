"""What could be shot: every (panel, filter) of the cart's objects, with
the hours it still needs. The one impure layer — it reads the context
(cart, mosaics, targets, stacks, rigs) and the masters' sidecars.

Need comes in tiers, each a run of hours at a value, spent in order. The
value of an hour is the IMPROVEMENT it buys, in one unit for everything
(weight 1 = VISIBLE_GAIN_UNIT of the field newly visible per hour):

  * depth: read off the master's measured quality curves
    (`metadata.master_analysis`: the fraction of the field at or above SNR
    3, 10, 30 and 100 after 1x..4x the current integration). An hour is
    worth the fraction of the OBJECT that climbs a level, summed over the
    levels: the field fractions climbing, divided by the fraction of the
    field detectable at 4x (what is there to improve; a small nebula on a
    long focal length is not penalised for the empty sky around it). A
    panel is not done at SNR 3 (detected); a portfolio image carries its
    nebulosity at 30 and up. Concave by construction, so the
    first hour is worth more than the tenth, and a panel drops out when its
    next tier is worth less than MIN_GAIN — "no more hours would improve it".
    Until a master is analysed, a flat FALLBACK_H goal at weight 1. On a
    NARROWBAND project (`project_kind`: more narrowband hours than broadband
    across the target or mosaic), R/G/B are star colour: their tiers stop at
    STAR_COLOUR_H (20 min a channel) however much the curve says more stars
    would show. An LRGB project keeps the curve for every filter.
  * seam (mosaic panels): the hours that bring this panel's noise within
    EQUAL_TOLERANCE of the MEDIAN of its overlapping neighbours' (σ² ∝ 1/t),
    valued by the size of the step (SEAM_WEIGHT per unit of excess noise).
    The median, not the deepest neighbour: one odd neighbour must not set
    the target for everything around it. A panel missing a filter its
    neighbours have needs their median hours at a full step's value (at
    weight 1 when the missing filter is star colour).

Noise is the master's RAW NOISE00. Scaling it by the flux calibration
(post_stack_calibrate's reference / scale) was tried on 2026-10-12 and
rejected: the scale varies 2-6x between stacks of one filter, roughly with
the sub count, and doubled the depth differences (a 0.2 h and a 0.7 h B
stack read 4x apart calibrated, 2x raw, sqrt(3.5) = 1.9x expected).
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict

from planner.model import Candidate, Rig

VISIBLE_GAIN_UNIT = 0.02      # 2% of the object climbing a quality level per hour ≈ weight 1
MIN_OBJECT = 0.05             # an object is at least 5% of the field, so an empty field cannot blow up
MIN_GAIN = 0.001              # below 0.1% of the field per hour, no more hours would improve it
# A curve that says more than this of the field is ALREADY visible is not
# trusted (the flat fallback applies). Measured 2026-10-12 on 118 stacks
# analysed from both a WBPP and a Siril master: 17% disagreed by over 0.2 of
# the field, every one a Siril FITS master reading 0.66-0.85 visible at 3-4x
# its twin's noise — the sky pedestal or gradient counted as structure. The
# root cause is metadata.visibility's background model; until it is fixed
# the guard keeps a bad curve from telling the planner a panel is done.
MAX_PLAUSIBLE_VISIBLE = 0.6
MAX_WEIGHT = 3.0              # a freak curve cannot outrank everything
EQUAL_TOLERANCE = 1.1         # a seam: noise more than 10% above the neighbours' median
SEAM_WEIGHT = 4.0             # per unit of excess noise: a 1.5x step ≈ weight 2
FALLBACK_H = {"H": 20.0, "S": 20.0, "O": 20.0, "L": 20.0, "R": 4.0, "G": 4.0, "B": 4.0}
STAR_COLOUR_H = 20.0 / 60.0   # star colour on a narrowband project: 20 min per channel is plenty (Stylistic7, 2026-10-12)
# Raw masters, in order of preference: one family per stack, so neighbours
# are compared like with like (Siril and WBPP masters of one stack differ by
# 10-25% in NOISE00).
MASTER_GLOBS = ("integrate_with_siril/*.fits", "stack_files/*.xisf")
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
    """`{stack stable_hash: (noise, master integration s, analysis)}` from the
    newest master on disk per stack, from the first family in MASTER_GLOBS
    that has one (STBLHASH from its header, noise and integration from the
    sidecar `compute_image_metadata` wrote, the analysis section
    `analyze_master` added). Not the header's PCL:TotalExposureTime, which is
    unreliable (the Lion OIII master read 5.6 h against 20.3 h, 2026-10-08).
    Read directly so it is right even while the catalog's pipeline snapshots
    are cold after a deploy."""
    from pathlib import Path
    from file_io.generic import read_header
    from file_io.sidecar import get_section
    from metadata.sidecar_io import read_image_metadata
    newest: dict[str, tuple[int, float, Path]] = {}    # stack -> (family rank, mtime, path)
    for rank, pattern in enumerate(MASTER_GLOBS):
        for path in Path(output_dir).glob(pattern):
            try:
                sid = str(read_header(path).get("STBLHASH") or "")
                mtime = path.stat().st_mtime
            except (OSError, ValueError):
                continue
            if sid and (sid not in newest or (rank, -mtime) < (newest[sid][0], -newest[sid][1])):
                newest[sid] = (rank, mtime, path)
    out = {}
    for sid, (_, _, path) in newest.items():
        meta = read_image_metadata(path)
        if meta and meta.noise_estimate and meta.master_integration_s:
            out[sid] = (meta.noise_estimate, meta.master_integration_s, get_section(path, "analysis"))
    return out


def curve_tiers(analysis: dict | None, hours_now: float) -> list[list[float]] | None:
    """Tiers read off the master's measured quality curves: tier k is the
    hours from factor k to factor k+1 of the master's integration, valued at
    the fraction of the object climbing a level per hour (field fractions
    summed over the levels, divided by the field fraction detectable at the
    last factor) in weight units, from `hours_now` (the library's
    integration, which may exceed the master's). Stops at the first tier
    under MIN_GAIN. None without a usable curve — a version-1 analysis (one
    threshold) is not one; the leaf node re-analyses on the next pass."""
    vis = (analysis or {}).get("visibility") or {}
    t_m = (analysis or {}).get("integration_s")
    curves = vis.get("fraction") if vis.get("thresholds") else None
    if not curves or not t_m or curves[0][0] > MAX_PLAUSIBLE_VISIBLE:
        return None
    t_m_h = t_m / 3600.0
    factors = vis["factors"]
    extent = max(curves[0][-1], MIN_OBJECT)         # the object: detectable at the deepest factor
    tiers = []
    for k, (f0, f1) in enumerate(zip(factors, factors[1:])):
        start = max(f0, hours_now / t_m_h)
        if f1 <= start:
            continue
        gain = sum(c[k + 1] - c[k] for c in curves) / extent / ((f1 - f0) * t_m_h)   # object-levels per hour
        if gain < MIN_GAIN:
            break
        tiers.append([(f1 - start) * t_m_h, min(MAX_WEIGHT, gain / VISIBLE_GAIN_UNIT)])
    return tiers


STAR_COLOUR = ("R", "G", "B")
NARROWBAND = ("H", "S", "O")
BROADBAND = ("L", "R", "G", "B")


def project_kind(hours_by_filter: dict[str, float]) -> str:
    """"narrowband" (R/G/B are star colour) or "lrgb", from the hours shot so
    far across the target or mosaic: whichever family has more. Nothing in
    the user's input says which a project is, but after some hours it is
    plain. A decision-framework override is the planned way to say otherwise."""
    nb = sum(h for f, h in hours_by_filter.items() if f in NARROWBAND)
    bb = sum(h for f, h in hours_by_filter.items() if f in BROADBAND)
    return "narrowband" if nb > bb else "lrgb"


def cap_tiers(tiers: list[list[float]], cap_h: float) -> list[list[float]]:
    """`tiers` cut to `cap_h` hours in total."""
    out = []
    for hours, value in tiers:
        take = min(hours, cap_h)
        if take <= 1e-9:
            break
        out.append([take, value])
        cap_h -= take
    return out


def kind_override(context, mosaic_id: str | None, object_id: str) -> str | None:
    """The user's word for the project's kind, if any (DecisionType.PROJECT_KIND)."""
    from decisions.decisions import DecisionKey, ObjectType
    key = DecisionKey.project_kind(ObjectType.MOSAIC if mosaic_id else ObjectType.TARGET, mosaic_id or object_id)
    override = (context.settings.decision_overrides or {}).get(key)
    return str(override.choice) if override is not None else None


def star_colour_cap(filt: str, kind: str, hours: float) -> float | None:
    """Hours still worth shooting in `filt` when it is star colour on a
    narrowband project: STAR_COLOUR_H less what is shot; None when the curve
    should decide (a narrowband filter, or an LRGB project)."""
    if filt in STAR_COLOUR and kind == "narrowband":
        return max(0.0, STAR_COLOUR_H - hours)
    return None


def seam_tier(sigma: float, ref: float, hours: float) -> list[float] | None:
    """The hours that bring a panel's noise within EQUAL_TOLERANCE of its
    neighbours' median (σ² ∝ 1/t), valued by the size of the step."""
    if sigma <= EQUAL_TOLERANCE * ref:
        return None
    return [hours * ((sigma / (EQUAL_TOLERANCE * ref)) ** 2 - 1), SEAM_WEIGHT * (sigma / ref - 1)]


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
        hours_by_filter: dict[str, float] = defaultdict(float)     # across the whole group
        for tid, t in targets.items():
            by_filter = defaultdict(list)
            for sid in t.stack_ids:
                if sid in context.stacks:
                    s = context.stacks[sid].get()
                    by_filter[s.filter].append(s)
            for f, stacks in by_filter.items():
                hours = sum(s.integration for s in stacks) / 3600.0
                hours_by_filter[f] += hours
                deepest = max(stacks, key=lambda s: s.integration)
                depth[(tid, f)] = (hours, _sigma_now(deepest, depths))
                analysis[(tid, f)] = (depths.get(deepest.stable_hash) or (None, None, None))[2]
                exposure[(tid, f)] = float(deepest.exposure)
        filters = sorted({f for _, f in depth if f in FALLBACK_H})
        kind = kind_override(context, mosaic_id, oid) or project_kind(hours_by_filter)
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
                        star_colour = star_colour_cap(f, kind, 0.0) is not None
                        if star_colour:
                            need = min(need, STAR_COLOUR_H)
                        tiers.append([need, 1.0 if star_colour else SEAM_WEIGHT])
                        why = f"no {f} yet; neighbours median {need:.1f} h"
                    elif sigma and ref:
                        typical = statistics.median(ref)
                        seam = seam_tier(sigma, typical, hours)
                        if seam:
                            tiers.append(seam)
                            why = f"σ {sigma / typical:.2f}x its neighbours' median"
                depth_tiers = curve_tiers(analysis.get((tid, f)), hours) if hours > 0 else None
                if depth_tiers is None:
                    deepen = max(0.0, FALLBACK_H[f] - hours - sum(h for h, _ in tiers))
                    if deepen > 0:
                        tiers.append([deepen, 1.0])
                    why = why or f"{hours:.1f} h of {FALLBACK_H[f]:.0f} h goal (master not yet analysed)"
                else:
                    cap = star_colour_cap(f, kind, hours)
                    if cap is not None:
                        depth_tiers = cap_tiers(depth_tiers, cap)
                    tiers += depth_tiers
                    why = why or (f"+{100 * depth_tiers[0][1] * VISIBLE_GAIN_UNIT:.2f}% of the object climbs a level per hour"
                                  if depth_tiers else "no more hours would improve it")
                tiers = [t for t in tiers if t[0] > 0 and t[1] > 0]
                if not tiers:
                    continue
                per_rig[rig.key].append(Candidate(
                    target_id=tid, label=label, mosaic_id=mosaic_id, filter=f,
                    ra_deg=float(t.center[0]), dec_deg=float(t.center[1]), hours=hours,
                    tiers=tiers, why=why, kind=kind,
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
