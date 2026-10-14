"""One rig's night: blocks chosen greedily, at most MAX_SIDE_CHANGES pier-side
changes (Stylistic7, 2026-10-14: "maybe we should allow 2").

Each visit takes the pier side its target is on when the visit starts — east
of the meridian or west of it — and runs only while the target stays on that
side, so no visit crosses the meridian. A visit on the other side from the
previous one is a side change: it costs `Costs.flip` (slew, re-centre, guiding
restart, and the cable-snag risk Stylistic7 named), and once MAX_SIDE_CHANGES have
happened only visits on the current side remain. So a night can start on a
setting target in the west, go east for the risers, and come back west after
they cross — which the old single flip (east, then west) could not express:
everything already west at dusk, every target leaving for the season, waited
for the flip.

Within each side, the next block goes to MOONLESS work while the moon is
down and any is usable (`efficiency.moonless`: OIII and faint broadband,
which have no other hours; Hα/SII have every night — Stylistic7: "dark work
requires dark hours", above the star), then to a STARRED candidate while
any is usable (`Candidate.rank` 0 — unstarred ones only when no starred
one is in a reasonable position), and within the tier to the best value per
hour: the need it closes, minus fixed costs that reward staying put —
re-stacking a (panel, filter) touched tonight, re-running a mosaic
channel's colour tail, and slewing. A block runs as long as the panel
needs and stays usable, so a night becomes a few long visits.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from planner.efficiency import efficiency, moon_rank
from planner.model import Block, Candidate, Rig, RigPlan
from planner.sky import SLOT_MIN, Night, Track, track

SLOT_H = SLOT_MIN / 60.0


@dataclass(frozen=True)
class Costs:
    """In sky-hours: what each event is worth avoiding."""
    restack: float = 0.25        # WBPP on one (panel, filter)
    tail_mosaic: float = 0.5     # the colour tail of one mosaic channel
    tail_target: float = 0.15    # a single target's channel
    slew_per_90deg: float = 0.3
    flip: float = 0.5            # one pier-side change
    min_block_slots: int = 6     # 60 min: a visit is never shorter (slew, solve, guide, re-stack)
    tail_min_slots: int = 3      # 30 min: except the night's last on a new target
    idle_per_hour: float = 1.0   # an idle dark hour


MAX_SIDE_CHANGES = 2


def _separation_deg(a: Candidate, b: Candidate) -> float:
    ra1, d1, ra2, d2 = map(math.radians, (a.ra_deg, a.dec_deg, b.ra_deg, b.dec_deg))
    c = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(ra1 - ra2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _usable(night: Night, c: Candidate, tr: Track) -> np.ndarray:
    return efficiency(c.filter, tr.alt, night.moon_alt, night.moon_illum, tr.moon_sep, c.kind)


def _greedy(night: Night, cands: list[Candidate], tracks: dict, eff: dict,
            costs: Costs, max_changes: int = MAX_SIDE_CHANGES) -> tuple[list[Block], list[int], float]:
    cands = copy.deepcopy(cands)
    blocks: list[Block] = []
    changes: list[int] = []          # the slots where the pier side changed
    touched_stacks: set = set()
    touched_tails: set = set()
    prev: Candidate | None = None
    side: str | None = None
    idle = 0
    t = 0
    while t < night.n:
        best = None
        for i, c in enumerate(cands):
            if c.need_h <= 0:
                continue
            e, ha = eff[i], tracks[c.target_id].hour_angle
            if e[t] <= 0:
                continue
            here = "east" if ha[t] < 0 else "west"
            if side is not None and here != side and len(changes) >= max_changes:
                continue
            # A block closes the candidate's CURRENT tier and stops — but never
            # before the minimum visit: a small need is rounded up into the next
            # tier rather than visited for 20 minutes. It never crosses the
            # meridian: it ends where the target changes side.
            want = max(c.tiers[0][0], costs.min_block_slots * SLOT_H)
            s, gained = t, 0.0
            while s < night.n and e[s] > 0 and (("east" if ha[s] < 0 else "west") == here) and gained < want:
                gained += e[s] * SLOT_H
                s += 1
            length = s - t
            # Never a visit under the minimum, with two exceptions that keep
            # every dark minute assigned (Stylistic7, 2026-10-12): continuing the
            # previous stack costs no slew and no re-stack, so it may run however
            # short; and the night's last visit may be short, at least
            # `tail_min_slots`.
            same = prev is not None and (prev.target_id, prev.filter) == (c.target_id, c.filter)
            if length == 0 or (length < costs.min_block_slots and not same and
                               not (s >= night.n and length >= costs.tail_min_slots)):
                continue
            gained = min(gained, c.need_h)
            stack_key, tail_key = (c.target_id, c.filter), (c.mosaic_id or c.target_id, c.filter)
            cost = (costs.restack if stack_key not in touched_stacks else 0.0) + \
                   ((costs.tail_mosaic if c.mosaic_id else costs.tail_target) if tail_key not in touched_tails else 0.0)
            if prev is not None and prev.target_id != c.target_id:
                cost += costs.slew_per_90deg * _separation_deg(prev, c) / 90.0
            if side is not None and here != side:
                cost += costs.flip
            value = c.worth(gained) - cost
            rate = value / (length * SLOT_H)
            key = (-moon_rank(c.filter, c.kind), -c.rank, rate)   # moonless work, star, value
            if value > 0 and (best is None or key > best[0]):
                best = (key, i, s, gained, value, here)
        if best is not None and cands[best[1]].rank != 0 and _starred_soon(cands, eff, t, costs.min_block_slots):
            best = None     # wait for it: an unstarred visit would lock starred work out for a whole visit
        if best is None:
            idle += 1
            t += 1
            continue
        _, i, s, gained, value, here = best
        c = cands[i]
        if side is not None and here != side:
            changes.append(t)
        side = here
        blocks.append(Block(copy.copy(c), t, s, gained, value, here))
        c.take(gained)
        touched_stacks.add((c.target_id, c.filter))
        touched_tails.add((c.mosaic_id or c.target_id, c.filter))
        prev, t = c, s
    total = sum(b.value for b in blocks) - idle * SLOT_H * costs.idle_per_hour
    return blocks, changes, total


def _starred_soon(cands: list[Candidate], eff: dict, t: int, within: int) -> bool:
    """Whether a starred candidate with need left becomes usable in the next
    `within` slots (not now: then it would have been chosen). Dusk is the
    case: twilight allows broadband only, and a one-slot-early unstarred
    visit used to hold the first dark hour (2026-10-14: Andromeda G for 80
    min while 75a1 and 7098 waited)."""
    for i, c in enumerate(cands):
        if c.rank == 0 and c.need_h > 0 and eff[i][t] <= 0 and any(eff[i][t + 1:t + within] > 0):
            return True
    return False


def plan_rig(night: Night, rig: Rig, cands: list[Candidate],
             costs: Costs = Costs()) -> RigPlan:
    tracks = {c.target_id: track(night, c.ra_deg, c.dec_deg) for c in cands}
    eff = {i: _usable(night, c, tracks[c.target_id]) for i, c in enumerate(cands)}
    blocks, changes, _ = _greedy(night, cands, tracks, eff, costs)
    return RigPlan(rig=rig, flips=changes, blocks=_fill_gaps(blocks, night.n))


def _fill_gaps(blocks: list[Block], n: int) -> list[Block]:
    """A plan has no gaps. The ASIAir has no way to idle in Plan mode: it
    keeps shooting the previous target until the next one starts, so a
    stretch the greedy could not assign (nothing usable, the mount waiting
    at a side change) stays on the previous target — the night's first target
    takes any wait before it. Values are unchanged: the stretch earned
    nothing, it is just where the rig will be."""
    if not blocks:
        return blocks
    out = [copy.copy(b) for b in sorted(blocks, key=lambda b: b.start_slot)]
    out[0].start_slot = 0
    for prev, nxt in zip(out, out[1:]):
        prev.end_slot = nxt.start_slot
    out[-1].end_slot = n
    return out
