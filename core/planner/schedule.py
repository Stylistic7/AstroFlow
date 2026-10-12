"""One rig's night: blocks chosen greedily, at most one meridian flip.

The night is split at a flip slot T. Before T the mount stays on one pier
side and may only shoot targets east of the meridian, in blocks that end
before the target reaches it; after T only targets already west of it.
T = 0 or T = n means no flip at all. Every T is tried and the best plan
kept, ties going to no flip (Stylistic7: flips are when cables snag).

Within each side, the next block is the candidate with the best value per
hour: the need it closes, minus fixed costs that reward staying put —
re-stacking a (panel, filter) touched tonight, re-running a mosaic
channel's colour tail, and slewing. Value is scaled by the filter's
scarcity (`efficiency.scarcity`), so dark hours go to the filters only dark
hours can give, and by the target's urgency (`inventory.set_urgency`), so
a target leaving the sky this season is shot while it still can be. A block runs as long as the panel
needs and stays usable, so a night becomes a few long visits.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from planner.efficiency import efficiency
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
    flip: float = 0.5
    min_block_slots: int = 6     # 60 min: a visit is never shorter (slew, solve, guide, re-stack)
    tail_min_slots: int = 3      # 30 min: except the night's last on a new target
    idle_per_hour: float = 1.0   # an idle dark hour: so a flip moves rather than leave a gap


def _separation_deg(a: Candidate, b: Candidate) -> float:
    ra1, d1, ra2, d2 = map(math.radians, (a.ra_deg, a.dec_deg, b.ra_deg, b.dec_deg))
    c = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(ra1 - ra2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _usable(night: Night, c: Candidate, tr: Track) -> np.ndarray:
    return efficiency(c.filter, tr.alt, night.moon_alt, night.moon_illum, tr.moon_sep)


def _greedy(night: Night, cands: list[Candidate], tracks: dict, eff: dict,
            flip: int, costs: Costs, rarity: dict[str, float]) -> tuple[list[Block], float]:
    cands = copy.deepcopy(cands)
    blocks: list[Block] = []
    touched_stacks: set = set()
    touched_tails: set = set()
    prev: Candidate | None = None
    idle = 0
    t = 0
    while t < night.n:
        east = t < flip
        limit = flip if east else night.n
        best = None
        for i, c in enumerate(cands):
            if c.need_h <= 0:
                continue
            # A block closes the candidate's CURRENT tier and stops — but never
            # before the minimum visit: a small need is rounded up into the next
            # tier rather than visited for 20 minutes.
            want = max(c.tiers[0][0], costs.min_block_slots * SLOT_H)
            e, ha = eff[i], tracks[c.target_id].hour_angle
            s, gained = t, 0.0
            while s < limit and e[s] > 0 and ((ha[s] < 0) if east else (ha[s] >= 0)) and gained < want:
                gained += e[s] * SLOT_H
                s += 1
            length = s - t
            # Never a visit under the minimum, with two exceptions that keep
            # every dark minute assigned (Stylistic7, 2026-10-12): continuing the
            # previous stack costs no slew and no re-stack, so it may run however short; and
            # the night's last visit may be short, at least `tail_min_slots`.
            # The flip gets no exemption: it is the planner's own choice, so it
            # moves instead of minting a sliver.
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
            value = c.worth(gained) * rarity.get(c.filter, 1.0) * c.urgency - cost
            rate = value / (length * SLOT_H)
            if value > 0 and (best is None or rate > best[0]):
                best = (rate, i, s, gained, value)
        if best is None:
            idle += 1
            t += 1
            continue
        _, i, s, gained, value = best
        c = cands[i]
        blocks.append(Block(copy.copy(c), t, s, gained, value, "east" if east else "west"))
        c.take(gained)
        touched_stacks.add((c.target_id, c.filter))
        touched_tails.add((c.mosaic_id or c.target_id, c.filter))
        prev, t = c, s
    sides = {b.side for b in blocks}
    total = sum(b.value for b in blocks) - (costs.flip if len(sides) == 2 else 0.0) \
        - idle * SLOT_H * costs.idle_per_hour
    return blocks, total


def plan_rig(night: Night, rig: Rig, cands: list[Candidate], rarity: dict[str, float],
             costs: Costs = Costs()) -> RigPlan:
    tracks = {c.target_id: track(night, c.ra_deg, c.dec_deg) for c in cands}
    eff = {i: _usable(night, c, tracks[c.target_id]) for i, c in enumerate(cands)}
    best = None
    for flip in range(night.n + 1):
        blocks, total = _greedy(night, cands, tracks, eff, flip, costs, rarity)
        flips = len({b.side for b in blocks}) == 2
        key = (round(total, 6), not flips)
        if best is None or key > best[0]:
            best = (key, flip if flips else None, blocks)
    _, flip, blocks = best
    return RigPlan(rig=rig, flip_slot=flip, blocks=_fill_gaps(blocks, night.n))


def _fill_gaps(blocks: list[Block], n: int) -> list[Block]:
    """A plan has no gaps. The ASIAir has no way to idle in Plan mode: it
    keeps shooting the previous target until the next one starts, so a
    stretch the greedy could not assign (nothing usable, the mount waiting
    at the flip) stays on the previous target — the night's first target
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
