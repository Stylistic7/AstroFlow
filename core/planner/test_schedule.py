"""The scheduler on a real night with synthetic candidates: at most one
meridian flip, never a visit under the minimum, east before the flip and
west after, and dark hours priced by scarcity. Runs as a script too."""
from __future__ import annotations

from datetime import datetime, timezone

from planner import efficiency, schedule, sky
from planner.model import Candidate, Rig

LON, LAT = -122.4139, 37.7433
NOON_UTC = datetime(2026, 10, 8, 19, tzinfo=timezone.utc)
RIG = Rig("sn", "RIG", 351.0, 2.21, (9576, 6388))


def _cand(tid, f, ra, dec, need, weight=2.0, mosaic="m"):
    return Candidate(target_id=tid, label=tid, mosaic_id=mosaic, filter=f, ra_deg=ra, dec_deg=dec,
                     hours=3.0, tiers=[[need, weight]], why="test")


def _plan(cands):
    night = sky.tonight(LON, LAT, NOON_UTC)
    rarity = efficiency.scarcity(efficiency.availability(LON, LAT, NOON_UTC, days=8))
    return night, schedule.plan_rig(night, RIG, cands, rarity)


def test_one_flip_at_most_minimum_visits_and_sides():
    night, plan = _plan([_cand("cyg", "O", 310.0, 45.0, 3.0), _cand("cas", "O", 15.0, 62.0, 3.0),
                         _cand("ori", "H", 85.0, 0.0, 3.0)])
    assert plan.blocks
    flips = sum(1 for a, b in zip(plan.blocks, plan.blocks[1:]) if a.side != b.side)
    assert flips <= 1
    # no gaps: the ASIAir cannot idle, so the plan tiles the whole night
    assert plan.blocks[0].start_slot == 0 and plan.blocks[-1].end_slot == night.n
    assert all(a.end_slot == b.start_slot for a, b in zip(plan.blocks, plan.blocks[1:]))
    for b in plan.blocks:
        prev = next((p for p in plan.blocks if p.end_slot == b.start_slot), None)
        continues = prev is not None and (prev.candidate.target_id, prev.candidate.filter) == \
            (b.candidate.target_id, b.candidate.filter)
        assert b.end_slot - b.start_slot >= schedule.Costs().min_block_slots or continues or \
            (b.end_slot == night.n and b.end_slot - b.start_slot >= schedule.Costs().tail_min_slots)
    # one pier side, then the other: every east visit starts before the flip,
    # every west visit ends after it (a filled gap may lean across it)
    if plan.flip_slot is not None:
        assert all(b.start_slot < plan.flip_slot for b in plan.blocks if b.side == "east")
        assert all(b.end_slot > plan.flip_slot for b in plan.blocks if b.side == "west")
    assert [b.side for b in plan.blocks] == sorted((b.side for b in plan.blocks), key=lambda s: s != "east")


def test_no_dark_minute_is_left_idle_when_a_target_is_up():
    """One target up all night: the visits tile the whole dark window, the
    last one however short."""
    night, plan = _plan([_cand("cas", "H", 15.0, 62.0, 40.0, mosaic=None)])
    covered = sorted((b.start_slot, b.end_slot) for b in plan.blocks)
    assert covered[0][0] == 0 and covered[-1][1] == night.n
    assert all(a[1] == b[0] for a, b in zip(covered, covered[1:]))


def test_scarce_filter_wins_a_dark_night():
    _, plan = _plan([_cand("cas", "O", 15.0, 62.0, 4.0, mosaic=None), _cand("cas", "S", 15.0, 62.0, 4.0, mosaic=None)])
    assert plan.blocks[0].candidate.filter == "O"


if __name__ == "__main__":
    test_one_flip_at_most_minimum_visits_and_sides(); print("ok flip / visits / sides")
    test_scarce_filter_wins_a_dark_night(); print("ok scarcity")
