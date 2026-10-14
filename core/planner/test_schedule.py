"""The scheduler on a real night with synthetic candidates: at most
MAX_SIDE_CHANGES pier-side changes, no visit crossing the meridian, never a
visit under the minimum, and dark hours for dark work. Runs as a script too."""
from __future__ import annotations

from datetime import datetime, timezone

from planner import schedule, sky
from planner.model import Candidate, Rig

LON, LAT = -122.4139, 37.7433
NOON_UTC = datetime(2026, 10, 8, 19, tzinfo=timezone.utc)
RIG = Rig("sn", "RIG", 351.0, 2.21, (9576, 6388))


def _cand(tid, f, ra, dec, need, weight=2.0, mosaic="m"):
    return Candidate(target_id=tid, label=tid, mosaic_id=mosaic, filter=f, ra_deg=ra, dec_deg=dec,
                     hours=3.0, tiers=[[need, weight]], why="test")


def _plan(cands):
    night = sky.tonight(LON, LAT, NOON_UTC)
    return night, schedule.plan_rig(night, RIG, cands)


def test_side_changes_minimum_visits_and_sides():
    night, plan = _plan([_cand("cyg", "O", 310.0, 45.0, 3.0), _cand("cas", "O", 15.0, 62.0, 3.0),
                         _cand("ori", "H", 85.0, 0.0, 3.0)])
    assert plan.blocks
    changes = sum(1 for a, b in zip(plan.blocks, plan.blocks[1:]) if a.side != b.side)
    assert changes == len(plan.flips) <= schedule.MAX_SIDE_CHANGES
    # no gaps: the ASIAir cannot idle, so the plan tiles the whole night
    assert plan.blocks[0].start_slot == 0 and plan.blocks[-1].end_slot == night.n
    assert all(a.end_slot == b.start_slot for a, b in zip(plan.blocks, plan.blocks[1:]))
    for b in plan.blocks:
        prev = next((p for p in plan.blocks if p.end_slot == b.start_slot), None)
        continues = prev is not None and (prev.candidate.target_id, prev.candidate.filter) == \
            (b.candidate.target_id, b.candidate.filter)
        assert b.end_slot - b.start_slot >= schedule.Costs().min_block_slots or continues or \
            (b.end_slot == night.n and b.end_slot - b.start_slot >= schedule.Costs().tail_min_slots)


def test_each_visit_is_on_the_side_its_target_starts_on():
    """A visit takes its target's pier side at the visit's start (the first
    visit may have been stretched back to dusk to fill a wait)."""
    night, plan = _plan([_cand("cyg", "O", 310.0, 45.0, 6.0), _cand("cas", "O", 15.0, 62.0, 6.0),
                         _cand("ori", "H", 85.0, 0.0, 6.0)])
    for b in plan.blocks[1:]:
        ha = sky.track(night, b.candidate.ra_deg, b.candidate.dec_deg).hour_angle
        assert b.side == ("east" if ha[b.start_slot] < 0 else "west")


def test_the_night_can_start_on_a_setting_target_in_the_west():
    """A target already west at dusk — leaving for the season — gets the dusk
    hours; the old single flip (east, then west) could not start west."""
    night, plan = _plan([_cand("setting", "O", 270.0, 30.0, 2.0, weight=5.0),
                         _cand("rising", "O", 60.0, 30.0, 6.0, weight=2.0)])
    first = plan.blocks[0]
    assert first.candidate.target_id == "setting" and first.side == "west"
    assert any(b.candidate.target_id == "rising" for b in plan.blocks)


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


def test_moonless_work_is_a_tier_above_ha_sii():
    """A dark night goes to OIII (and to RGB on an LRGB project) before a
    far more valuable SII; star colour on a narrowband project is not
    moonless work."""
    night = sky.tonight(LON, LAT, NOON_UTC)
    o, s = _cand("cas", "O", 15.0, 62.0, 4.0, weight=1.0, mosaic=None), _cand("cas", "S", 15.0, 62.0, 4.0, weight=9.0, mosaic=None)
    plan = schedule.plan_rig(night, RIG, [o, s])
    assert plan.blocks[0].candidate.filter == "O"
    b = _cand("cas", "B", 15.0, 62.0, 4.0, weight=1.0, mosaic=None)
    from dataclasses import replace
    assert schedule.plan_rig(night, RIG, [replace(b, kind="lrgb"), s]).blocks[0].candidate.filter == "B"
    assert schedule.plan_rig(night, RIG, [replace(b, kind="narrowband"), s]).blocks[0].candidate.filter == "S"
    # ... and above the star: unstarred OIII beats a starred SII on a dark night
    assert schedule.plan_rig(night, RIG, [o, replace(s, rank=0)]).blocks[0].candidate.filter == "O"


if __name__ == "__main__":
    test_side_changes_minimum_visits_and_sides(); print("ok side changes / visits")
    test_scarce_filter_wins_a_dark_night(); print("ok scarcity")


def test_an_unstarred_visit_never_holds_dusk_from_starred_work():
    """In twilight only broadband is usable; starred OIII becomes usable a
    few minutes later. The night must not open with a full unstarred visit."""
    from dataclasses import replace
    night, plan = _plan([replace(_cand("star", "O", 300.0, 40.0, 4.0, weight=2.0), rank=0),
                         replace(_cand("filler", "G", 300.0, 40.0, 4.0, weight=2.0, mosaic=None), rank=1, kind="lrgb")])
    starred = [b for b in plan.blocks if b.candidate.target_id == "star"]
    assert starred and starred[0].start_slot < schedule.Costs().min_block_slots
