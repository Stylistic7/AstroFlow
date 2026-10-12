"""The horizon on a real sky: dark blocks only, per-night indices, and
edges only where a candidate is usable."""
from datetime import datetime, timezone

import numpy as np

from planner import horizon
from planner.model import Candidate, Rig

LON, LAT = -122.42, 37.77
NOON_UTC = datetime(2026, 10, 12, 19, tzinfo=timezone.utc)
RIG = Rig("rc71", "RC71", 351.0, 2.21, (9576, 6388))


def _cand(tid, f, ra, dec):
    return Candidate(target_id=tid, label=tid, mosaic_id=None, filter=f, ra_deg=ra, dec_deg=dec,
                     hours=1.0, tiers=[[2.0, 1.0]], why="test")


def test_dark_blocks_cover_two_nights_of_astronomical_darkness():
    h = horizon.dark_blocks(LON, LAT, NOON_UTC, nights=2)
    assert set(h.night) == {0, 1}
    per_night = np.bincount(h.night)
    assert 16 <= per_night[0] <= 24 and 16 <= per_night[1] <= 24     # ~10 h of darkness in October
    assert h.block_h == 0.5
    assert horizon.night_dates(NOON_UTC, LON, 2)[0].isoformat() == "2026-10-12"


def test_edges_only_where_usable():
    h = horizon.dark_blocks(LON, LAT, NOON_UTC, nights=2)
    cands = [("rc71", _cand("cyg", "O", 306.5, 41.0)),      # Cygnus: up in the evening
             ("rc71", _cand("south", "H", 306.5, -80.0))]   # never above 30° from SF
    slots, cap = horizon.supply([RIG], h, [1.0] * h.n)
    assert len(slots) == h.n and all(c == 0.5 for c in cap)
    es = horizon.edges(slots, h, cands)
    assert es and all(e.need == 0 for e in es)
    cyg = h.track(306.5, 41.0)
    for e in es:
        assert cyg.alt[slots[e.slot].block] >= 40.0 and h.moon_alt[slots[e.slot].block] <= 0
        assert 0 < e.efficiency <= 1.0


def test_a_deadline_drops_later_edges():
    h = horizon.dark_blocks(LON, LAT, NOON_UTC, nights=2)
    cands = [("rc71", _cand("cyg", "O", 306.5, 41.0))]
    slots, _ = horizon.supply([RIG], h, [1.0] * h.n)
    nights = {h.night[slots[e.slot].block] for e in horizon.edges(slots, h, cands, last_night={0: 0})}
    assert nights == {0}
