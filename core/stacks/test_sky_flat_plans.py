"""Night-sky flat plans (`stacks/sky_flat_plans.py`) and the name rule that
makes a stack of SkyFlat panes a night-sky flat source (`stacks/sky_flat.py`)."""
import datetime as dt

import numpy as np

from stacks import sky_flat_plans as p
from stacks.sky_flat import is_sky_flat_source

SENSOR = (9576, 6388)


def _sep_deg(a, b):
    a, b = np.radians(a), np.radians(b)
    return float(np.degrees(np.arccos(np.clip(
        np.sin(a[1]) * np.sin(b[1]) + np.cos(a[1]) * np.cos(b[1]) * np.cos(a[0] - b[0]), -1, 1))))


def test_the_field_catalogue_is_well_formed():
    fields = p.fields()
    assert len(fields) > 100
    assert len({f["id"] for f in fields}) == len(fields)
    assert {f["tier"] for f in fields} == {"wide", "narrow"}
    for f in fields:
        assert -90 <= f["dec_deg"] <= 90 and 0 <= f["ra_deg"] < 360
        assert abs(f["gal_lat_deg"]) >= 25
    # both hemispheres, every season
    assert {int(f["ra_deg"] // 30) for f in fields} == set(range(12))
    assert any(f["dec_deg"] < -30 for f in fields) and any(f["dec_deg"] > 60 for f in fields)


def test_panes_land_in_one_stack_but_move_the_stars():
    """Grouping admits frames within 10% of the short side of the FIRST
    frame; the panes stay inside half of that, yet move stars hundreds of
    pixels between pointings."""
    for focal in (351.0, 1233.0):
        long_side, short = p.field_of_view_deg(focal, 3.76, SENSOR)
        scale_deg = short / min(SENSOR)
        field = p.fields()[0]
        centres = p.pane_centres(field, short)
        assert len(centres) == p.PANES
        first = centres[0]
        for c in centres[1:]:
            d = _sep_deg(first, c)
            assert d <= 0.05 * short, (focal, d, short)
            assert d / scale_deg >= 200, (focal, d / scale_deg)


def test_pane_names_are_recognised_from_asiair_file_names():
    names = p.pane_names("O", "W239+35")
    assert names[0] == "SkyFlat O W239+35 1"
    files = [f"/Raw/2026-10-16/350mm/Light_{n}_300.0s_Bin1_6200MM_O_20261016-0{i}1500_90deg_0001.fit"
             for i, n in enumerate(names)]
    assert [p.frame_target_name(f) for f in files] == names
    assert p.is_named_sky_flat(files)
    assert not p.is_named_sky_flat(["/x/Light_Pane 1_300.0s_Bin1_6200MM_O_20261016-011500_90deg_0001.fit"])
    # a stack only partly of SkyFlat panes is not one
    mixed = files[:2] + ["/x/Light_75a1 Panel 7_300.0s_Bin1_6200MM_O_20261016-011500_90deg_0001.fit"] * 3
    assert not p.is_named_sky_flat(mixed)


class _Key:
    def __init__(self, sid):
        from decisions.decisions import DecisionType
        self.decision_type, self.subject_id = DecisionType.SKY_FLAT_SOURCE, sid


class _Override:
    def __init__(self, choice):
        self.choice = choice


def test_a_named_stack_is_a_source_until_demoted_and_any_stack_can_be_promoted():
    from decisions.decisions import BinaryChoice
    named = ["/x/Light_SkyFlat H N021+31 1_300.0s_Bin1_6200MM_H_20261016-011500_90deg_0001.fit"] * 4
    plain = ["/x/Light_Pane 1_300.0s_Bin1_6200MM_H_20261016-011500_90deg_0001.fit"] * 4
    assert is_sky_flat_source("s1", named, {})
    assert not is_sky_flat_source("s2", plain, {})
    assert not is_sky_flat_source("s1", named, {_Key("s1"): _Override(BinaryChoice.EXCLUDE)})
    assert is_sky_flat_source("s2", plain, {_Key("s2"): _Override(BinaryChoice.INCLUDE)})


def test_fields_that_clear_the_rig_come_first():
    ranked = p.rank_fields(351.0, 3.76, SENSOR, dt.date(2026, 10, 15), -122.42, 37.75,
                           p.plan_minutes(300, 4), limit=6)
    assert ranked, "some field is usable from San Francisco on an October night"
    fits = [r["fits"] for r in ranked]
    assert fits == sorted(fits, reverse=True)
    for r in ranked:
        assert r["max_alt_deg"] >= p.MIN_ALT_DEG and r["window_h"] > 0
