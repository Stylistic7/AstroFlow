"""The default flat (`decisions/resolve.py`): a night-sky flat first, then
the flats shot beside the lights (their folder, or a sibling of it), then
the nearest in time. A star flat is never the default (see
`decisions/test_decisions.py` for that, and for the NO_FLAT pin)."""
import datetime
from pathlib import Path

import numpy as np
import pytest
from frozendict import frozendict

import concerns
from decisions.resolve import resolve_calibration
from metadata.classes import FrameHash, ImageMetadata, MetadataDict
from stacks.composite_stack import CalibrationStack, CalibrationStacksDict
from stacks.run import CalibrationRun, RunsDict
from stacks.sky_flat import SkyFlatStack
from stacks.types import StackHash, StackIdDependency
from units import Celsius, Seconds

SENSOR = dict(width_pixels=9576, height_pixels=6388)
NIGHT = datetime.datetime(2026, 3, 10, 22)
LIGHTS = "/data/Heart/Lights"


def _frame(h, folder, when, image_type="light"):
    return ImageMetadata(
        content_hash=h, stable_hash=h, filename=Path(f"{folder}/{h}.fits"),
        header={"IMAGETYP": image_type, "FILTER": "H", "FOCALLEN": 1234.0, "INSTRUME": "cam",
                "EXPOSURE": 300.0, "CCD-TEMP": -10.0, "DATE-OBS": when,
                "NAXIS1": SENSOR["width_pixels"], "NAXIS2": SENSOR["height_pixels"]},
        channels=1, dtype=np.dtype(np.uint16), **SENSOR,
    )


def _run(rid, folder, when=NIGHT, image_type="light"):
    h = FrameHash("f" + rid)
    return CalibrationRun(
        content_hash=rid, stable_hash=StackHash(rid), image_type=image_type,
        camera="cam", filter="H", mean_temperature=Celsius(-10.0),
        exposure=Seconds(300.0), integration=Seconds(3000.0),
        oldest_date=when, newest_date=when, state=frozendict(),
        frames=(h,), metadata=MetadataDict({h: _frame(h, folder, when, image_type)}),
        dependencies=frozendict(), **SENSOR,
    )


def _stack(sid, run, image_type):
    return CalibrationStack(
        content_hash=sid, stable_hash=StackHash(sid), image_type=image_type,
        camera="cam", filter="H", mean_temperature=run.mean_temperature,
        exposure=run.exposure, integration=run.integration,
        oldest_date=run.oldest_date, newest_date=run.newest_date,
        state=frozendict(), contents=RunsDict({str(run.stable_hash): run}),
        camera_id_hash=None, **SENSOR,
    )


def _flat(sid, folder, when):
    """A flat stack AstroFlow built from raw flats in `folder`."""
    return _stack(sid, _run("r" + sid, folder, when, "flat"), "flat")


def _sky_flat(sid, when):
    return SkyFlatStack(
        content_hash=sid, stable_hash=StackHash(sid), image_type="flat",
        camera="cam", filter="H", mean_temperature=Celsius(-10.0),
        exposure=Seconds(300.0), integration=Seconds(0.0),
        oldest_date=when, newest_date=when, state=frozendict(),
        contents=RunsDict(), camera_id_hash=None,
        source_stack_id="src", member_files=("/a", "/b"), focal_length=1234.0,
        **SENSOR,
    )


@pytest.fixture(autouse=True)
def _no_concerns(monkeypatch):
    monkeypatch.setattr(concerns, "warn", lambda title, detail="", **f: None)


def _default_flat(cal: dict) -> str:
    run = _run("light", LIGHTS)
    resolved = resolve_calibration(_stack("light", run, "light"), CalibrationStacksDict(cal), frozendict())
    dep = resolved.contents["light"].dependencies["flat"]
    assert isinstance(dep, StackIdDependency)
    return dep.stack_id


def test_flats_in_the_lights_folder_beat_a_same_night_flat_elsewhere():
    assert _default_flat({
        StackHash("beside"): _flat("beside", LIGHTS, NIGHT - datetime.timedelta(days=40)),
        StackHash("elsewhere"): _flat("elsewhere", "/data/Other/Flats", NIGHT),
    }) == "beside"


def test_flats_in_a_sibling_folder_beat_elsewhere_and_lose_to_the_lights_folder():
    sibling = {StackHash("sibling"): _flat("sibling", "/data/Heart/Flats", NIGHT - datetime.timedelta(days=40)),
               StackHash("elsewhere"): _flat("elsewhere", "/data/Other/Flats", NIGHT)}
    assert _default_flat(sibling) == "sibling"
    sibling[StackHash("beside")] = _flat("beside", LIGHTS, NIGHT - datetime.timedelta(days=80))
    assert _default_flat(sibling) == "beside"


def test_among_affine_flats_the_nearest_in_time_wins():
    assert _default_flat({
        StackHash("old"): _flat("old", LIGHTS, NIGHT - datetime.timedelta(days=40)),
        StackHash("recent"): _flat("recent", LIGHTS, NIGHT + datetime.timedelta(days=2)),
    }) == "recent"


def test_without_any_affinity_the_nearest_in_time_wins():
    assert _default_flat({
        StackHash("far"): _flat("far", "/data/A/Flats", NIGHT - datetime.timedelta(days=40)),
        StackHash("near"): _flat("near", "/data/B/Flats", NIGHT + datetime.timedelta(days=1)),
    }) == "near"


def test_a_night_sky_flat_beats_the_flats_beside_the_lights():
    assert _default_flat({
        StackHash("beside"): _flat("beside", LIGHTS, NIGHT),
        StackHash("sky"): _sky_flat("sky", NIGHT - datetime.timedelta(days=100)),
    }) == "sky"
