"""Night-sky flats: build, store, resolve.

A synthetic sensor prints a smooth fringe; "blank sky" frames are
sky x response x (1 + a.g), dithered, with stars sprinkled on top. The flat
is built from DARK-SUBTRACTED frames only, so it recovers the sensor's whole
normalised response — etalon, vignetting and dust together — and DIVIDING by
it removes that response from the background and the object alike.

What is deliberately NOT here any more: there is no subtractive path, no
fold-into-the-dark, and no fourth calibration type. A night-sky flat is a
flat (`kind == "sky"`), and everything downstream treats it as one.
"""
from pathlib import Path

import numpy as np
import pytest

from stacks.sky_flat import (
    MIN_FRAMES, combine_sky_flat, read_sky_flat, write_sky_flat,
)

H, W = 600, 900


def _pattern(seed=3, amp=0.02, sigma=150):
    """A smooth sensor-fixed ripple, unit-normalised, mean 0.

    `sigma` is the etalon's own scale — broad, smooth blobs, not fine
    structure — so the synthetic sensor prints something the combine can
    actually recover from dithered frames. 150 px: the real OIII flat's
    fringes are ~1000 px, and what it holds at mote scales (< 100 px) is
    ~0.1 %; a 2 % pattern at 45 px would be mote-sized dips the dust
    cleaner is right to take out (2026-10-13)."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.standard_normal((H, W)), sigma)
    return amp * (f - f.mean()) / f.std()


def _response(seed=4, depth=0.18):
    """Vignetting: a smooth bowl, median 1."""
    yy, xx = np.mgrid[:H, :W]
    r = ((xx - W/2)/(W/2))**2 + ((yy - H/2)/(H/2))**2
    v = 1.0 - depth*r
    return v / np.median(v)


def _blank_sky(rng, g, resp, a, n=24, sky=300.0, dark=500.0, stars=60):
    """Dithered blank-sky frames: (sky*(1 + a*g) + dark) * response, plus stars
    at per-frame positions so a clipped mean can reject them."""
    out = []
    for _ in range(n):
        f = (sky * (1.0 + a * g)) * resp + dark
        for _ in range(stars):
            y, x = rng.integers(6, H - 6), rng.integers(6, W - 6)
            f[y-1:y+2, x-1:x+2] += rng.uniform(4, 40) * sky
        out.append(f + rng.normal(0, 0.02 * sky, f.shape))
    return out


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    g, resp = _pattern(), _response()
    frames = _blank_sky(rng, g, resp, a=1.0)
    dark = np.full((H, W), 500.0)
    return g, resp, frames, dark



def test_the_flat_recovers_the_sensor_pattern(tmp_path):
    """Dark subtracted and NOTHING else. What the combine leaves is the
    sensor's normalised response per unit sky. Dividing the members by a flat
    first would make this that flat's RESIDUAL instead — a different object,
    and the one that came back as the star flat inverted (corr -0.981) when
    the flat it was paired with was wrong."""
    rng = np.random.default_rng(0)
    g = _pattern()
    frames = _blank_sky(rng, g, np.ones((H, W)), a=1.0)
    dark = np.full((H, W), 500.0)
    paths = []
    import astropy.io.fits as pf
    for i, f in enumerate(frames):
        q = tmp_path / f"blank{i:03d}.fits"
        pf.PrimaryHDU(f.astype(np.float32)).writeto(q)
        paths.append(str(q))
    flat, q = combine_sky_flat(paths, dark)
    assert q["n_frames"] == len(paths) and q["dark_subtracted"]
    assert abs(float(np.median(flat))) < 1e-4   # normalised by its MEDIAN
    r = np.corrcoef(flat.ravel(), g.ravel())[0, 1]
    assert r > 0.95, f"recovered pattern correlates {r:.3f} with the truth"
    assert 0.8 < float(np.std(flat)) / float(np.std(g)) < 1.25


def test_the_flat_is_stored_MULTIPLICATIVELY(tmp_path):
    """The 2x bug, nailed down.

    A flat is `(1 + M) * 0.5`: normalising by the median recovers `1 + M`,
    the amplitude the map actually measured. Storing `M + 0.5` instead — a
    zero-mean map on an ADDITIVE pedestal — normalises to `1 + 2M` and
    applies the pattern at DOUBLE amplitude. That shipped on 2026-09-25 and
    was invisible in the image; only the rms gave it away (0.00681 against
    the map's own 0.00341).

    So: assert the NORMALISED rms equals the map's, not twice it.
    """
    g = _pattern(amp=0.02)
    g = g - np.median(g)        # what `combine_sky_flat` actually emits
    dest = tmp_path / "skyflat.xisf"
    write_sky_flat(g.astype(np.float32), dest, {"IMAGETYP": "Master Flat", "FILTER": "H"})

    from file_io.generic import read_image
    raw = np.asarray(read_image(Path(dest)), np.float64).squeeze()
    assert abs(float(np.median(raw)) - 0.5) < 1e-6, "a master flat is stored at median 0.5"

    applied = raw / float(np.median(raw))          # what WBPP divides by
    ratio = float(np.std(applied)) / float(np.std(g))
    assert abs(ratio - 1.0) < 0.02, (
        f"the flat applies the pattern at {ratio:.2f}x its measured amplitude; "
        "2.0 means it was stored as M + 0.5 instead of (1 + M) * 0.5"
    )


def test_the_flat_survives_the_xisf_round_trip(tmp_path):
    """`read_sky_flat` returns the zero-mean modulation back, by normalising
    on the median rather than trusting a recorded pedestal — so it reads a
    master written here or any other flat."""
    g = _pattern()
    g = g - np.median(g)
    dest = tmp_path / "skyflat.xisf"
    write_sky_flat(g.astype(np.float32), dest, {"IMAGETYP": "Master Flat", "FILTER": "H"})
    back = read_sky_flat(dest)
    assert back.shape == g.shape
    np.testing.assert_allclose(back, g, atol=1e-5)
    assert abs(float(np.median(back))) < 1e-5


def test_a_flat_needs_enough_frames_to_be_worth_dividing():
    """Below MIN_FRAMES the pattern has not emerged from the noise. A flat of
    zeros divides by 1 — the honest answer — and the reason rides in the
    sidecar so the UI can say why."""
    assert MIN_FRAMES >= 15


import datetime

from frozendict import frozendict

from metadata.classes import FrameHash, MetadataDict
from stacks.composite_stack import CalibrationStack, CalibrationStacksDict
from stacks.sky_flat import SkyFlatStack
from stacks.run import CalibrationRun, RunsDict
from stacks.types import StackHash, StackIdDependency
from units import Celsius, Seconds

SENSOR = dict(width_pixels=9576, height_pixels=6388)


def _frame(h, focal, when=datetime.datetime(2026, 3, 1, 22), filt="H"):
    """One light's metadata, with the keywords calibration matching reads."""
    from metadata.classes import ImageMetadata
    return ImageMetadata(
        content_hash=h, stable_hash=h, filename=Path(f"/nonexistent/{h}.fits"),
        header={"IMAGETYP": "light", "FILTER": filt, "FOCALLEN": focal,
                "INSTRUME": "cam", "EXPOSURE": 300.0, "CCD-TEMP": -10.0,
                "DATE-OBS": when,
                "NAXIS1": SENSOR["width_pixels"], "NAXIS2": SENSOR["height_pixels"]},
        channels=1, dtype=np.dtype(np.uint16), **SENSOR,
    )


def _light_run(rid, *, filt="H", focal=1234.0, when=datetime.datetime(2026, 3, 1)):
    return CalibrationRun(
        content_hash=rid, stable_hash=StackHash(rid), image_type="light",
        camera="cam", filter=filt, mean_temperature=Celsius(-10.0),
        exposure=Seconds(300.0), integration=Seconds(3000.0),
        oldest_date=when, newest_date=when, state=frozendict(),
        frames=(), metadata=MetadataDict({FrameHash("f" + rid): _frame("f" + rid, focal, filt=filt)}),
        dependencies=frozendict(), **SENSOR,
    )


def _light_stack(run, sid="light"):
    return CalibrationStack(
        content_hash=sid, stable_hash=StackHash(sid), image_type="light",
        camera="cam", filter=run.filter, mean_temperature=run.mean_temperature,
        exposure=run.exposure, integration=run.integration,
        oldest_date=run.oldest_date, newest_date=run.newest_date,
        state=frozendict(), contents=RunsDict({str(run.stable_hash): run}),
        camera_id_hash=None, **SENSOR,
    )


def _sky_flat(sid, *, filt="H", focal=1234.0, when=datetime.datetime(2026, 3, 1)):
    """A night-sky flat as the resolver sees it: image_type "flat" (so it is
    a flat), kind "sky" (so the ladder can prefer it)."""
    return SkyFlatStack(
        content_hash=sid, stable_hash=StackHash(sid), image_type="flat",
        camera="cam", filter=filt, mean_temperature=Celsius(-10.0),
        exposure=Seconds(300.0), integration=Seconds(0.0),
        oldest_date=when, newest_date=when, state=frozendict(),
        contents=RunsDict(), camera_id_hash=None,
        source_stack_id="src", member_files=("/a", "/b"), focal_length=focal,
        **SENSOR,
    )



def test_a_night_sky_flat_resolves_through_the_FLAT_slot():
    """The whole point of the collapse: it is not a fourth calibration type.
    A promoted stack lands in `dependencies["flat"]`, which is what makes it
    pickable in the ordinary flat selector alongside panel and star flats."""
    import concerns
    from decisions.resolve import resolve_calibration
    run = _light_run("r1", when=datetime.datetime(2026, 3, 10))
    cal = CalibrationStacksDict({StackHash("sky"): _sky_flat("sky")})
    resolved = resolve_calibration(_light_stack(run), cal, frozendict())
    deps = resolved.contents["r1"].dependencies
    assert deps["flat"] == StackIdDependency(stack_id="sky")
    assert "fringe" not in deps, "there is no fringe calibration type any more"


def test_the_nearest_night_sky_flat_of_this_train_wins(monkeypatch):
    """Matched like any other flat — same sensor, same filter, same optical
    train — and among those, nearest in time."""
    import concerns
    from decisions.resolve import resolve_calibration
    monkeypatch.setattr(concerns, "warn", lambda title, detail="", **f: None)
    run = _light_run("r1", when=datetime.datetime(2026, 3, 10))
    cal = CalibrationStacksDict({
        StackHash("near"): _sky_flat("near", when=datetime.datetime(2026, 3, 12)),
        StackHash("far"): _sky_flat("far", when=datetime.datetime(2025, 3, 12)),
        StackHash("wrongfilter"): _sky_flat("wrongfilter", filt="S",
                                            when=datetime.datetime(2026, 3, 11)),
        StackHash("wrongtrain"): _sky_flat("wrongtrain", focal=350.0,
                                           when=datetime.datetime(2026, 3, 11)),
    })
    resolved = resolve_calibration(_light_stack(run), cal, frozendict())
    assert resolved.contents["r1"].dependencies["flat"] == StackIdDependency(stack_id="near")


def test_a_promoted_stack_yields_a_FLAT_typed_calibration_stack(tmp_path):
    """`sky_flat_for` turns a light Stack into a calibration stack that is a
    FLAT (`image_type == "flat"`, so it appears in the flat picker and
    resolves through the flat slot) whose ONE synthetic run is typed `sky`
    (so `CALIBRATION_TOPOLOGY["sky"]` gives it a DARK). Typing the run
    `light` would make it depend on itself; typing the stack `sky` would hide
    it from the flat picker."""
    from stacks.sky_flat import sky_flat_for, sky_flat_id_for

    frames = [FrameHash("h1"), FrameHash("h2")]
    meta = MetadataDict({
        h: _frame(h, 1234.0, when=datetime.datetime(2026, 3, i + 1, 22))
        for i, h in enumerate(frames)
    })

    class _Src:
        stable_hash = StackHash("srcstack")
        camera, filter = "cam", "H"
        mean_temperature, exposure, integration = Celsius(-10.0), Seconds(300.0), Seconds(600.0)
        width_pixels, height_pixels = SENSOR["width_pixels"], SENSOR["height_pixels"]
        oldest_date = datetime.datetime(2026, 3, 1)
        newest_date = datetime.datetime(2026, 3, 2)
        camera_id_hash = None

        def all_frames(self):
            return frames

    sf = sky_flat_for(_Src(), meta)
    assert sf.kind == "sky", "the ladder prefers a sky flat by its kind"
    assert sf.image_type == "flat", "it IS a flat — that is the whole collapse"
    assert sf.n_frames == 2 and len(sf.member_files) == 2
    assert sf.source_stack_id == "srcstack"
    assert sf.focal_length == 1234.0
    assert sf.stable_hash == sky_flat_id_for("srcstack") != "srcstack"
    (run,) = sf.contents.values()
    assert run.image_type == "sky", "the RUN is typed sky so the topology gives it a dark"


def test_the_producer_is_wired_as_an_artifact_producer():
    """`build_sky_flat` must carry a `filename_fn`, or it dies on its first
    real trace and never on a test.

    `@node` decides an op writes files by resolving its RETURN ANNOTATION at
    DECORATION time. Under `from __future__ import annotations` that
    annotation is still a string while the module executes, resolution
    fails, and the op is silently treated as a non-producer (2026-09-24,
    boot99), which no unit test catches because none calls the producer body."""
    from stacks.sky_flat import build_sky_flat
    assert callable(getattr(build_sky_flat, "filename_fn", None)), (
        "build_sky_flat has no filename_fn — is `from __future__ import "
        "annotations` back at the top of stacks/sky_flat.py?"
    )
    import __future__
    import stacks.sky_flat as mod
    assert getattr(mod, "annotations", None) is not __future__.annotations, (
        "stacks/sky_flat.py has PEP 563 annotations enabled again"
    )


def test_the_producer_takes_no_flat():
    """Dark only. A flat argument would quietly reintroduce the residual
    map — the thing that lost the 2026-09-25 comparison."""
    import inspect
    from stacks.sky_flat import build_sky_flat, combine_sky_flat
    assert "flat" not in inspect.signature(combine_sky_flat).parameters
    inner = getattr(build_sky_flat, "__wrapped__", build_sky_flat)
    assert "flat" not in inspect.signature(inner).parameters



def _mote(shape, x, y, r, depth):
    """A sharp multiplicative shadow with a soft 3 px rim."""
    from scipy.ndimage import gaussian_filter
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    disk = (((xx - x) ** 2 + (yy - y) ** 2) <= r * r).astype(np.float64)
    return 1.0 - depth * gaussian_filter(disk, 1.5)


def _write(tmp_path, frames):
    import astropy.io.fits as pf
    paths = []
    for i, f in enumerate(frames):
        q = tmp_path / f"blank{i:03d}.fits"
        pf.PrimaryHDU(f.astype(np.float32)).writeto(q)
        paths.append(str(q))
    return paths


def test_the_flat_s_own_dust_motes_are_divided_out(tmp_path):
    """Two motes printed on the synthetic sensor (4 % and 2 %, r 40 and 30 px)
    on top of a broad pattern. The built flat reports both, carries neither,
    and still recovers the pattern."""
    rng = np.random.default_rng(1)
    # the sensor pattern's mote-scale content: the real OIII flat holds 0.1 %
    # at these scales once its broad fringes are divided out (2026-10-13)
    g = _pattern(amp=0.005, sigma=300)      # on a 112-block frame this is the real fringes' ~1000 px
    motes = [(300, 200, 40, 0.04), (650, 420, 30, 0.02)]
    shadow = np.ones((H, W))
    for x, y, r, depth in motes:
        shadow *= _mote((H, W), x, y, r, depth)
    # no vignetting here: at 112 blocks across, any bowl is many times
    # steeper per block than the real sensor's, and this test is about dust
    frames = _blank_sky(rng, g, shadow, a=1.0)
    flat, q = combine_sky_flat(_write(tmp_path, frames), np.full((H, W), 500.0))
    assert len(q["motes"]) == 2, q["motes"]
    for x, y, r, depth in motes:
        m = min(q["motes"], key=lambda m: (m["x"] - x) ** 2 + (m["y"] - y) ** 2)
        assert abs(m["x"] - x) < 12 and abs(m["y"] - y) < 12, (m, (x, y))
        yy, xx = np.ogrid[:H, :W]
        inside = ((xx - x) ** 2 + (yy - y) ** 2) <= (0.6 * r) ** 2
        ring = (((xx - x) ** 2 + (yy - y) ** 2) > (1.6 * r) ** 2) & (((xx - x) ** 2 + (yy - y) ** 2) <= (2.2 * r) ** 2)
        left = float(np.mean(1 + flat[ring]) - np.mean(1 + flat[inside]))
        assert abs(left) < 0.004, f"{depth:.0%} mote at {(x, y)} leaves {left:+.2%}"
    r = np.corrcoef(flat.ravel(), g.ravel())[0, 1]
    assert r > 0.95, f"pattern still recovered: {r:.3f}"


def test_a_clean_flat_reports_no_motes(tmp_path):
    rng = np.random.default_rng(2)
    frames = _blank_sky(rng, _pattern(amp=0.005, sigma=300), _response(depth=0.04), a=1.0)
    _, q = combine_sky_flat(_write(tmp_path, frames), np.full((H, W), 500.0))
    assert q["motes"] == []
