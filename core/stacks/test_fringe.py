"""The fringe frame: build, fit, apply — and the invariant that matters most,
that a stack with no fringe frame is calibrated exactly as it was before.

A synthetic sensor prints a smooth fringe; "blank sky" frames are sky x
response x (1 + a.g), dithered, with stars sprinkled on top. The map built
from them must recover g, the per-run fit must recover a, and subtracting
the two must remove the pattern from a frame that has it.
"""
from pathlib import Path

import numpy as np
import pytest

from stacks.fringe import (
    MIN_FRAMES, apply_fringe, combine_fringe, read_fringe_frame, write_fringe_frame,
)

H, W = 600, 900


def _pattern(seed=3, amp=0.02, sigma=45):
    """A smooth sensor-fixed ripple, unit-normalised, mean 0.

    `sigma` is the etalon's own scale — broad, smooth blobs, not fine
    structure — so the synthetic sensor prints something the combine can
    actually recover from dithered frames."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.standard_normal((H, W)), sigma)
    return amp * (f - f.mean()) / f.std()


def _response(seed=4):
    """Vignetting: a smooth bowl, median 1."""
    yy, xx = np.mgrid[:H, :W]
    r = ((xx - W/2)/(W/2))**2 + ((yy - H/2)/(H/2))**2
    v = 1.0 - 0.18*r
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


def test_the_map_recovers_the_pattern_the_flat_could_not(world, tmp_path):
    """The frames are calibrated the way science frames are — dark subtracted,
    flat divided — so what the combine leaves is the residual the flat did NOT
    remove. With a flat that carries none of the fringe, that residual IS the
    fringe."""
    g, resp, frames, dark = world
    paths = []
    import astropy.io.fits as pf
    for i, f in enumerate(frames):
        p = tmp_path / f"blank{i:03d}.fits"
        pf.PrimaryHDU(f.astype(np.float32)).writeto(p)
        paths.append(str(p))
    fringe, q = combine_fringe(paths, dark, resp)
    assert q["n_frames"] == len(paths) and q["flat_divided"] and q["dark_subtracted"]
    assert abs(float(np.median(fringe))) < 1e-4    # normalised by its MEDIAN, so that is the zero
    r = np.corrcoef(fringe.ravel(), g.ravel())[0, 1]
    assert r > 0.95, f"recovered pattern correlates {r:.3f} with the truth"
    # and at the right amplitude: the map is per unit sky
    assert 0.8 < float(np.std(fringe)) / float(np.std(g)) < 1.25


def test_the_map_survives_the_xisf_round_trip(tmp_path):
    """Stored with a pedestal so readers that assume [0, 1] do not clip it;
    `read_fringe_frame` takes the pedestal back off."""
    g = _pattern()
    dest = tmp_path / "fringe.xisf"
    write_fringe_frame(g.astype(np.float32), dest, {"IMAGETYP": "Master Fringe", "FILTER": "H"})
    back = read_fringe_frame(dest)
    assert back.shape == g.shape
    np.testing.assert_allclose(back, g, atol=2e-6)
    assert abs(float(np.mean(back))) < 1e-5


def test_apply_scales_the_map_to_the_sky_and_subtracts(world):
    """Arne's whole method: "scale that to match the sky background in your
    image, subtract, and voila". `sky` defaults to the frame's own median.

    On a noiseless, starless frame the removal is essentially total — there
    is no fit involved, so nothing to be approximately right about."""
    g, resp, frames, dark = world
    rel = lambda a: a / np.median(a) - 1.0        # noqa: E731

    clean = (np.full(g.shape, 300.0) * (1.0 + g)) * resp + dark
    fc = (clean - dark) / resp
    ratio = float(np.std(rel(apply_fringe(fc, g)))) / float(np.std(rel(fc)))
    assert ratio < 0.02, ratio

    # An explicit sky overrides the frame's own median, and the subtraction
    # is exactly that scalar times the map — no filtering, no regression.
    np.testing.assert_allclose(fc - 100.0 * g, apply_fringe(fc, g, sky=100.0), rtol=1e-6)

    # On a real frame the pattern goes; stars and photon noise are untouched.
    # Correlate on 16x16 BLOCK MEANS, not per pixel: the fringe is ~2% of sky
    # while a single sub's per-pixel noise is many times that, so per-pixel
    # the correlation is 0.02 and says nothing. Averaging is how you see a
    # signal under noise — it is a measurement here, not a filter on the
    # data, and nothing in the production path does it.
    def binned(a, k=16):
        """Block MEDIAN, not mean: the synthetic stars are 1200-12000 ADU
        against a 6 ADU pattern, so one star ruins a mean block. A median
        over 256 px rejects the ~0.1% of pixels a star occupies."""
        h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
        return np.median(a[:h, :w].reshape(h // k, k, w // k, k), axis=(1, 3))

    f = (frames[0] - dark) / resp
    truth = binned(g).ravel()
    before = float(np.corrcoef(binned(rel(f)).ravel(), truth)[0, 1])
    after = float(np.corrcoef(binned(rel(apply_fringe(f, g))).ravel(), truth)[0, 1])
    assert before > 0.3 and abs(after) < abs(before) / 3, (before, after)


def test_a_map_needs_enough_frames_to_be_worth_subtracting():
    """Arne's last gotcha: "since you are subtracting this, you want good
    signal/noise in the master fringe image, not always easy to obtain." A
    map built from too few frames is noise added to every frame it touches."""
    assert MIN_FRAMES >= 15


# ------------------------------------------------- the PixInsight fold
def _cal(tmp_path, name, image_type, pixels, stblhash):
    """A minimal cal master on disk with the keywords the fold reads."""
    from xisf import XISF
    p = tmp_path / name
    XISF.write(str(p), np.asarray(pixels, np.float32)[..., None], creator_app="test",
               image_metadata={"FITSKeywords": {
                   "IMAGETYP": [{"value": image_type, "comment": ""}],
                   "STBLHASH": [{"value": str(stblhash), "comment": ""}]},
                   "XISFProperties": {}})
    return str(p)


class _Dep:
    def __init__(self, stack_id):
        self.stack_id = stack_id


class _Run:
    def __init__(self, deps, metadata):
        self.dependencies = deps
        self.metadata = metadata
        self.stable_hash = "run0"


def test_without_a_fringe_frame_nothing_changes(tmp_path):
    """THE invariant. A run whose dependencies are the usual bias/dark/flat
    must be staged byte-for-byte as it was before fringe frames existed: no
    override, nothing skipped, no dark rewritten."""
    from stacks.stacking.pixinsight import _fold_fringe_into_dark
    dark = _cal(tmp_path, "dark.xisf", "Master Dark", np.full((40, 60), 500.0), "D")
    flat = _cal(tmp_path, "flat.xisf", "Master Flat", np.full((40, 60), 0.5), "F")
    run = _Run({"dark": _Dep("D"), "flat": _Dep("F")}, {})
    overrides, skip = _fold_fringe_into_dark(run, {"D": dark, "F": flat}, str(tmp_path))
    assert overrides == {} and skip == set()


def test_a_fringe_frame_is_folded_into_the_dark(tmp_path):
    """WBPP has no subtractive slot but the dark, so the map is folded in
    there — and the fringe frame itself is NEVER staged, because WBPP would
    not know what it was."""
    from stacks.fringe import write_fringe_frame
    from stacks.stacking import pixinsight as pi

    g = _pattern()
    H_, W_ = g.shape
    dark_px = np.full((H_, W_), 500.0)
    dark = _cal(tmp_path, "dark.xisf", "Master Dark", dark_px, "D")
    flat = _cal(tmp_path, "flat.xisf", "Master Flat", _response(), "F")
    fpath = tmp_path / "fringe.xisf"
    write_fringe_frame(g.astype(np.float32), fpath, {"IMAGETYP": "Master Fringe", "STBLHASH": "R"})

    class _Meta:
        def __init__(self, fn): self.filename = fn
    light = tmp_path / "light.fits"
    import astropy.io.fits as pf
    sky = 300.0
    pf.PrimaryHDU(((sky * (1 + g)) * _response() + dark_px).astype(np.float32)).writeto(light)
    run = _Run({"d": _Dep("D"), "f": _Dep("F"), "r": _Dep("R")},
               {i: _Meta(str(light)) for i in range(4)})

    overrides, skip = pi._fold_fringe_into_dark(
        run, {"D": dark, "F": flat, "R": str(fpath)}, str(tmp_path))
    assert skip == {"R"}, "the fringe frame must never reach WBPP's working dir"
    assert set(overrides) == {"D"}, "only the dark is replaced"

    # the folded dark, divided out, is the correction: (raw - dark')/flat
    from file_io.generic import read_image, read_header
    folded = np.asarray(read_image(Path(overrides["D"])), np.float64).squeeze()
    assert str(read_header(Path(overrides["D"]))["IMAGETYP"]).lower().endswith("dark")
    raw = np.asarray(read_image(Path(str(light))), np.float64)
    resp = _response()
    corrected = (raw - folded) / resp
    rel = corrected / np.median(corrected) - 1.0
    assert float(np.std(rel)) < 0.25 * float(np.std(g)), float(np.std(rel)) / float(np.std(g))


# ------------------------------------------- the calibration type it becomes
import datetime

from frozendict import frozendict

from metadata.classes import FrameHash, MetadataDict
from stacks.composite_stack import CalibrationStack, CalibrationStacksDict
from stacks.fringe import FringeStack
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


def _fringe(sid, *, filt="H", focal=1234.0, when=datetime.datetime(2026, 3, 1)):
    return FringeStack(
        content_hash=sid, stable_hash=StackHash(sid), image_type="fringe",
        camera="cam", filter=filt, mean_temperature=Celsius(-10.0),
        exposure=Seconds(300.0), integration=Seconds(0.0),
        oldest_date=when, newest_date=when, state=frozendict(),
        contents=RunsDict(), camera_id_hash=None,
        source_stack_id="src", member_files=("/a", "/b"), focal_length=focal,
        **SENSOR,
    )


def test_a_light_with_no_fringe_frame_resolves_none_and_says_nothing(monkeypatch):
    """The whole point of OPTIONAL_CALIBRATION. A fringe frame exists only
    where the user built one, so its absence is the normal case: no
    dependency, and — unlike a missing dark or flat — no concern."""
    import concerns
    from decisions.resolve import resolve_calibration
    run = _light_run("r0")
    warned = []
    monkeypatch.setattr(concerns, "warn", lambda title, detail="", **f: warned.append(f.get("missing", {})))
    resolved = resolve_calibration(_light_stack(run), CalibrationStacksDict(), frozendict())
    assert "fringe" not in resolved.contents["r0"].dependencies
    assert all("fringe" not in m for m in warned), warned


def test_the_nearest_fringe_frame_of_this_train_wins(monkeypatch):
    """Matched like a flat — same sensor, same filter, same optical train —
    and among those, nearest in time."""
    import concerns
    from decisions.resolve import resolve_calibration
    monkeypatch.setattr(concerns, "warn", lambda title, detail="", **f: None)
    run = _light_run("r1", when=datetime.datetime(2026, 3, 10))
    cal = CalibrationStacksDict({
        StackHash("near"): _fringe("near", when=datetime.datetime(2026, 3, 12)),
        StackHash("far"): _fringe("far", when=datetime.datetime(2025, 3, 12)),
        StackHash("wrongfilter"): _fringe("wrongfilter", filt="S",
                                          when=datetime.datetime(2026, 3, 11)),
        StackHash("wrongtrain"): _fringe("wrongtrain", focal=350.0,
                                         when=datetime.datetime(2026, 3, 11)),
    })
    resolved = resolve_calibration(_light_stack(run), cal, frozendict())
    assert resolved.contents["r1"].dependencies["fringe"] == StackIdDependency(stack_id="near")

    # A filter with no fringe frame at all gets none, silently.
    s_run = _light_run("r2", filt="S")
    s_cal = CalibrationStacksDict({StackHash("near"): _fringe("near")})
    s_resolved = resolve_calibration(_light_stack(s_run, "lights"), s_cal, frozendict())
    assert "fringe" not in s_resolved.contents["r2"].dependencies


def test_a_promoted_stack_yields_a_fringe_typed_calibration_stack(tmp_path):
    """`fringe_stack_for` turns a light Stack into a calibration stack whose
    ONE run is typed `fringe` — that run is what resolves the dark and flat
    the members are calibrated with, and typing it `light` would make the
    fringe frame depend on itself."""
    from stacks.fringe import fringe_stack_for, fringe_stack_id_for

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

    fringe = fringe_stack_for(_Src(), meta)
    assert fringe.kind == "fringe" and fringe.image_type == "fringe"
    assert fringe.n_frames == 2 and len(fringe.member_files) == 2
    assert fringe.source_stack_id == "srcstack"
    assert fringe.focal_length == 1234.0
    # Its id is DERIVED from the source's, never equal to it: a fringe frame
    # and the stack it came from must not collide on an id.
    assert fringe.stable_hash == fringe_stack_id_for("srcstack") != "srcstack"
    (run,) = fringe.contents.values()
    assert run.image_type == "fringe", "a light-typed run would depend on a fringe"
