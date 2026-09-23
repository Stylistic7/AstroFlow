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
    MIN_FRAMES, combine_fringe, read_fringe_frame, write_fringe_frame,
)
from stacks.fringe_gain import GAIN_RANGE, apply_fringe, fit_amplitude

H, W = 600, 900


def _pattern(seed=3, amp=0.02, sigma=45):
    """A smooth sensor-fixed ripple, unit-normalised, mean 0.

    `sigma` puts the structure inside `fringe_gain.BAND_PX` (64-400 px), which
    is where the real etalon's power lives and the only band the fit looks at.
    A finer pattern is filtered away before the regression ever sees it."""
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


def test_the_amplitude_is_fitted_per_run(world):
    """The map is a SHAPE; how strongly a frame carries it depends on how much
    of its sky is the monochromatic component, which moves with conditions. So
    it is measured, not assumed."""
    g, resp, _, dark = world
    rng = np.random.default_rng(11)
    for a_true in (0.6, 1.0, 1.6):
        frames = _blank_sky(rng, g, resp, a=a_true, n=8, stars=20)
        fit = fit_amplitude(range(len(frames)), g, dark=dark, flat=resp,
                            read=lambda i: frames[int(i)])
        assert "reason" not in fit, fit
        assert abs(fit["A"] - a_true) < 0.25, (a_true, fit["A"])
        assert abs(fit["corr"]) > 0.3      # diluted by photon noise in the band


def test_a_sky_without_the_pattern_subtracts_nothing(world):
    """The neutral value is ZERO — unlike the star flat's gain, where "I do not
    believe this" once resolved to the maximum and printed the flat's own
    structure into the masters. Here the safe fallback and the do-nothing
    fallback are the same number."""
    g, resp, _, dark = world
    rng = np.random.default_rng(12)
    frames = _blank_sky(rng, g, resp, a=0.0, n=8, stars=20)   # no fringe in this sky
    foreign = _pattern(seed=99)                                # and a foreign map
    fit = fit_amplitude(range(len(frames)), foreign, dark=dark, flat=resp,
                        read=lambda i: frames[int(i)])
    assert fit["A"] == 0.0
    assert "reason" in fit
    fewer = fit_amplitude(range(2), g, dark=dark, flat=resp, read=lambda i: frames[int(i)])
    assert fewer["A"] == 0.0 and "only" in fewer["reason"]


def test_apply_removes_the_pattern(world):
    """frame - A * sky * fringe, in the frame's own units.

    Judged by CORRELATION with the pattern, not by scatter: a real frame's
    scatter in that band is dominated by stars and photon noise, which the
    correction neither should nor does touch, so an rms ratio understates it
    badly (it reads 0.85 while the pattern is entirely gone)."""
    from stacks.fringe_gain import _bandpass, _block_mean
    g, resp, frames, dark = world
    band = lambda a: _bandpass(_block_mean(a / np.median(a) - 1.0, 8))  # noqa: E731
    truth = _bandpass(_block_mean(g, 8))
    f = (frames[0] - dark) / resp
    before = float(np.corrcoef(band(f).ravel(), truth.ravel())[0, 1])
    after = float(np.corrcoef(band(apply_fringe(f, g, 1.0)).ravel(), truth.ravel())[0, 1])
    assert before > 0.4 and abs(after) < 0.1, (before, after)

    # on a noiseless, starless frame the removal is essentially total
    clean = (np.full(g.shape, 300.0) * (1.0 + g)) * resp + dark
    fc = (clean - dark) / resp
    ratio = float(np.std(band(apply_fringe(fc, g, 1.0)))) / float(np.std(band(fc)))
    assert ratio < 0.02, ratio


def test_gain_range_brackets_one():
    """A sky can carry none of the pattern or somewhat more than the map has;
    anything far outside that is fitting something else."""
    assert GAIN_RANGE[0] <= 0.0 < 1.0 < GAIN_RANGE[1]
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


def test_a_fringe_frame_is_folded_into_the_dark(tmp_path, monkeypatch):
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

    monkeypatch.setattr(pi, "fit_amplitude", lambda *a, **k: {"A": 1.0, "n_used": 4, "corr": 0.9},
                        raising=False)
    import stacks.fringe_gain as fg
    monkeypatch.setattr(fg, "fit_amplitude", lambda *a, **k: {"A": 1.0, "n_used": 4, "corr": 0.9})

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
