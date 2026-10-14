"""The solve retry ladder: standard extraction first, then --downsample 4
for nebula-dominated fields; tombstone only after every rung fails.

Regression for 2026-08-09: 26 of 27 "unsolvable" Starbase narrowband subs
(WR134/Veil OIII + SII) extracted 1000 filament knots instead of stars at
--downsample 2 and burned the full CPU budget; the same frames solve blind
at --downsample 4."""
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits as pyfits

import plate_solving
from file_io.sidecar import get_section


@pytest.fixture(autouse=True)
def _no_gaia_check(monkeypatch):
    """These tests drive the solve ladder with FAKE solutions on a synthetic
    frame; the Gaia check-and-refit after it judges real geometry against
    real pixels and is tested in photometry/test_solution_check.py."""
    monkeypatch.setattr(plate_solving, "_gaia_checked", lambda filename, header, result: result)


@pytest.fixture
def frame(tmp_path) -> Path:
    f = tmp_path / "sub.fits"
    hdu = pyfits.PrimaryHDU(data=np.zeros((16, 16), dtype=np.float32))
    hdu.writeto(str(f))
    return f


def _downsample_of(arguments) -> str:
    return arguments[arguments.index("--downsample") + 1]


def _solution(filename) -> "plate_solving.PlateSolution":
    return plate_solving.PlateSolution(
        header={"CRVAL1": 311.5, "CRVAL2": 30.7},
        num_stars=42, center=(311.5, 30.7), image_scale=1.2,
        rotation_angle=0.1,
    )


def test_rung2_rescues_and_writes_real_solution(frame, monkeypatch):
    attempts = []

    def fake_solve(filename, arguments, debug, verify=None):
        if verify is not None:
            return None                      # the refine stage: not under test here
        attempts.append(_downsample_of(arguments))
        if _downsample_of(arguments) == "4":
            return _solution(filename)
        return None

    monkeypatch.setattr(plate_solving, "_solve_field", fake_solve)
    result = plate_solving.solve_field(frame)

    assert attempts == ["2", "4"], "ladder must try 2 then 4, in order"
    assert result is not None
    ps = get_section(frame, "plate_solution")
    assert isinstance(ps, dict) and ps.get("data") is not None, (
        "a rung-2 rescue must write a REAL solution, not a tombstone")


def test_all_rungs_fail_writes_tombstone_once(frame, monkeypatch):
    attempts = []
    monkeypatch.setattr(
        plate_solving, "_solve_field",
        lambda filename, arguments, debug:
            attempts.append(_downsample_of(arguments)))
    assert plate_solving.solve_field(frame) is None
    # Full ladder: blind ds2/ds4 on the original, then ds2/ds4 on the
    # background-FLATTENED twin (rung 3, 2026-09-01 — smooth bright
    # nebulosity defeats the extractor; see _solve_flattened). The
    # tombstone is still written exactly once, after the whole ladder.
    assert attempts == ["2", "4", "2", "4"]
    ps = get_section(frame, "plate_solution")
    assert isinstance(ps, dict) and ps.get("data") is None, "tombstone expected"


def test_rung1_success_never_reaches_rung2(frame, monkeypatch):
    attempts = []

    def fake_solve(filename, arguments, debug, verify=None):
        if verify is not None:          # the refine is not a rung
            return None
        attempts.append(_downsample_of(arguments))
        return _solution(filename)

    monkeypatch.setattr(plate_solving, "_solve_field", fake_solve)
    assert plate_solving.solve_field(frame) is not None
    assert attempts == ["2"], "no second rung after a first-rung solve"


def test_tombstone_blocks_all_rungs(frame, monkeypatch):
    """A tombstoned frame must not re-enter the ladder (don't-retry
    semantics are unchanged)."""
    monkeypatch.setattr(
        plate_solving, "_solve_field",
        lambda *a: (_ for _ in ()).throw(AssertionError("must not solve")))
    from file_io.sidecar import set_section
    set_section(frame, "plate_solution",
                {"schema": plate_solving._PLATESOLUTION_SCHEMA, "data": None})
    assert plate_solving.solve_field(frame) is None

class TestHeaderHints:
    """Rung-0 priors: conservative parse, pixels still decide the solve."""

    def test_full_hints_from_numeric_cards(self):
        from plate_solving import _header_hints
        h = _header_hints({"RA": 51.1135, "DEC": 59.3211,
                           "FOCALLEN": 937.716, "XPIXSZ": 5.4})
        assert h[:6] == ["--ra", "51.11350", "--dec", "59.32110",
                         "--radius", "5"]
        lo = float(h[h.index("--scale-low") + 1])
        hi = float(h[h.index("--scale-high") + 1])
        # 206.265 * 5.4 / 937.716 = 1.1878 arcsec/px, +/-15%
        assert 1.00 < lo < 1.02 and 1.36 < hi < 1.38

    def test_sexagesimal_ra_yields_no_hints(self):
        from plate_solving import _header_hints
        assert _header_hints({"RA": "3 24 27.242", "DEC": "+59 19 15.89"}) == []

    def test_missing_cards_yield_no_hints(self):
        from plate_solving import _header_hints
        assert _header_hints({}) == []
        assert _header_hints({"RA": 51.1}) == []

    def test_out_of_range_coordinates_rejected(self):
        from plate_solving import _header_hints
        assert _header_hints({"RA": 400.0, "DEC": 10.0}) == []
        assert _header_hints({"RA": 51.0, "DEC": -95.0}) == []

    def test_position_without_optics_still_hints(self):
        from plate_solving import _header_hints
        h = _header_hints({"RA": 51.0, "DEC": 59.0, "FOCALLEN": None})
        assert "--ra" in h and "--scale-units" not in h


def test_find_then_refine(frame, monkeypatch):
    """Two stages. FIND: the ladder as before — quadratic SIP, 1000 sources,
    --downsample 4 on the hinted rung. REFINE: the found solution handed back
    with --verify (no quad search), SIP order 3, 5000 sources, --downsample
    2 — the stage that fits the distortion from every matchable star
    (2026-09-20: a quadratic left 10-25 px at the corners of the 0.63"/px
    train; a cubic fitted in the FIND stage was noise). A refine that
    fails keeps the find."""
    with pyfits.open(frame, mode="update") as hdul:
        hdul[0].header["RA"] = 311.5; hdul[0].header["DEC"] = 30.7
        hdul[0].header["FOCALLEN"] = 350.0; hdul[0].header["XPIXSZ"] = 3.76
    calls = []

    def fake_solve(filename, arguments, debug, verify=None):
        calls.append((list(arguments), verify))
        if verify is not None:
            return plate_solving.PlateSolution(header={"CRVAL1": 311.5, "CRVAL2": 30.7, "A_ORDER": 3},
                                               num_stars=185, center=(311.5, 30.7), image_scale=1.2, rotation_angle=0.1)
        return _solution(filename)

    monkeypatch.setattr(plate_solving, "_solve_field", fake_solve)
    result = plate_solving.solve_field(frame)
    assert len(calls) == 2
    find, refine = calls
    assert find[1] is None and find[0][find[0].index("--tweak-order") + 1] == "2" and _downsample_of(find[0]) == "4"
    assert refine[1] is not None and refine[0][refine[0].index("--tweak-order") + 1] == "3"
    assert refine[0][refine[0].index("--objs") + 1] == "5000" and _downsample_of(refine[0]) == "2"
    assert "--ra" in refine[0], "the refine keeps the header hints (a scale window bounds verification)"
    assert result.num_stars == 185 and result.header["A_ORDER"] == 3, "the refined solution is the one kept"
    ps = get_section(frame, "plate_solution")
    assert ps["data"]["num_stars"] == 185


def test_refine_failure_keeps_the_find(frame, monkeypatch):
    def fake_solve(filename, arguments, debug, verify=None):
        return None if verify is not None else _solution(filename)
    monkeypatch.setattr(plate_solving, "_solve_field", fake_solve)
    result = plate_solving.solve_field(frame)
    assert result is not None and result.num_stars == 42
    assert get_section(frame, "plate_solution")["data"]["num_stars"] == 42
