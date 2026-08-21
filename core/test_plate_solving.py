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

    def fake_solve(filename, arguments, debug):
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
    assert attempts == ["2", "4"]
    ps = get_section(frame, "plate_solution")
    assert isinstance(ps, dict) and ps.get("data") is None, "tombstone expected"


def test_rung1_success_never_reaches_rung2(frame, monkeypatch):
    attempts = []

    def fake_solve(filename, arguments, debug):
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
