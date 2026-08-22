"""Solver-hint inheritance: outputs from external tools that drop the
capture cards get them back from the producing op's input image."""
from products.registry import _inherited_solver_hints


def _meta(**header):
    from metadata.classes import ImageMetadata
    m = ImageMetadata.__new__(ImageMetadata)
    object.__setattr__(m, "header", header)
    return m


def test_hints_inherited_when_output_has_none():
    src = _meta(RA=51.1135, DEC=59.3211, FOCALLEN=937.7, XPIXSZ=5.4)
    got = _inherited_solver_hints({}, ("filter", src))
    assert got == {"RA": 51.1135, "DEC": 59.3211,
                   "FOCALLEN": 937.7, "XPIXSZ": 5.4}


def test_output_cards_are_never_overwritten():
    src = _meta(RA=51.0, DEC=59.0, FOCALLEN=900.0, XPIXSZ=5.4)
    out = {"RA": 12.0, "DEC": -3.0, "FOCALLEN": 500.0, "XPIXSZ": 3.76}
    assert _inherited_solver_hints(out, (src,)) == {}


def test_partial_fill_only_touches_missing_cards():
    src = _meta(RA=51.0, DEC=59.0, FOCALLEN=900.0, XPIXSZ=5.4)
    got = _inherited_solver_hints({"RA": 12.0, "DEC": -3.0}, (src,))
    assert got == {"FOCALLEN": 900.0, "XPIXSZ": 5.4}


def test_no_image_inputs_yields_nothing():
    assert _inherited_solver_hints({}, ("filter", 3, None)) == {}


def test_first_input_carrying_a_card_wins():
    a = _meta(RA=51.0)
    b = _meta(RA=99.0, DEC=59.0)
    got = _inherited_solver_hints({}, (a, b))
    assert got["RA"] == 51.0 and got["DEC"] == 59.0
