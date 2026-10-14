"""Solver-hint inheritance: outputs from external tools that drop the
capture cards get them back from the producing op's input image."""
from products.registry import _inherited_cards


def _meta(**header):
    from metadata.classes import ImageMetadata
    m = ImageMetadata.__new__(ImageMetadata)
    object.__setattr__(m, "header", header)
    return m


def test_hints_inherited_when_output_has_none():
    src = _meta(RA=51.1135, DEC=59.3211, FOCALLEN=937.7, XPIXSZ=5.4)
    got = _inherited_cards({}, ("filter", src))
    assert got == {"RA": 51.1135, "DEC": 59.3211,
                   "FOCALLEN": 937.7, "XPIXSZ": 5.4}


def test_output_cards_are_never_overwritten():
    src = _meta(RA=51.0, DEC=59.0, FOCALLEN=900.0, XPIXSZ=5.4)
    out = {"RA": 12.0, "DEC": -3.0, "FOCALLEN": 500.0, "XPIXSZ": 3.76}
    assert _inherited_cards(out, (src,)) == {}


def test_partial_fill_only_touches_missing_cards():
    src = _meta(RA=51.0, DEC=59.0, FOCALLEN=900.0, XPIXSZ=5.4)
    got = _inherited_cards({"RA": 12.0, "DEC": -3.0}, (src,))
    assert got == {"FOCALLEN": 900.0, "XPIXSZ": 5.4}


def test_no_image_inputs_yields_nothing():
    assert _inherited_cards({}, ("filter", 3, None)) == {}


def test_first_input_carrying_a_card_wins():
    a = _meta(RA=51.0)
    b = _meta(RA=99.0, DEC=59.0)
    got = _inherited_cards({}, (a, b))
    assert got["RA"] == 51.0 and got["DEC"] == 59.0


def test_filter_is_inherited_when_the_output_reads_none():
    """rc-astro's BXT/SXT TIFFs carry no FILTER; read through
    normalize_header that is "none". A deconvolved or starless channel is
    still its input's band (2026-10-14: every BXT and SXT output read 'none')."""
    src = _meta(FILTER="O", RA=1.0, DEC=2.0, FOCALLEN=3.0, XPIXSZ=4.0)
    assert _inherited_cards({"FILTER": "none"}, (src,))["FILTER"] == "O"
    assert "FILTER" not in _inherited_cards({"FILTER": "H"}, (src,))
    assert "FILTER" not in _inherited_cards({"FILTER": "none"}, (_meta(FILTER="none"),))
