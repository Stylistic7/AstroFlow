"""ImageList indexing: int, slice, and filter-string lookup.

Regression for `'slice' object has no attribute 'lower'` — StarAlignment does
`targets = images[1:]`, and a slice fell through to the filter-by-string branch
(`normalize_filter(slice)`), crashing every n>1 merge_target_channel HDR combine.
Ray-free: ImageList is a plain list subclass; we use lightweight stand-ins.
"""
from pix.image import ImageList


class _Img:
    def __init__(self, filter):
        self.filter = filter


def test_slice_returns_imagelist_sublist():
    il = ImageList([_Img("Ha"), _Img("OIII"), _Img("SII")])
    sub = il[1:]
    assert isinstance(sub, ImageList)
    assert [i.filter for i in sub] == ["OIII", "SII"]
    # The reference/targets split StarAlignment actually performs.
    assert il[0].filter == "Ha"


def test_int_index_unchanged():
    il = ImageList([_Img("Ha"), _Img("OIII")])
    assert il[0].filter == "Ha"
    assert il[-1].filter == "OIII"


def test_filter_string_lookup_unchanged():
    il = ImageList([_Img("Ha"), _Img("OIII")])
    assert il["OIII"].filter == "OIII"
    assert il["Lum"] is None
