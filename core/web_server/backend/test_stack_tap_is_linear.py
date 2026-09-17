"""Colorspace comes from the payload's own COLORSPC tag, not its display name.

Everything this app computes and serves is LINEAR; the non-linear step lives
in the frontend WebGPU shaders. The one deliberate exception is
`mosaics/merging/stretched.py`, which merges in stretched space because
photometry is intentionally broken there (decision 2026-08-21) and stamps its
output `COLORSPC: "display"` so the viewer "applies taste, not a second
autostretch".

Classifying by display NAME got both directions wrong, because a name is not
identity:

  * `merge_images_stretched` and `polish_background` BOTH tap
    `display="Mosaic Starless"`, which a name-keyed list called linear — so the
    already-stretched deliverable was marked linear and stretched twice;
  * every stack tap (`Master`, `Post-Processed`) was in no list, fell to the
    `display` default, and was served raw linear — near-black, since a master
    sits at sky ~0.005 with star cores near 1.0 (live 2026-09-11, stack
    6d5b69676e23fdd9 `Master`/O returned `"colorspace":"display"`).
"""
import json

import pytest

from web_server.backend.tiles import _declared_colorspace, _is_display_referred


class _Img:
    """Minimal pyvips stand-in: `format`, `bands`, and `get()`."""

    def __init__(self, fmt="float", bands=1, colorspc=None, raises=False):
        self.format = fmt
        self.bands = bands
        self._raises = raises
        self._desc = (
            None if colorspc is None
            else json.dumps({"PIPELINE": "AstroFlow-OSS", "COLORSPC": colorspc})
        )

    def get(self, field):
        if self._raises or self._desc is None:
            raise RuntimeError("no such field")
        return self._desc


def test_untagged_float_payload_is_linear():
    """The default. A WBPP master carries no COLORSPC and must be stretched."""
    assert _declared_colorspace(_Img()) is None
    assert _is_display_referred(_Img()) is False


def test_the_stretched_merge_opts_out_by_its_tag():
    """`merge_images_stretched` stamps COLORSPC=display — no second stretch."""
    assert _is_display_referred(_Img(colorspc="display")) is True


def test_linear_sho_tag_is_not_display_referred():
    assert _declared_colorspace(_Img(colorspc="linear-sho")) == "linear-sho"
    assert _is_display_referred(_Img(colorspc="linear-sho")) is False


def test_uchar_payloads_are_display_ready_by_encoding():
    """Subframe previews are uint8 and get rescaled /255 by the tile route;
    stretching them again would double-stretch."""
    assert _is_display_referred(_Img(fmt="uchar")) is True


@pytest.mark.parametrize("bad", [_Img(raises=True), _Img()])
def test_a_missing_or_unreadable_tag_degrades_to_linear(bad):
    """Never guess `display` from a read failure — that serves raw linear data
    to the shader, which is the near-black failure this replaced."""
    assert _declared_colorspace(bad) is None
    assert _is_display_referred(bad) is False
