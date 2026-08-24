"""ZERO MEANS NO DATA — enforced, not remembered.

`place_image` fills uncovered canvas with exact 0, and everything
downstream distinguishes data from border with `> 0`. Any unmasked
additive step breaks that silently: PixInsight's BackgroundNeutralization
did exactly this by hand on 2026-08-22, turning the ragged border into
three different small constants (a coloured fringe) and lifting the
counted-pixel fraction from 87% to 100%.

These tests pin the invariant for every op AstroFlow owns in the placed/
merged path. An op that shifts, scales, or rectifies a canvas must leave
exact zeros exactly zero.
"""
from pathlib import Path

import numpy as np

from mosaics.merging.oss import _blend_many, _equalize_sky, _UNSET_PEDESTAL
from mosaics.merging.stretched import _stretch_nonzero
from mosaics.placement.panel_io import write_placed_panel
from stacks.post_processing.background_pedestal import apply_background_pedestal


def _ragged(h=300, w=400, seed=0):
    """A panel with a genuinely ragged no-data border, like a placement."""
    rng = np.random.default_rng(seed)
    a = np.zeros((h, w), dtype=np.float32)
    a[20:h - 15, 25:w - 30] = 0.005 + rng.normal(0, 2e-4, (h - 35, w - 55))
    for k in range(h):                       # diagonal ragged corner
        a[k, :max(0, 60 - k)] = 0.0
    return a


def test_equalize_sky_preserves_zeros():
    a = _ragged()
    zeros = a == 0
    _equalize_sky([a], [a > 0], _UNSET_PEDESTAL)
    assert (a[zeros] == 0).all(), "sky equalization must not lift no-data"


def test_equalize_sky_preserves_zeros_with_explicit_target():
    a = _ragged(seed=1)
    zeros = a == 0
    _equalize_sky([a], [a > 0], 0.25)        # display-space caller
    assert (a[zeros] == 0).all()


def test_autostretch_panel_preserves_zeros():
    a = _ragged(seed=2)
    zeros = a == 0
    out = _stretch_nonzero(a)
    assert (out[zeros] == 0).all(), "per-panel autostretch must not lift no-data"


def test_background_pedestal_preserves_zeros():
    a = _ragged(seed=3)
    zeros = a == 0
    out = apply_background_pedestal(a, target=0.005)
    assert (out[zeros] == 0).all(), "pedestal must not lift no-data"


def test_blend_preserves_zeros_end_to_end(tmp_path):
    """The whole fused merge: two ragged panels in, uncovered canvas out at
    exact zero — with level matching and flux scaling both active."""
    W, H, span = 1200, 800, 600
    paths = []
    for k, (x0, gain, sky) in enumerate(zip((60, 520), (1.0, 1.4),
                                            (0.00504, 0.00623))):
        rng = np.random.default_rng(40 + k)
        img = (sky + np.abs(rng.normal(1e-3, 5e-4, (H, span)))).astype(np.float32)
        img[:40, :] = 0.0                    # ragged top edge inside the bbox
        p = Path(tmp_path) / f"nd{k}.tiff"
        write_placed_panel(p, img, (x0, 0), (W, H))
        paths.append(p)
    merged = _blend_many(paths, level_match=True)
    assert (merged[:, :60] == 0).all(), "canvas left of both panels must be 0"
    assert (merged[:, 1120:] == 0).all(), "canvas right of both panels must be 0"
    assert (merged[:40, 60:520] == 0).all(), "ragged edge must survive the merge"
    assert merged[400, 300] > 0, "covered canvas must carry data"


def test_blend_single_panel_preserves_zeros(tmp_path):
    img = _ragged(400, 500, seed=5)
    p = Path(tmp_path) / "solo.tiff"
    r0, r1 = 0, img.shape[0]
    write_placed_panel(p, img, (0, 0), (img.shape[1], img.shape[0]))
    merged = _blend_many([p])
    assert (merged[img == 0] == 0).all(), "passthrough must preserve no-data"
