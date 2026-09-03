"""The overlap-constants level solve — ground-truth lab.

Panels with KNOWN injected sky offsets on a shared canvas must come out of
`_blend_many(level_solve="overlap")` with matching skies, in exactly the
scene where the absolute-quantile equalizer fails: nebulosity covering
every panel's lower quantile differently. The quantile estimator itself
measured blind to the injected offsets on the 24-panel gamma Cyg data
(2026-08-31: quantile spread 1e-4 of pedestal, true offsets 1.3e-4)."""
import numpy as np
import pytest

from mosaics.merging.oss import (
    _blend_many, _overlap_offsets, _panel_bbox, _solve_level_offsets,
)
from mosaics.placement.panel_io import write_placed_panel

PED = 0.005
RNG = np.random.default_rng(7)


def _sky_field(canvas):
    """ONE shared sky: pedestal + two nebula blobs in CANVAS coordinates.
    Every panel samples THIS field — as real panels image the same sky —
    so overlap differences carry only the injected offsets + noise."""
    yy, xx = np.mgrid[0:canvas[0], 0:canvas[1]].astype(np.float32)
    f = np.full(canvas, PED, np.float32)
    for cy, cx, amp, sig in ((120, 110, 6e-3, 45), (80, 320, 2e-3, 30)):
        f += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2)
                            / (2 * sig ** 2))).astype(np.float32)
    return f


def _panel(field, r0, r1, c0, c1, offset):
    img = np.zeros(field.shape, np.float32)
    tile = field[r0:r1, c0:c1] + RNG.normal(0, 2e-4, (r1 - r0, c1 - c0)).astype(np.float32)
    img[r0:r1, c0:c1] = np.maximum(tile + offset, 1e-6)
    return img


def _write(tmp_path, name, img):
    p = tmp_path / f"{name}.tiff"
    r0, r1, c0, c1 = _panel_bbox(img > 0)
    write_placed_panel(
        p, img[r0:r1, c0:c1].astype(np.float32),
        offset_xy=(c0, r0), canvas_size=(img.shape[1], img.shape[0]),
    )
    return p


def _three_offset_panels(tmp_path):
    """Three overlapping panels of ONE sky, injected offsets (0,+8e-4,-5e-4)."""
    canvas = (240, 560)
    field = _sky_field(canvas)
    offs = [0.0, 8e-4, -5e-4]
    imgs = [
        _panel(field, 20, 220, 0, 220, offs[0]),
        _panel(field, 20, 220, 170, 390, offs[1]),
        _panel(field, 20, 220, 340, 560, offs[2]),
    ]
    return field, offs, [_write(tmp_path, f"p{i}", im)
                         for i, im in enumerate(imgs)]


def test_solver_recovers_injected_offsets(tmp_path):
    _, offs, paths = _three_offset_panels(tmp_path)
    from mosaics.placement.panel_io import read_placed_panel
    bboxes, masks, crops = [], [], []
    for p in paths:
        arr, (x0, y0), _ = read_placed_panel(p)
        t = _panel_bbox(arr > 0)
        r0, r1, c0, c1 = t
        crop = np.ascontiguousarray(arr[r0:r1, c0:c1])
        bboxes.append((y0 + r0, y0 + r1, x0 + c0, x0 + c1))
        masks.append(crop > 0)
        crops.append(crop)
    pairs = _overlap_offsets(bboxes, masks, crops)
    assert len(pairs) == 2  # chain adjacency: 0-1, 1-2
    solved = _solve_level_offsets(3, pairs)
    truth = np.array(offs) - np.mean(offs)  # solver's mean-zero gauge
    assert np.max(np.abs((solved - solved.mean()) - truth)) < 1e-4


def test_blend_output_skies_agree(tmp_path):
    """End to end: the blend must reproduce the SHARED sky field with the
    injected steps gone — residual vs the known field, per exclusive
    region, must agree across panels and sit on the pedestal."""
    field, offs, paths = _three_offset_panels(tmp_path)
    out = _blend_many(paths, level_solve="overlap")
    regions = (np.s_[150:210, 20:150], np.s_[150:210, 240:320],
               np.s_[30:210, 420:540])
    resid = [float(np.median((out - field)[r])) for r in regions]
    assert max(resid) - min(resid) < 2e-4, resid
    assert abs(float(np.mean(resid))) < 2e-4  # sky back on the pedestal


def test_single_panel_passthrough_unchanged(tmp_path):
    """The 1-panel contract survives the new mode."""
    canvas = (60, 80)
    img = _panel(_sky_field(canvas), 10, 50, 10, 70, 3e-4)
    p = _write(tmp_path, "solo", img)
    out = _blend_many([p], level_solve="overlap")
    assert np.allclose(out, img)


def test_no_overlap_falls_back_to_zero_offsets(tmp_path):
    """Disjoint panels: nothing to solve; global pedestal shift only."""
    canvas = (100, 300)
    f = _sky_field(canvas)
    a = _panel(f, 10, 90, 0, 120, 6e-4)
    b = _panel(f, 10, 90, 180, 300, -6e-4)
    out = _blend_many([_write(tmp_path, "a", a), _write(tmp_path, "b", b)],
                      level_solve="overlap")
    # both panels present, both positive where data
    assert (out[50, 60] > 0) and (out[50, 240] > 0)


def test_level_solve_none_leaves_panels_untouched(tmp_path):
    """`level_solve=None` (the STARS-layer mode) must apply NO shifts at
    all: exclusive regions keep their injected offsets verbatim — a stars
    layer's panels already agree by construction (`channel − starless`
    cancels additive terms), and any equalization would inject lifts."""
    field, offs, paths = _three_offset_panels(tmp_path)
    out = _blend_many(paths, level_solve=None)
    regions = (np.s_[150:210, 20:150], np.s_[150:210, 240:320],
               np.s_[30:210, 420:540])
    resid = [float(np.median((out - field)[r])) for r in regions]
    # Each exclusive region shows exactly its own panel's injected offset.
    assert np.max(np.abs(np.array(resid) - np.array(offs))) < 2e-4, resid


def test_unknown_level_solve_mode_raises(tmp_path):
    canvas = (100, 300)
    f = _sky_field(canvas)
    a = _panel(f, 10, 90, 0, 200, 0.0)
    b = _panel(f, 10, 90, 100, 300, 0.0)
    with pytest.raises(AssertionError, match="level_solve"):
        _blend_many([_write(tmp_path, "a", a), _write(tmp_path, "b", b)],
                    level_solve="typo")
