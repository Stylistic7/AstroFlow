"""`_blend_many` bbox rewrite is bit-identical to the full-canvas original.

The bbox version processes each panel in its on-disk bounding box (mask, EDT
weight, and accumulation slice all bbox-sized) instead of over the full mosaic
canvas, to cut peak RAM from ~5-6 full-canvas float32 arrays to ~3. The whole
justification is that this changes nothing numerically — the skipped
out-of-bbox terms are exactly +0 and `_edt_weight` reproduces the canvas EDT —
so these tests pin exact equality against a reference copy of the original,
across the two EDT-padding branches (a rotated rect with zero bbox corners,
interior to the canvas; and a rect flush against the canvas edge).
"""
import numpy as np
import edt

from mosaics.merging.oss import _blend_many, _panel_bbox
from mosaics.placement.panel_io import write_placed_panel, read_placed_panel


def _read_full_canvas(p):
    """Reconstruct the full-canvas array a placed panel represents — what the
    pre-bbox `_blend_many` read directly off disk."""
    arr, (x0, y0), (cw, ch) = read_placed_panel(p)
    full = np.zeros((ch, cw), np.float32)
    full[y0:y0 + arr.shape[0], x0:x0 + arr.shape[1]] = arr
    return full


def _blend_many_reference(paths):
    """The pre-bbox full-canvas implementation, verbatim, as the golden ref —
    now reading the full canvas back from the bbox-on-disk panels.

    Pure blend math, no level matching on either side (`level_solve=None`
    default since 2026-09-02): the equivalence these tests pin is "bbox
    math == full-canvas math", and the matching modes are orthogonal
    pre-steps with their own lab (test_level_solve.py)."""
    if len(paths) == 1:
        return _read_full_canvas(paths[0])
    canvases = [_read_full_canvas(p) for p in paths]
    # (No flux solve: gains are calibrated upstream by post_stack_calibrate —
    # the overlap flux solve was deleted 2026-09-01.)
    masks = [img > 0 for img in canvases]
    numer = np.zeros(masks[0].shape, np.float32)
    denom = np.zeros(masks[0].shape, np.float32)
    for i, img in enumerate(canvases):
        weight = edt.edt(masks[i].astype(np.uint8),
                         black_border=False, parallel=0).astype(np.float32)
        numer += weight * img
        denom += weight
    denom[denom == 0] = 1.0
    return numer / denom


def _write(tmp_path, name, img):
    """Persist a full-canvas panel as bbox-on-disk — place_image's format: the
    tight non-zero bbox pixels + their canvas offset in the header."""
    p = tmp_path / f"{name}.tiff"
    r0, r1, c0, c1 = _panel_bbox(img > 0)
    write_placed_panel(
        p, img[r0:r1, c0:c1].astype(np.float32),
        offset_xy=(c0, r0), canvas_size=(img.shape[1], img.shape[0]),
    )
    return p


def _rot_rect(H, W, cy, cx, hy, hx, angle_deg, base, rng):
    """Filled rotated rectangle of strictly-positive values on a zero canvas —
    its axis-aligned bbox has zero corners, exercising the padded EDT branch."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = np.deg2rad(angle_deg)
    u = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
    v = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
    m = (np.abs(u) <= hx) & (np.abs(v) <= hy)
    img = np.zeros((H, W), np.float32)
    img[m] = base + rng.random(int(m.sum())).astype(np.float32)
    return img


def test_blend_bbox_bit_identical(tmp_path):
    H, W = 400, 500
    rng = np.random.default_rng(0)
    paths = [
        _write(tmp_path, "p0", _rot_rect(H, W, 180, 200, 90, 130, 25, 10.0, rng)),
    ]
    edge = np.zeros((H, W), np.float32)          # flush to top-left canvas edge
    edge[0:150, 0:180] = 5.0 + rng.random((150, 180)).astype(np.float32)
    paths.append(_write(tmp_path, "p1", edge))
    paths.append(
        _write(tmp_path, "p2", _rot_rect(H, W, 230, 300, 100, 120, -35, 20.0, rng))
    )
    ov = np.zeros((H, W), np.float32)            # overlaps p1 and p2
    ov[120:320, 150:360] = 8.0 + rng.random((200, 210)).astype(np.float32)
    paths.append(_write(tmp_path, "p3", ov))

    new = _blend_many(paths)
    ref = _blend_many_reference(paths)
    assert new.shape == ref.shape == (H, W)
    assert np.array_equal(new, ref)


def test_blend_disjoint_panels(tmp_path):
    H, W = 200, 300
    a = np.zeros((H, W), np.float32); a[10:60, 10:80] = 3.0
    b = np.zeros((H, W), np.float32); b[120:180, 200:280] = 7.0
    pa, pb = _write(tmp_path, "a", a), _write(tmp_path, "b", b)
    assert np.array_equal(_blend_many([pa, pb]), _blend_many_reference([pa, pb]))


def test_blend_single_panel_passthrough(tmp_path):
    img = np.zeros((100, 120), np.float32); img[10:90, 20:100] = 4.0
    p = _write(tmp_path, "solo", img)
    assert np.array_equal(_blend_many([p]), _read_full_canvas(p))


def test_merge_writes_the_physical_layer_into_the_header(tmp_path):
    """A merged mosaic must SAY which layer it is. Layer identity used to be
    legible only from the op's output directory plus SKYVER ("overlap-2" vs
    "stars-none-1"), so a starless and a stars mosaic lifted out of the tree
    were indistinguishable — and they are the two halves of every view."""
    from types import SimpleNamespace

    from file_io.tiff_file_io import read_tiff_header
    from mosaics.merging.oss import _merge_layer

    img = np.zeros((60, 80), np.float32)
    img[10:50, 10:70] = 3.0
    src = _write(tmp_path, "panel", img)
    inp = SimpleNamespace(filename=src, stable_hash="abc")
    out = tmp_path / "merged.tiff"
    op = SimpleNamespace(filename_fn=lambda *a: str(out))

    _merge_layer(op, None, "Ha", (inp,), level_solve=None, skyver="v",
                 layer="stars")
    assert read_tiff_header(out)["LAYER"] == "stars"


def test_the_two_merge_ops_declare_opposite_layers(monkeypatch):
    """The shared `_merge_layer` body stamps whatever it is handed, so the
    thing worth pinning is that the two callers hand it opposite values."""
    import inspect

    from mosaics.merging import oss

    seen = []
    monkeypatch.setattr(
        oss, "_merge_layer",
        lambda op, mosaic, filter, inputs, **kw: seen.append(kw["layer"]))
    for op in (oss.merge_images_oss, oss.merge_images_stars):
        # past beartype: the bodies only forward, so a real Mosaic is noise
        inspect.unwrap(op._function)(None, "Ha")
    assert seen == ["starless", "stars"]


def _noisy_canvas(rng, H, W, base, sigma, margin=20):
    """A panel filling the canvas but for a zero `margin` (a placed panel
    always sits inside its canvas; a footprint with no background has an
    infinite distance weight): flat `base` plus Gaussian noise `sigma`,
    clipped strictly positive so the footprint mask holds."""
    img = np.zeros((H, W), np.float32)
    img[margin:-margin, margin:-margin] = np.maximum(
        base + rng.normal(0.0, sigma, (H - 2 * margin, W - 2 * margin)), 1e-3)
    return img


def test_noise_sigma_measures_gaussian_noise(tmp_path):
    from mosaics.merging.oss import _noise_sigma
    rng = np.random.default_rng(1)
    img = _noisy_canvas(rng, 400, 600, 1.0, 0.02)
    assert abs(_noise_sigma(img, img > 0) / 0.02 - 1.0) < 0.05


def test_depth_weighting_makes_coincident_panels_deeper(tmp_path):
    """Two framings of one field, sigma and 3 sigma. Geometry-only weights
    average them 50/50 — noisier than the deep panel alone; inverse-variance
    weights give sqrt(0.9) of its noise. Equal panels come out sqrt(2) deeper."""
    rng = np.random.default_rng(2)
    H, W, base, sigma = 400, 600, 1.0, 0.02
    deep = _write(tmp_path, "deep", _noisy_canvas(rng, H, W, base, sigma))
    shallow = _write(tmp_path, "shallow", _noisy_canvas(rng, H, W, base, 3 * sigma))
    twin = _write(tmp_path, "twin", _noisy_canvas(rng, H, W, base, sigma))
    interior = (slice(40, H - 40), slice(40, W - 40))

    def noise(out):
        return float(np.std(out[interior] - base))

    assert noise(_blend_many([deep, shallow])) > 1.4 * sigma              # the old 50/50: sqrt(10)/2 = 1.58
    assert 0.88 * sigma < noise(_blend_many([deep, shallow], depth_weighted=True)) < 0.98 * sigma
    assert abs(noise(_blend_many([deep, twin], depth_weighted=True)) / (sigma / np.sqrt(2)) - 1.0) < 0.1


if __name__ == "__main__":
    import tempfile
    from pathlib import Path as _P
    with tempfile.TemporaryDirectory() as d:
        test_noise_sigma_measures_gaussian_noise(_P(d))
        print("ok test_noise_sigma_measures_gaussian_noise")
        test_depth_weighting_makes_coincident_panels_deeper(_P(d))
        print("ok test_depth_weighting_makes_coincident_panels_deeper")
