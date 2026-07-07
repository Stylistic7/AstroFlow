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
import astropy.io.fits as pyfits
import edt

from mosaics.merging.oss import _blend_many, _solve_flux_scales
from file_io.generic import read_image


def _blend_many_reference(paths):
    """The pre-bbox full-canvas implementation, verbatim, as the golden ref."""
    if len(paths) == 1:
        return read_image(paths[0]).astype(np.float32)
    n = len(paths)
    masks, overlap_means = [], {}
    for i, p in enumerate(paths):
        img = read_image(p).astype(np.float32)
        mask = img > 0
        for j, mask_j in enumerate(masks):
            both = mask & mask_j
            if both.any():
                overlap_means[(j, i)] = (None, float(img[both].mean()))
        masks.append(mask)
    for j, p in enumerate(paths):
        img = read_image(p).astype(np.float32)
        for (a, b), (_, mi) in list(overlap_means.items()):
            if a != j:
                continue
            both = masks[a] & masks[b]
            mj = float(img[both].mean()) if both.any() else 0.0
            overlap_means[(a, b)] = (mj, mi)
    scales = _solve_flux_scales(n, overlap_means)
    numer = np.zeros(masks[0].shape, np.float32)
    denom = np.zeros(masks[0].shape, np.float32)
    for i, p in enumerate(paths):
        img = read_image(p).astype(np.float32)
        weight = edt.edt(masks[i].astype(np.uint8),
                         black_border=False, parallel=0).astype(np.float32)
        numer += weight * (img * scales[i])
        denom += weight
    denom[denom == 0] = 1.0
    return numer / denom


def _write(tmp_path, name, img):
    p = tmp_path / f"{name}.fits"
    pyfits.writeto(p, img)
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
    assert np.array_equal(_blend_many([p]), read_image(p).astype(np.float32))
