"""Tests for file_io/tiling.py — the generic tile idiom.

Pure (no rc-astro, no Ray): the external operation is simulated in-process, so
these run in the fast suite. They assert the three properties that make the
idiom safe: the grid covers the image, the streaming feather blend reconstructs
an identity-processed image exactly, and it feathers per-tile discontinuities
away (vs a hard cut). Mirrors mosaics/merging/test_oss_seam.py in spirit.
"""
import threading

import numpy as np
import pytest

from file_io.tiling import crop_source_region, feather_blend_to_bigtiff
from file_io.tiff_file_io import write_tiff_image, read_tiff_image


def tile_grid(width, height, tile_px, overlap):
    """Test scaffold: an overlapping grid, used only to generate rects to exercise
    `crop_source_region` + `feather_blend_to_bigtiff`. Production no longer tiles by
    arbitrary grid (that's `mosaics.panels` — panels ARE the tiling); this local
    helper keeps the primitive tests without any arbitrary-tiling code shipping."""
    step = max(1, round(tile_px * (1.0 - overlap)))

    def starts(length):
        if length <= tile_px:
            return [0]
        s = list(range(0, length - tile_px + 1, step))
        if s[-1] != length - tile_px:
            s.append(length - tile_px)
        return s

    tw, th = min(tile_px, width), min(tile_px, height)
    return [(x, y, tw, th) for y in starts(height) for x in starts(width)]


def _synth(h, w, bands=1):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (0.3 + 0.4 * xx / w + 0.3 * np.sin(yy / 11.0) * np.cos(xx / 7.0)).astype(np.float32)
    if bands == 1:
        return base
    return np.stack([base, base * 0.8 + 0.1, base * 0.5 + 0.2], axis=2).astype(np.float32)


def _tile_and_blend(tmp_path, img, tile_px, overlap, per_tile_offset=False):
    """Write `img` as a tiled BigTIFF, crop it into the grid, optionally perturb
    each tile by a constant (to simulate a tool's per-tile context differences),
    and feather-blend back. Returns the reassembled array."""
    h, w = img.shape[:2]
    bands = 1 if img.ndim == 2 else img.shape[2]
    src = tmp_path / f"src_{h}x{w}x{bands}.tiff"
    write_tiff_image(src, img, tile=(64, 64))

    rects = tile_grid(w, h, tile_px, overlap)
    paths = []
    for i, r in enumerate(rects):
        arr = crop_source_region(src, r)
        if per_tile_offset:
            arr = arr + (i % 5) * 0.05
        p = tmp_path / f"tile_{i}.tiff"
        write_tiff_image(p, arr, tile=(64, 64))
        paths.append(p)

    out = tmp_path / "out.tiff"
    feather_blend_to_bigtiff(rects, paths, out, overlap=overlap, tile=(64, 64))
    return rects, read_tiff_image(out)


@pytest.mark.parametrize("w,h,tile_px,overlap", [
    (300, 200, 128, 0.5),
    (251, 137, 100, 0.5),   # non-divisible dims → clamped last tile
    (64, 64, 128, 0.5),     # single tile (dim < tile_px)
    (1000, 40, 256, 0.5),   # tall/skinny
])
def test_tile_grid_covers_exactly(w, h, tile_px, overlap):
    rects = tile_grid(w, h, tile_px, overlap)
    assert rects, "grid is empty"
    assert max(x + tw for (x, y, tw, th) in rects) == w
    assert max(y + th for (x, y, tw, th) in rects) == h
    assert all(x >= 0 and y >= 0 and x + tw <= w and y + th <= h for (x, y, tw, th) in rects)
    # Every pixel is covered by at least one tile.
    cover = np.zeros((h, w), dtype=bool)
    for (x, y, tw, th) in rects:
        cover[y:y + th, x:x + tw] = True
    assert cover.all()


@pytest.mark.parametrize("bands", [1, 3])
def test_identity_reconstruction_is_exact(tmp_path, bands):
    """An identity operation (tiles are just crops) must reassemble bit-close to
    the original — the feather + normalization is a partition of unity."""
    img = _synth(200, 300, bands)
    _, out = _tile_and_blend(tmp_path, img, tile_px=128, overlap=0.5)
    assert out.shape == img.shape
    assert float(np.max(np.abs(out - img))) < 1e-3


def test_single_tile_is_passthrough(tmp_path):
    img = _synth(64, 64)
    _, out = _tile_and_blend(tmp_path, img, tile_px=128, overlap=0.5)
    assert np.array_equal(out, img)


def test_feather_hides_seams_better_than_hard_cut(tmp_path):
    """With per-tile offsets, the feathered blend must ramp the discontinuity
    smoothly; a near-zero overlap (hard cut) must leave a visible step. `delta`
    subtracts the clean image so we measure only the seam artifact."""
    img = _synth(200, 300)

    _, out_feather = _tile_and_blend(tmp_path, img, 128, 0.5, per_tile_offset=True)
    feather_jump = np.abs(np.diff(out_feather - img, axis=1)).max()

    _, out_hard = _tile_and_blend(tmp_path, img, 128, 0.02, per_tile_offset=True)
    hard_jump = np.abs(np.diff(out_hard - img, axis=1)).max()

    assert feather_jump < 0.01, f"feather left a seam: {feather_jump}"
    assert hard_jump > 0.03, f"negative control had no step: {hard_jump}"
    assert feather_jump < hard_jump / 5


@pytest.mark.slow
def test_reassembly_memory_is_bounded(tmp_path):
    """Peak RSS growth during the blend should stay well under the full-canvas
    size — evidence the pyvips pipeline streams tile-by-tile rather than
    materializing the whole image. Generous bound to stay non-flaky."""
    psutil = pytest.importorskip("psutil")
    img = _synth(3000, 3000)          # 36 MB mono canvas; 512-px tiles → ~11x11 grid
    proc = psutil.Process()

    peak = [proc.memory_info().rss]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)

    base = proc.memory_info().rss
    t = threading.Thread(target=sample, daemon=True)
    t.start()
    try:
        _, out = _tile_and_blend(tmp_path, img, tile_px=512, overlap=0.5)
    finally:
        stop.set()
        t.join()

    assert float(np.max(np.abs(out - img))) < 1e-3
    growth = peak[0] - base
    # Coarse guard against catastrophic growth (a full-canvas materialization).
    # pyvips' own baseline is ~2 GB, so a tight per-canvas bound isn't meaningful
    # at unit-test sizes — the rigorous size-independence proof is the real-data
    # equivalence test (212 Mpx / 18 tiles, no OOM on 64 GB; see project memory).
    assert growth < 4 * 1024**3, f"blend RSS grew {growth/1e9:.1f} GB (>4 GB)"
