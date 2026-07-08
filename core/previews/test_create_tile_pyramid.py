import struct
from pathlib import Path

import brotli
import numpy as np
import cv2
import pyvips
import astropy.io.fits as pyfits

from file_io.tiff_file_io import write_tiff_image
from previews.create_tile_pyramid import (
    _CONTEXT_MAGIC,
    _CONTEXT_RATIO,
    _MIN_CONTEXT_DIM,
    _TILE_SIZE,
    context_map_path_for,
    default_make_context_map,
    default_make_tile_pyramid,
    pyramid_thumbnail,
)


def _write_fits(path, arr):
    pyfits.writeto(str(path), arr.astype(np.float32), overwrite=True)


def _png_min_dim(png: bytes) -> int:
    arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    h, w = arr.shape[:2]
    return min(h, w)


def _synthetic(tmp_path, side=4096):
    y, x = np.mgrid[0:side, 0:side]
    arr = ((x + y) % 1000).astype(np.float32)
    src = tmp_path / "src.fits"
    dst = tmp_path / "src.tiff"
    _write_fits(src, arr)
    default_make_tile_pyramid(src, dst)
    return src, dst


def test_tile_pyramid_is_multiresolution(tmp_path):
    _src, dst = _synthetic(tmp_path)
    assert dst.exists()
    # A genuine pyramid: multiple stored overview levels, not one IFD.
    assert pyvips.Image.tiffload(str(dst), page=0).get("n-pages") > 1


def test_pyramid_thumbnail_picks_smallest_level_ge_min(tmp_path):
    _src, dst = _synthetic(tmp_path, side=4096)
    # Levels are 4096, 2048, 1024, ... so the smallest level whose min
    # dimension is >= 1200 is 2048 (not the ~256px top level).
    assert _png_min_dim(pyramid_thumbnail(dst, min_px=1200)) == 2048


def test_pyramid_thumbnail_never_reads_source(tmp_path):
    src, dst = _synthetic(tmp_path)
    src.unlink()  # the source is gone; serving must only touch the .tiff
    png = pyramid_thumbnail(dst, min_px=1200)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_pyramid_thumbnail_handles_legacy_single_resolution_tiff(tmp_path):
    # Tiffs written before this change are single-resolution tiled tiffs
    # (no overviews). Serving must still work — via shrink-on-load — and
    # must not blow up reading the whole thing.
    arr = (np.mgrid[0:4096, 0:4096][0] % 256).astype(np.uint8)
    legacy = tmp_path / "legacy.tiff"
    write_tiff_image(legacy, arr, tile=(256, 256))
    assert pyvips.Image.tiffload(str(legacy), page=0).get("n-pages") == 1
    png = pyramid_thumbnail(legacy, min_px=1200)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    arr2 = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    assert 0 < min(arr2.shape[:2]) <= 1200


def test_pyramid_thumbnail_autostretches_float32_so_data_isnt_binary(tmp_path):
    """Regression: place_image + merge_images_subframe write raw float32
    TIFFs with pixel values up to ~65535. Without autostretching first,
    pyvips' default linear-to-uint8 scale clips every data pixel above
    255 to 255 — every "active" pixel becomes pure white, producing the
    white-on-black binary look the user reported.

    Verify a float32 source with a realistic raw-frame distribution
    (mostly-zero background + data pixels around 500-1000 with stars up
    to 65535) renders as a real grayscale, not a binary mask.
    """
    side = 1024
    arr = np.zeros((side, side), dtype=np.float32)
    yy, xx = np.ogrid[:side, :side]
    mask = (yy - side // 2) ** 2 + (xx - side // 2) ** 2 < (side // 2 - 30) ** 2
    rng = np.random.default_rng(0)
    arr[mask] = 800 + rng.normal(0, 40, size=mask.sum()).astype(np.float32)
    # A couple of bright stars to give the stretch real headroom.
    for _ in range(5):
        cy = int(rng.integers(side // 4, 3 * side // 4))
        cx = int(rng.integers(side // 4, 3 * side // 4))
        arr[cy - 3:cy + 4, cx - 3:cx + 4] = 65535
    src = tmp_path / "raw_float.tiff"
    write_tiff_image(src, arr)

    png = pyramid_thumbnail(src, min_px=800)
    decoded = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)

    # The bug signature: virtually every data pixel ends up at uint8 max.
    # The fix: data pixels spread across many uint8 values.
    data_pixels = decoded[decoded > 0]
    assert len(data_pixels) > 0, "thumbnail has no non-zero pixels at all"
    unique = len(np.unique(data_pixels))
    assert unique >= 20, (
        f"expected >=20 unique non-zero values in thumbnail, got {unique} "
        f"— autostretch likely didn't run, image is effectively binary"
    )
    frac_at_max = float(np.mean(data_pixels == 255))
    assert frac_at_max < 0.5, (
        f"{frac_at_max:.1%} of non-zero pixels saturated to 255 "
        f"(the binary-mask symptom)"
    )


# ---------------------------------------------------------------------------
# Linear-float pyramid + context map tests for the new architecture.
#
# Canonical contract these defend:
#   * Pyramid stores LINEAR pixel data (no autostretch); the WebGL shader
#     applies the stretch at display time.
#   * Float32 dtype is preserved through the pyramid round-trip.
#   * Context map dimensions are EXACTLY source/5 on each axis.
#   * Sources too small for a useful context map skip the file.
#   * Context map's brotli payload decodes to magic + dims + float16 pixels.
#   * Lanczos3 downsample is smooth (linear ramp stays linear).
#   * Determinism: same input twice → byte-identical output.
# ---------------------------------------------------------------------------


def _write_linear_master(
    tmp_path, name, shape, fill=None, pattern=None,
) -> Path:
    """Write a linear float32 TIFF the producer would have produced."""
    h, w = shape
    if fill is not None:
        arr = np.full((h, w), fill, dtype=np.float32)
    elif pattern == "horizontal_ramp":
        arr = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    elif pattern == "gauss":
        ys = np.linspace(-2, 2, h)[:, None]
        xs = np.linspace(-2, 2, w)[None, :]
        arr = np.exp(-(xs**2 + ys**2)).astype(np.float32)
    else:
        rng = np.random.default_rng(seed=0xCAFE)
        arr = rng.uniform(0.02, 0.05, size=(h, w)).astype(np.float32)
        for x, y in [(w // 4, h // 3), (w // 2, h // 2), (3 * w // 4, 2 * h // 3)]:
            arr[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = 0.85
    img = pyvips.Image.new_from_memory(arr.tobytes(), w, h, 1, "float")
    path = tmp_path / name
    img.tiffsave(str(path), compression="deflate", bigtiff=True)
    return path


def _read_pyramid_level(pyramid_path: Path, level: int) -> np.ndarray:
    img = pyvips.Image.tiffload(str(pyramid_path), page=level)
    raw = img.write_to_memory()
    bands = img.bands
    if bands == 1:
        return np.frombuffer(raw, dtype=np.float32).reshape(img.height, img.width)
    return np.frombuffer(raw, dtype=np.float32).reshape(img.height, img.width, bands)


def _decode_context_map(context_path: Path) -> tuple[int, int, np.ndarray]:
    raw = context_path.read_bytes()
    payload = brotli.decompress(raw)
    magic = payload[:4]
    assert magic == _CONTEXT_MAGIC, (
        f"context map magic header mismatch: got {magic!r}"
    )
    width, height = struct.unpack("<II", payload[4:12])
    pixels = np.frombuffer(payload[12:], dtype=np.float16)
    assert pixels.size == width * height
    return width, height, pixels.reshape(height, width)


def test_pyramid_preserves_linear_float32_pixels(tmp_path):
    """Linear input reads back as linear output at level 0. The
    headline contract: this pyramid does NOT autostretch."""
    source = _write_linear_master(tmp_path, "in.tiff", (720, 1080), fill=0.04)
    dest = tmp_path / "out.tiff"
    default_make_tile_pyramid(source, dest)

    level0 = _read_pyramid_level(dest, 0)
    assert level0.dtype == np.float32, (
        f"level 0 dtype = {level0.dtype}, expected float32 "
        f"(autostretch quantization to uint8 would show here)"
    )
    assert np.isclose(level0.mean(), 0.04, atol=1e-5), (
        f"flat-input level 0 mean = {level0.mean()}, expected 0.04. "
        f"A non-trivial difference means autostretch is still running."
    )


def test_pyramid_has_multiple_levels_on_a_large_input(tmp_path):
    """720×1080 with 256-pixel tiles should give at least 2 pyramid levels."""
    source = _write_linear_master(tmp_path, "in.tiff", (720, 1080), fill=0.1)
    dest = tmp_path / "out.tiff"
    default_make_tile_pyramid(source, dest)

    base = pyvips.Image.tiffload(str(dest), page=0)
    assert base.get("n-pages") >= 2


def test_pyramid_levels_are_factor_two(tmp_path):
    """Each level is half the previous level's dimensions (±1 px)."""
    source = _write_linear_master(tmp_path, "in.tiff", (720, 1080), fill=0.05)
    dest = tmp_path / "out.tiff"
    default_make_tile_pyramid(source, dest)

    base = pyvips.Image.tiffload(str(dest), page=0)
    n_pages = base.get("n-pages")
    prev = (base.width, base.height)
    for lvl in range(1, n_pages):
        img = pyvips.Image.tiffload(str(dest), page=lvl)
        w, h = img.width, img.height
        expected = (prev[0] // 2, prev[1] // 2)
        assert abs(w - expected[0]) <= 1
        assert abs(h - expected[1]) <= 1
        prev = (w, h)


def test_pyramid_is_deterministic(tmp_path):
    """Same input twice → byte-identical level 0 pixel data."""
    source = _write_linear_master(
        tmp_path, "in.tiff", (512, 768), pattern="gauss",
    )
    out_a = tmp_path / "a.tiff"
    out_b = tmp_path / "b.tiff"
    default_make_tile_pyramid(source, out_a)
    default_make_tile_pyramid(source, out_b)

    n_pages = pyvips.Image.tiffload(str(out_a), page=0).get("n-pages")
    for lvl in range(n_pages):
        a = _read_pyramid_level(out_a, lvl)
        b = _read_pyramid_level(out_b, lvl)
        assert np.array_equal(a, b), (
            f"pyramid level {lvl} differs between runs"
        )


def test_pyramid_preserves_negative_values(tmp_path):
    """Post-pedestal masters can have small negative excursions in
    heavy-noise regions; the pyramid must round-trip them without
    clipping or sign-flipping."""
    h, w = 480, 640
    arr = np.full((h, w), 0.05, dtype=np.float32)
    arr[100:110, 100:110] = -0.01
    img = pyvips.Image.new_from_memory(arr.tobytes(), w, h, 1, "float")
    source = tmp_path / "negs.tiff"
    img.tiffsave(str(source), compression="deflate")

    dest = tmp_path / "negs_pyramid.tiff"
    default_make_tile_pyramid(source, dest)

    level0 = _read_pyramid_level(dest, 0)
    assert level0.min() < 0, (
        f"negative values clipped on round-trip: min={level0.min()}"
    )


def test_context_map_exact_one_to_five_ratio(tmp_path):
    """For a 1500×3000 source the context map is exactly 300×600."""
    source = _write_linear_master(
        tmp_path, "src.tiff", (1500, 3000), pattern="horizontal_ramp",
    )
    dest = tmp_path / "out.context"
    wrote = default_make_context_map(source, dest)
    assert wrote

    w, h, arr = _decode_context_map(dest)
    assert (w, h) == (3000 // _CONTEXT_RATIO, 1500 // _CONTEXT_RATIO)


def test_context_map_skipped_for_small_sources(tmp_path):
    """720×1080 → 144×216 context map, below the 256-px threshold; skip."""
    source = _write_linear_master(tmp_path, "small.tiff", (720, 1080), fill=0.1)
    dest = tmp_path / "small.context"
    wrote = default_make_context_map(source, dest)
    assert not wrote
    assert not dest.exists()


def test_context_map_threshold_boundary(tmp_path):
    """A source exactly at the threshold produces a context map."""
    short = _CONTEXT_RATIO * _MIN_CONTEXT_DIM
    source = _write_linear_master(
        tmp_path, "boundary.tiff", (short, short * 2), fill=0.1,
    )
    dest = tmp_path / "boundary.context"
    assert default_make_context_map(source, dest) is True


def test_context_map_dtype_is_float16(tmp_path):
    """float16 round-trip preserves a 0.04 fill within float16 precision."""
    source = _write_linear_master(tmp_path, "src.tiff", (1500, 1500), fill=0.04)
    dest = tmp_path / "src.context"
    default_make_context_map(source, dest)

    w, h, arr = _decode_context_map(dest)
    assert arr.dtype == np.float16
    assert np.allclose(arr, 0.04, atol=1e-3)


def test_context_map_lanczos_quality_on_smooth_ramp(tmp_path):
    """Horizontal ramp downsamples to a smooth ramp; Lanczos's second-
    derivative variance is tiny vs box-averaging which would stairstep."""
    source = _write_linear_master(
        tmp_path, "ramp.tiff", (1500, 1500), pattern="horizontal_ramp",
    )
    dest = tmp_path / "ramp.context"
    default_make_context_map(source, dest)

    w, h, arr = _decode_context_map(dest)
    slice_ = arr[h // 2, :].astype(np.float32)
    second_deriv = np.diff(slice_, n=2)
    rms = float(np.sqrt(np.mean(second_deriv ** 2)))
    assert rms < 1e-3, (
        f"Lanczos3 downsample of a ramp produced second-derivative RMS={rms}; "
        f"should be ≪ 1e-3 (box-average would stairstep)"
    )


def test_context_map_brotli_compressed_at_rest(tmp_path):
    """Raw on-disk bytes are brotli-compressed; only after decompression
    do we see the magic header."""
    source = _write_linear_master(
        tmp_path, "src.tiff", (1500, 1500), pattern="gauss",
    )
    dest = tmp_path / "src.context"
    default_make_context_map(source, dest)

    raw = dest.read_bytes()
    decompressed = brotli.decompress(raw)
    assert raw[:4] != _CONTEXT_MAGIC, (
        "raw file starts with the magic tag — brotli compression didn't run"
    )
    assert decompressed[:4] == _CONTEXT_MAGIC


def test_context_map_path_convention():
    """`<master>.tiff` → `<master>.context`."""
    assert context_map_path_for(Path("/tmp/x/master.tiff")) == Path("/tmp/x/master.context")


def test_context_map_is_deterministic(tmp_path):
    source = _write_linear_master(
        tmp_path, "src.tiff", (1500, 1500), pattern="gauss",
    )
    a = tmp_path / "a.context"
    b = tmp_path / "b.context"
    default_make_context_map(source, a)
    default_make_context_map(source, b)
    assert a.read_bytes() == b.read_bytes()
