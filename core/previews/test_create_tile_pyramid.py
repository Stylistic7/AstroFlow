import numpy as np
import cv2
import pyvips
import tifffile
import astropy.io.fits as pyfits

from previews.create_tile_pyramid import default_make_tile_pyramid, pyramid_thumbnail


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
    tifffile.imwrite(str(legacy), arr, tile=(256, 256), compression="deflate")
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
    tifffile.imwrite(str(src), arr, compression="deflate")

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
