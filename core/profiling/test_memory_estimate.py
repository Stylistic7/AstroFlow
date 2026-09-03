"""`_pixel_count` header-peeking for filename args.

`compute_image_metadata`'s only sizing arg is a filename, so without this the
estimator can't tell a 50 MB subframe from a 9 GB gigapixel master (both bucket
"unknown") and can't right-size its auto_memory reservation. These tests pin
that a bare path arg is sized from its on-disk header — for every format, from
the decompressed pixel count (not the file's byte size) — while non-path args
and dimensioned objects behave exactly as before.
"""
import numpy as np
import astropy.io.fits as pyfits

from profiling.memory_estimate import (
    _pixel_count, estimate_memory, _FIXED_MEMORY_BY_OP, _max_memory_bytes,
)
from file_io.tiff_file_io import write_tiff_image
from file_io.xisf_file_io import write_xisf_image


def test_reads_fits_header(tmp_path):
    p = tmp_path / "frame.fits"
    pyfits.writeto(p, np.zeros((100, 200), np.float32))     # (H, W)
    assert _pixel_count((p,), {}) == 100 * 200


def test_reads_tiff_header_decompressed_count(tmp_path):
    p = tmp_path / "panel.tiff"
    # write_tiff_image uses deflate — a near-empty array is tiny on disk, so a
    # file-size proxy would badly under-count. The header gives the true dims.
    write_tiff_image(p, np.zeros((120, 210), np.float32))
    assert _pixel_count((p,), {}) == 120 * 210


def test_reads_xisf_header(tmp_path):
    p = tmp_path / "master.xisf"
    write_xisf_image(p, np.zeros((130, 220, 1), np.float32))
    assert _pixel_count((p,), {}) == 130 * 220


def test_counts_channels_fits(tmp_path):
    p = tmp_path / "rgb.fits"
    pyfits.writeto(p, np.zeros((3, 100, 200), np.float32))   # (C, H, W)
    assert _pixel_count((p,), {}) == 3 * 100 * 200           # not just H*W


def test_counts_channels_tiff(tmp_path):
    p = tmp_path / "rgb.tiff"
    write_tiff_image(p, np.zeros((120, 210, 3), np.float32))  # (H, W, C)
    assert _pixel_count((p,), {}) == 120 * 210 * 3


def test_dimensioned_object_counts_channels():
    class FakeRGB:
        width_pixels = 10
        height_pixels = 20
        channels = 3
    assert _pixel_count((FakeRGB(),), {}) == 10 * 20 * 3


def test_non_image_strings_do_no_io_and_count_zero():
    # filter name, content hash, plate_solve flag — none are image paths.
    assert _pixel_count(("H", "deadbeef1234", None, True), {}) == 0


def test_missing_file_is_zero(tmp_path):
    assert _pixel_count((tmp_path / "nope.fits",), {}) == 0


def test_dimensioned_object_wins_without_disk_read():
    class FakeMeta:            # ImageMetadata-like
        width_pixels = 10
        height_pixels = 20
    assert _pixel_count((FakeMeta(),), {}) == 200


def test_fixed_memory_ops_reserve_their_fixed_budget():
    # Ops in _FIXED_MEMORY_BY_OP bypass profiling entirely: bxt_channel /
    # sxt_channel shell out to an rc-astro CLI child the worker-RSS profiler
    # can't see, and
    # compute_image_metadata is a roughly-constant op dispatched ~30k times that
    # we keep on a flat reservation. Each must reserve its fixed budget
    # regardless of profile samples / input pixel count.
    cap = _max_memory_bytes()
    for name, want in _FIXED_MEMORY_BY_OP.items():
        f = lambda: None
        f.__name__ = name
        # a huge fake input must NOT change the reservation (fixed override)
        assert estimate_memory(f, (object(),), {}) == min(want, cap), name


def test_sums_path_plus_output_viewbox(tmp_path):
    p = tmp_path / "src.fits"
    pyfits.writeto(p, np.zeros((50, 60), np.float32))

    class FakeViewbox:         # output canvas
        num_pixels_x = 1000
        num_pixels_y = 2000
    assert _pixel_count((p, FakeViewbox()), {}) == 50 * 60 + 1000 * 2000


# --- max_calls=1 wiring (Change 1): profiled auto_memory ops are isolated in a
# fresh worker so their absolute-RSS profile sample reflects only that op. These
# assert the baked Ray option; the actual isolation is proven in
# profiling/test_worker_isolation.py. `-> int` returns keep decoration off the
# artifact-producer/registry path, so no Ray/context is needed here.

def _baked_max_calls(op):
    # The @node wrapper exposes the underlying Ray RemoteFunction via `.options`
    # (a bound method); `_default_options["max_calls"]` is the baked value.
    # None => unset => Ray's default worker reuse.
    return op.options.__self__._default_options.get("max_calls")


def test_profiled_auto_memory_op_gets_max_calls_1():
    from decorators import node
    @node(auto_memory=True)
    def place_image(x) -> int:        # name NOT in _FIXED_MEMORY_BY_OP -> profiled
        return x
    assert _baked_max_calls(place_image) == 1


def test_fixed_auto_memory_ops_keep_worker_reuse():
    from decorators import node
    assert "compute_image_metadata" in _FIXED_MEMORY_BY_OP   # guards the Change-2 entry
    @node(auto_memory=True)
    def compute_image_metadata(x) -> int:   # fixed -> excluded from max_calls
        return x
    @node(auto_memory=True)
    def bxt_channel(x) -> int:              # fixed -> excluded from max_calls
        return x
    assert _baked_max_calls(compute_image_metadata) is None
    assert _baked_max_calls(bxt_channel) is None


def test_non_auto_memory_op_keeps_worker_reuse():
    from decorators import node
    @node
    def plain_op(x) -> int:
        return x
    assert _baked_max_calls(plain_op) is None
