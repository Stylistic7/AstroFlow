"""Serving XISF masters by memory map (`file_io.xisf_mmap`).

The load-bearing tests are the GUARDS. The fast path reinterprets a byte range
as raw float32, so a compressed or multi-channel block read that way yields
plausible-looking GARBAGE rather than an error — the failure would surface as a
wrong picture, not an exception. Every precondition therefore has a test.

Ray-free: fixtures are tiny synthetic XISFs written by the same library that
reads them.
"""
from __future__ import annotations

import numpy as np
import pytest
from xisf import XISF

from file_io.xisf_mmap import can_mmap, open_vips, pyramid_levels


def _write(path, arr, codec=None):
    XISF.write(str(path), arr if arr.ndim == 3 else arr[:, :, None],
               creator_app="test", codec=codec)
    return path


@pytest.fixture
def mono(tmp_path):
    a = (np.arange(64 * 48, dtype=np.float32).reshape(48, 64) / 1e4)
    return _write(tmp_path / "mono.xisf", a), a


def test_uncompressed_mono_is_mappable(mono):
    path, _ = mono
    assert can_mmap(path) is True


def test_mapped_pixels_match_the_library_reader(mono):
    """The whole design rests on this: the mapping must be bit-identical to
    what `read_image` produces, or the viewer shows something subtly wrong."""
    path, expected = mono
    img, keep = open_vips(path)

    got = np.ndarray(buffer=img.write_to_memory(), dtype=np.float32,
                     shape=(img.height, img.width))
    assert img.width == expected.shape[1]
    assert img.height == expected.shape[0]
    assert np.array_equal(got, expected)
    del img, keep


def test_compressed_falls_back_instead_of_reading_garbage(tmp_path):
    """A compressed block read as raw floats is GARBAGE, not an error — so the
    guard must refuse the fast path, and the fallback must still be correct."""
    a = (np.arange(64 * 48, dtype=np.float32).reshape(48, 64) / 1e4)
    path = _write(tmp_path / "zlib.xisf", a, codec="zlib")

    assert can_mmap(path) is False          # guard refuses

    img, keep = open_vips(path)             # fallback still works
    got = np.ndarray(buffer=img.write_to_memory(), dtype=np.float32,
                     shape=(img.height, img.width))
    assert np.array_equal(got, a)
    del img, keep


def test_multichannel_falls_back(tmp_path):
    """XISF may store channels planar while vips expects interleaved, so
    multi-channel takes the decode path rather than risking a scrambled map."""
    a = np.random.default_rng(0).random((32, 40, 3)).astype(np.float32)
    path = _write(tmp_path / "rgb.xisf", a)

    assert can_mmap(path) is False

    img, keep = open_vips(path)
    assert (img.width, img.height, img.bands) == (40, 32, 3)
    del img, keep


def test_non_xisf_is_not_mappable(tmp_path):
    p = tmp_path / "not.tiff"
    p.write_bytes(b"II*\x00")
    assert can_mmap(p) is False


def test_unreadable_header_is_not_mappable(tmp_path):
    p = tmp_path / "truncated.xisf"
    p.write_bytes(b"XISF0100 truncated garbage")
    assert can_mmap(p) is False


@pytest.mark.parametrize("w,h,expected", [
    (256, 256, 1),        # already one tile
    (512, 512, 2),
    (9570, 6385, 7),      # a real stack master — matches its old pyramid
    (48207, 46699, 9),    # a gigapixel mosaic
])
def test_level_count_matches_a_stored_pyramid(w, h, expected):
    """The viewer's zoom range must not change with the storage format: this
    has to agree with what `tiffsave(pyramid=True)` would have produced."""
    assert pyramid_levels(w, h) == expected


def test_viewable_from_master_serves_xisf_directly(tmp_path, mono):
    """`viewable_from_master` is the ONE 'exists = viewable' definition; an
    XISF with no pyramid must now resolve to itself rather than None."""
    from catalog.queries import viewable_from_master
    path, _ = mono
    assert viewable_from_master(path) == path


def test_xisf_never_gets_a_pyramid_built(mono):
    """The 273.8 GB must not come back on the next run."""
    from previews.create_tile_pyramid import needs_display_pyramid
    path, _ = mono
    assert needs_display_pyramid(path) is False
