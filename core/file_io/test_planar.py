"""Planar reconstruction of tiled pyramids (`file_io.planar`).

Pins the contract that lets the redundant TIFF masters stay deleted: a tiled
pyramid must reconstruct to a PIXEL-IDENTICAL stripped TIFF, metadata intact,
because PixInsight cannot read tiled TIFF and fails silently on one.

Ray-free and fast — the fixtures are small synthetic images, not real mosaics.
"""
from __future__ import annotations

import numpy as np
import pyvips
import pytest

from file_io.planar import (
    is_tiled_tiff, planar_path_for, reconstruct_planar_tiff,
)

_DESC = '{"PIPELINE": "AstroFlow-OSS", "CONTHASH": "deadbeef", "FILTER": "Ha"}'


def _write(path, *, tile, pyramid=False, bands=3, w=600, h=400, desc=_DESC):
    a = (np.arange(h * w * bands, dtype=np.float32).reshape(h, w, bands) / 1e5)
    im = pyvips.Image.new_from_memory(a.tobytes(), w, h, bands, "float")
    if desc:
        im.set_type(pyvips.GValue.gstr_type, "image-description", desc)
    im.tiffsave(str(path), compression="deflate", tile=tile, pyramid=pyramid,
                tile_width=128, tile_height=128, bigtiff=True)
    return a


def test_detects_tiled_and_stripped(tmp_path):
    _write(tmp_path / "tiled.tiff", tile=True, pyramid=True)
    _write(tmp_path / "strip.tiff", tile=False)
    assert is_tiled_tiff(tmp_path / "tiled.tiff") is True
    assert is_tiled_tiff(tmp_path / "strip.tiff") is False


def test_non_tiff_passes_through_untouched(tmp_path):
    """XISF/FITS go to PixInsight as-is — reconstruction must not touch them."""
    p = tmp_path / "master.xisf"
    p.write_bytes(b"XISF0100 not really")
    assert reconstruct_planar_tiff(p) == p


def test_stripped_tiff_passes_through_untouched(tmp_path):
    """The common case costs nothing: already readable, so return it as-is."""
    p = tmp_path / "strip.tiff"
    _write(p, tile=False)
    assert reconstruct_planar_tiff(p) == p


def test_reconstruction_is_pixel_identical_and_stripped(tmp_path):
    src = tmp_path / "tiled.tiff"
    expected = _write(src, tile=True, pyramid=True)

    out = reconstruct_planar_tiff(src)

    assert out != src
    # The whole point: PixInsight reads stripped TIFF only.
    assert is_tiled_tiff(out) is False

    got = pyvips.Image.tiffload(str(out), page=0, unlimited=True)
    arr = np.ndarray(buffer=got.write_to_memory(), dtype=np.float32,
                     shape=(got.height, got.width, got.bands))
    assert np.array_equal(arr, expected)
    assert got.get("image-description") == _DESC


def test_reconstruction_is_cached(tmp_path):
    """A gigapixel rebuild is ~100 s; a second click must not repeat it."""
    src = tmp_path / "tiled.tiff"
    _write(src, tile=True, pyramid=True)

    first = reconstruct_planar_tiff(src)
    stamp = first.stat().st_mtime_ns
    second = reconstruct_planar_tiff(src)

    assert second == first
    assert second.stat().st_mtime_ns == stamp   # not rewritten


def test_rebuilt_source_invalidates_the_cache(tmp_path):
    """The cache key folds in (mtime, size), so a rebuilt tap can never serve a
    stale reconstruction — the failure that would silently show old pixels."""
    src = tmp_path / "tiled.tiff"
    _write(src, tile=True, pyramid=True)
    before = planar_path_for(src)

    _write(src, tile=True, pyramid=True, w=640)   # rebuilt, different geometry
    after = planar_path_for(src)

    assert before != after


def test_missing_source_does_not_raise_on_path_computation(tmp_path):
    """`planar_path_for` is used for reporting too — it must tolerate a
    vanished file rather than blowing up the caller."""
    assert planar_path_for(tmp_path / "gone.tiff").name.endswith(".planar.tiff")
