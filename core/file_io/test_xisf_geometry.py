"""XISF geometry must agree with the array the reader returns.

XISF's own `geometry` is `(width, height, channels)`; numpy's shape — what
`read_xisf_image` returns — is `(height, width, channels)`. Passing one where
the other is expected transposes silently: no exception, no wrong pixel, just
an `ImageMetadata` whose dimensions contradict its own image. It went
unnoticed across 1,781 masters because the two consumers that actually need
dimensions (`place_image`'s footprint, the thumbnail budget) both re-read from
disk, and the rest either ignore the fields or use symmetric functions of them
(`min(w, h)`, `w * h`).

The asymmetric fixture is the whole point: a square image passes either way.

Ray-free: tiny synthetic XISFs written by the same library that reads them.
"""
from __future__ import annotations

import numpy as np
import pytest
from xisf import XISF

from file_io.xisf_file_io import read_xisf_image, read_xisf_image_geometry


def _write(path, arr):
    XISF.write(str(path), arr if arr.ndim == 3 else arr[:, :, None],
               creator_app="test")
    return path


@pytest.mark.parametrize("height,width", [(48, 64), (64, 48), (7, 129)])
def test_geometry_matches_the_array_it_describes(tmp_path, height, width):
    """Both orientations, so a transposition can't pass by symmetry."""
    arr = (np.arange(height * width, dtype=np.float32).reshape(height, width) / 1e4)
    path = _write(tmp_path / f"{height}x{width}.xisf", arr)

    geometry = read_xisf_image_geometry(path)
    actual = read_xisf_image(path)

    assert actual.shape[:2] == (height, width), "fixture wrote the wrong shape"
    assert geometry.height_pixels == height
    assert geometry.width_pixels == width
    assert (geometry.height_pixels, geometry.width_pixels) == actual.shape[:2]


def test_pixel_count_is_orientation_independent(tmp_path):
    """`pixels` is a product, so it stayed correct even while transposed —
    which is why nothing downstream caught this."""
    arr = np.zeros((7, 129), dtype=np.float32)
    geometry = read_xisf_image_geometry(_write(tmp_path / "wide.xisf", arr))
    assert geometry.pixels == 7 * 129
