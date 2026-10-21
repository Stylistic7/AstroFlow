"""The face rasters and the forward projection against astropy-healpix.

`testdata/healpix_reference.json` holds nside-1024 NESTED indices of 400 sky
points (40 within 1e-3 of the |z| = 2/3 belt boundary, 40 within 2° of a
pole) and the centres of 100 cells, both from astropy-healpix. Pure numpy;
no app, no Ray."""
import json
from pathlib import Path

import numpy as np
import pytest

from allsky.healpix_faces import (
    compact_bits, degrade, faces_from_nested, nested_index, spread_bits, vec_from_lonlat,
    xyf_from_vec,
)

_REF = json.loads((Path(__file__).parent / "testdata" / "healpix_reference.json").read_text())


def test_bit_spread_and_compact_invert():
    v = np.arange(0, 1 << 12, dtype=np.uint64)
    assert np.array_equal(compact_bits(spread_bits(v)), v)
    assert np.array_equal(compact_bits(spread_bits(v) << np.uint64(1)), np.zeros_like(v))


def test_forward_projection_matches_astropy_on_random_points():
    nside = _REF["nside"]
    pts = np.array(_REF["points"])
    face, fx, fy = xyf_from_vec(vec_from_lonlat(pts[:, 0], pts[:, 1]))
    got = nested_index(face, np.floor(fx * nside), np.floor(fy * nside), nside)
    assert np.array_equal(got, pts[:, 2].astype(np.int64))


def test_cell_centres_project_to_the_middle_of_their_cell():
    nside = _REF["nside"]
    c = np.array(_REF["centres"])
    face, fx, fy = xyf_from_vec(vec_from_lonlat(c[:, 1], c[:, 2]))
    ix, iy = np.floor(fx * nside), np.floor(fy * nside)
    assert np.array_equal(nested_index(face, ix, iy, nside), c[:, 0].astype(np.int64))
    # The continuous coordinate is the real thing, not just the right bin.
    assert np.allclose(fx * nside - ix, 0.5, atol=2e-3)
    assert np.allclose(fy * nside - iy, 0.5, atol=2e-3)


def test_faces_are_the_map_in_raster_order():
    nside = 4
    values = np.arange(12 * nside * nside, dtype=np.float32)
    faces = faces_from_nested(values, nside)
    assert faces.shape == (12, nside, nside)
    f, iy, ix = np.meshgrid(np.arange(12), np.arange(nside), np.arange(nside), indexing="ij")
    assert np.array_equal(faces, values[nested_index(f, ix, iy, nside)].reshape(faces.shape))


def test_faces_reject_the_wrong_size():
    with pytest.raises(ValueError):
        faces_from_nested(np.zeros(12 * 16 + 1), 4)
    with pytest.raises(ValueError):
        faces_from_nested(np.zeros(12 * 9), 3)


def test_degrade_averages_two_by_two_blocks():
    faces = np.arange(12 * 4 * 4, dtype=np.float32).reshape(12, 4, 4)
    d = degrade(faces)
    assert d.shape == (12, 2, 2)
    assert d[0, 0, 0] == pytest.approx(faces[0, :2, :2].mean())
    assert d[11, 1, 1] == pytest.approx(faces[11, 2:, 2:].mean())
