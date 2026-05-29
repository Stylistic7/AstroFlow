import numpy as np

from mosaics.placement.astroflow import _erode_edges


def _disk_with_hole(side=200, r=80, hole=(100, 100)):
    img = np.zeros((side, side), np.float32)
    yy, xx = np.mgrid[0:side, 0:side]
    img[(xx - 100) ** 2 + (yy - 100) ** 2 <= r ** 2] = 1000.0
    img[hole[1], hole[0]] = 0.0  # interior zero — must NOT carve a crater
    return img


def test_erode_edges_shape_following():
    img = _disk_with_hole()
    ero = _erode_edges(img, 10)

    assert (ero != 0).sum() < (img != 0).sum()   # shrank inward
    assert ero[150, 100] == 1000.0               # deep interior preserved
    assert ero[103, 100] == 1000.0               # no crater around interior hole
    assert ero[178, 100] == 0.0                  # ~2px from outer edge -> eroded
    assert ero[165, 100] == 1000.0               # ~15px in -> preserved


def test_erode_edges_zero_is_noop():
    img = np.ones((10, 10), np.float32)
    assert _erode_edges(img, 0) is img
