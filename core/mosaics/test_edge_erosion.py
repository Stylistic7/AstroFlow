import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes

from mosaics.placement.astroflow import _erode_edges


def _erode_edges_original(image, px):
    """The original pre-optimization implementation — global distance
    transform, `binary_fill_holes`, out-of-place. Kept here as the golden
    reference: the new tiled/in-place/no-fill-holes version must match this
    bit-for-bit on the solid footprints that `place_image` actually produces."""
    if px <= 0:
        return image
    mask = binary_fill_holes(image != 0)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return np.where(dist >= px, image, 0.0).astype(np.float32)


def _disk(side=200, r=80):
    """A solid, hole-free footprint — the shape `place_image` actually
    produces (a projected source rectangle is simply-connected)."""
    img = np.zeros((side, side), np.float32)
    yy, xx = np.mgrid[0:side, 0:side]
    img[(xx - 100) ** 2 + (yy - 100) ** 2 <= r ** 2] = 1000.0
    return img


def test_erode_edges_bit_identical_to_original():
    """The whole justification for the rewrite: on solid, hole-free footprints
    (every real placed panel), the new tiled/in-place/no-fill-holes erosion is
    exactly equal to the original global one — across footprint shapes, erosion
    radii, AND tile sizes small enough to force many-block + halo paths."""
    rng = np.random.default_rng(0)
    disk = _disk(side=300, r=120)
    disk[disk != 0] += rng.random(int((disk != 0).sum())).astype(np.float32)
    rect = np.zeros((300, 300), np.float32)
    rect[40:260, 60:240] = rng.random((220, 180)).astype(np.float32) + 0.05
    for name, img in (("disk", disk), ("rect", rect)):
        for px in (1, 10, 37):
            for tile in (64, 128, 512):   # 64/128 force multi-block + halos
                new = _erode_edges(img.copy(), px, tile=tile)
                old = _erode_edges_original(img.copy(), px)
                assert np.array_equal(new, old), f"{name} px={px} tile={tile}"


def test_erode_edges_shape_following():
    img = _disk()
    before = int((img != 0).sum())
    ero = _erode_edges(img, 10, tile=64)     # small tile -> multi-block path

    assert (ero != 0).sum() < before             # shrank inward
    assert ero[150, 100] == 1000.0               # deep interior preserved
    assert ero[178, 100] == 0.0                  # ~2px from outer edge -> eroded
    assert ero[165, 100] == 1000.0               # ~15px in -> preserved


def test_erode_edges_zero_is_noop():
    img = np.ones((10, 10), np.float32)
    assert _erode_edges(img, 0) is img


def test_erode_edges_no_hole_filling():
    """Documents the intentional removal of `binary_fill_holes`: without it,
    an *interior* exact-zero shaves a crater. This never happens for a real
    placed panel — `place_image` projects a gap-free source rectangle and the
    masters carry a background pedestal, so there are no interior zeros — but
    the test pins the simpler local semantics so a future reader knows the
    global hole-fill was dropped deliberately, not by accident."""
    img = _disk()
    img[100, 100] = 0.0  # a synthetic interior zero (cannot occur in practice)
    ero = _erode_edges(img, 10)
    assert ero[103, 100] == 0.0  # crater IS carved around the interior zero
