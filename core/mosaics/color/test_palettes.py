"""Unit tests for the pure-numpy color math (autostretch, palettes, screen)."""
import numpy as np

from mosaics.color.palettes import (
    autostretch,
    combine_star_rgb,
    foraxx,
    hubble,
    screen,
)


def test_autostretch_maps_median_to_target_and_bounds():
    rng = np.random.default_rng(0)
    x = np.clip(0.10 + 0.02 * rng.standard_normal((256, 256)), 0, 1).astype(np.float32)
    out = autostretch(x, target_background=0.15)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.all(np.isfinite(out))
    # median is preserved by the monotonic MTF, so it lands on the target
    assert abs(float(np.median(out)) - 0.15) < 0.02


def test_hubble_is_sho_channel_assignment():
    s = np.full((4, 4), 0.2, np.float32)
    h = np.full((4, 4), 0.5, np.float32)
    o = np.full((4, 4), 0.8, np.float32)
    img = hubble(s, h, o)
    assert img.shape == (4, 4, 3)
    assert np.allclose(img[..., 0], 0.2)  # R = SII
    assert np.allclose(img[..., 1], 0.5)  # G = Ha
    assert np.allclose(img[..., 2], 0.8)  # B = OIII


def test_foraxx_extremes_and_bounds():
    s = np.full((4, 4), 0.4, np.float32)
    h = np.full((4, 4), 0.6, np.float32)
    # O3 == 0: wR=0^1=0 -> R=Ha; B=0; m=0 -> wG=0 -> G=O3=0
    img0 = foraxx(s, h, np.zeros((4, 4), np.float32))
    assert np.allclose(img0[..., 0], 0.6)   # R == Ha
    assert np.allclose(img0[..., 1], 0.0)   # G == 0
    assert np.allclose(img0[..., 2], 0.0)   # B == 0
    # O3 == 1: wR=1^0=1 -> R=SII; B=1
    img1 = foraxx(s, h, np.ones((4, 4), np.float32))
    assert np.allclose(img1[..., 0], 0.4)   # R == SII
    assert np.allclose(img1[..., 2], 1.0)   # B == 1
    # general: finite + bounded (no NaN from 0**0 etc.)
    rng = np.random.default_rng(1)
    a, b, c = (rng.random((32, 32)).astype(np.float32) for _ in range(3))
    img = foraxx(a, b, c)
    assert img.shape == (32, 32, 3)
    assert np.all(np.isfinite(img)) and img.min() >= 0 and img.max() <= 1


def test_combine_star_rgb_shape():
    r, g, b = (np.full((8, 8), v, np.float32) for v in (0.1, 0.2, 0.3))
    img = combine_star_rgb(r, g, b)
    assert img.shape == (8, 8, 3)
    assert np.allclose(img[..., 0], 0.1) and np.allclose(img[..., 2], 0.3)


def test_screen_identity_bounds_and_brighten():
    rng = np.random.default_rng(2)
    base = rng.random((16, 16, 3)).astype(np.float32)
    stars = rng.random((16, 16, 3)).astype(np.float32)
    assert np.allclose(screen(base, np.zeros_like(base)), base)        # screen with black = no-op
    assert np.allclose(screen(base, np.ones_like(base)), 1.0)          # screen with white = white
    out = screen(base, stars)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.all(out >= base - 1e-6)                                  # screen only brightens
