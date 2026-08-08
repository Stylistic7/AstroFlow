"""Unit tests for the pure-numpy color math (autostretch, palettes, screen)."""
import numpy as np
import pyvips

from mosaics.color.palettes import (
    autostretch,
    combine_star_rgb,
    combine_star_rgb_vips,
    foraxx,
    foraxx_vips,
    hubble,
    hubble_vips,
    screen,
    screen_vips,
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


# --- pyvips ports must match the numpy reference ----------------------------
# The @node deliverables run the streaming vips versions on full-res mosaics
# (a 2.25 Gpx RGB image is 27 GB); this pins them to the tested numpy math.

def _to_vips(a: np.ndarray) -> pyvips.Image:
    a = np.ascontiguousarray(a, np.float32)
    bands = 1 if a.ndim == 2 else a.shape[2]
    return pyvips.Image.new_from_memory(a.tobytes(), a.shape[1], a.shape[0], bands, "float")


def _to_np(im: pyvips.Image) -> np.ndarray:
    return np.ndarray(buffer=im.write_to_memory(), dtype=np.float32,
                      shape=[im.height, im.width, im.bands])


def test_autostretch_vips_matches_numpy():
    """The streaming (two-pass histogram) autostretch must match the numpy
    reference. Median/MAD are quantized to 1/65535 by the 16-bit histogram,
    which perturbs the MTF parameters by ~1.5e-5; the resulting pixel delta
    is bounded well under 2e-3 — invisible for a starting display stretch,
    and deterministic. Exercises the exact code path `combine_star_rgb`
    runs on gigapixel mosaics (where the numpy version held 2-3 full
    copies and got the worker SIGKILLed)."""
    from mosaics.color.palettes import autostretch, autostretch_vips

    rng = np.random.default_rng(7)
    # Star-layer-like: mostly near-black with sparse bright values.
    arr = (rng.random((64, 64, 3)).astype(np.float32) ** 6)

    np_out = autostretch(arr)
    vips_out = _to_np(autostretch_vips(lambda: _to_vips(arr)))

    assert vips_out.shape == np_out.shape
    assert np.all(np.isfinite(vips_out))
    assert float(np.abs(vips_out - np_out).max()) < 2e-3


def test_vips_ports_match_numpy():
    rng = np.random.default_rng(3)
    s, h, o = (rng.random((48, 48)).astype(np.float32) for _ in range(3))
    r, g, b = (rng.random((48, 48)).astype(np.float32) for _ in range(3))
    base = rng.random((48, 48, 3)).astype(np.float32)
    overlay = rng.random((48, 48, 3)).astype(np.float32)

    def close(np_out, vips_out):
        return np.allclose(np_out, _to_np(vips_out), atol=1e-5)

    assert close(hubble(s, h, o), hubble_vips(_to_vips(s), _to_vips(h), _to_vips(o)))
    assert close(foraxx(s, h, o), foraxx_vips(_to_vips(s), _to_vips(h), _to_vips(o)))
    assert close(combine_star_rgb(r, g, b),
                 combine_star_rgb_vips(_to_vips(r), _to_vips(g), _to_vips(b)))
    assert close(screen(base, overlay), screen_vips(_to_vips(base), _to_vips(overlay)))
