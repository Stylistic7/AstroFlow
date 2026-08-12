"""Ground-truth the display ops (display/ops.py, display/finishes.py) against the
canonical algorithms they implement — fast, no PixInsight, no Ray.

These pin the MATH: SCNR is PixInsight's average-neutral formula exactly, the MTF
is the app's PixInsight MTF exactly, GHS is a well-formed monotone stretch, and
saturation preserves the luma it's defined on. The vs-actual-PixInsight
integration tests (run real PI) live in pix/test_display_ops_vs_pixinsight.py.
"""
import numpy as np
import pytest

from display.ops import _scnr_np, _saturation_np, ghs_channel_np
from display.finishes import _mtf as finish_mtf
from mosaics.color import palettes

_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _rgb(seed=0, n=64):
    return np.random.default_rng(seed).random((n, n, 3)).astype(np.float32)


class TestSCNR:
    """PixInsight SCNR, Average Neutral protection, with amount a:
        G' = (1-a)·G + a·min(G, (R+B)/2)   ;  R, B untouched."""

    @pytest.mark.parametrize("a", [0.0, 0.25, 0.5, 0.9, 1.0])
    def test_matches_pixinsight_average_neutral(self, a):
        rgb = _rgb()
        mine = _scnr_np(rgb, {"scnr": a})
        g = rgb[..., 1]
        ref_g = (1.0 - a) * g + a * np.minimum(g, 0.5 * (rgb[..., 0] + rgb[..., 2]))
        np.testing.assert_allclose(mine[..., 1], ref_g, atol=1e-6)
        np.testing.assert_array_equal(mine[..., 0], rgb[..., 0])   # R untouched
        np.testing.assert_array_equal(mine[..., 2], rgb[..., 2])   # B untouched

    def test_amount_zero_is_identity(self):
        rgb = _rgb()
        np.testing.assert_allclose(_scnr_np(rgb, {"scnr": 0.0}), rgb, atol=1e-7)

    @pytest.mark.parametrize("a", [0.3, 1.0])
    def test_never_increases_green(self, a):
        rgb = _rgb()
        assert np.all(_scnr_np(rgb, {"scnr": a})[..., 1] <= rgb[..., 1] + 1e-6)


class TestMTF:
    """The finish MTF must equal the app's PixInsight MTF (palettes._mtf)."""

    @pytest.mark.parametrize("m", [0.05, 0.1, 0.2, 0.5, 0.8, 0.95])
    def test_matches_app_pixinsight_mtf(self, m):
        x = np.linspace(0, 1, 1000).astype(np.float32)
        np.testing.assert_allclose(finish_mtf(x, m), palettes._mtf(x, m), atol=1e-5)

    def test_midtone_half_is_identity(self):
        x = np.linspace(0, 1, 101).astype(np.float32)
        np.testing.assert_allclose(finish_mtf(x, 0.5), x, atol=1e-6)

    def test_endpoints_fixed(self):
        assert abs(float(finish_mtf(np.array([0.0], np.float32), 0.2)[0])) < 1e-6
        assert abs(float(finish_mtf(np.array([1.0], np.float32), 0.2)[0]) - 1.0) < 1e-6


class TestGHS:
    """Generalized Hyperbolic Stretch must be a well-formed monotone stretch on
    [0,1] with fixed endpoints — the ghsChannel WGSL twin's numpy reference."""

    @pytest.mark.parametrize("d,b,sp", [
        (2.4, 0.0, 0.1), (1.0, -1.0, 0.1), (3.0, 2.0, 0.2), (0.5, -0.5, 0.05),
    ])
    def test_endpoints_monotone_bounded(self, d, b, sp):
        z = np.linspace(0, 1, 4001).astype(np.float32)
        y = ghs_channel_np(z, d, b, sp)
        assert abs(float(y[0])) < 1e-4                 # 0 → 0
        assert abs(float(y[-1]) - 1.0) < 1e-4          # 1 → 1
        assert np.all(np.diff(y) >= -1e-5)             # monotone non-decreasing
        assert np.all((y >= -1e-6) & (y <= 1.0 + 1e-6))

    def test_zero_strength_is_identity(self):
        z = np.linspace(0, 1, 100).astype(np.float32)
        np.testing.assert_allclose(ghs_channel_np(z, 0.0, 0.0, 0.1), z, atol=1e-6)

    def test_lifts_the_pivot(self):
        # SP sits at the dark-tail background; a positive stretch LIFTS it (the
        # crushed-blacks fix) — maps it to a brighter display value than input.
        y_sp = float(ghs_channel_np(np.array([0.1], np.float32), 2.4, 0.0, 0.1)[0])
        assert y_sp > 0.1


class TestSaturation:
    """Luma-preserving saturation: chroma scales, luma is invariant (where the
    result doesn't clip). NOTE: this is NOT PixInsight ColorSaturation (HSV) — a
    deliberate definitional choice, exercised against real PI in the slow tests."""

    @pytest.mark.parametrize("s", [0.0, 0.5, 1.5])
    def test_preserves_luma_where_unclipped(self, s):
        rgb = _rgb()
        out = _saturation_np(rgb, {"saturation": s})
        unclipped = np.all((out > 1e-4) & (out < 1.0 - 1e-4), axis=2)
        np.testing.assert_allclose((out @ _LUMA)[unclipped],
                                   (rgb @ _LUMA)[unclipped], atol=1e-5)

    def test_zero_is_gray(self):
        out = _saturation_np(_rgb(), {"saturation": 0.0})
        assert np.allclose(out[..., 0], out[..., 1], atol=1e-5)
        assert np.allclose(out[..., 1], out[..., 2], atol=1e-5)


class TestLiveBroadbandFinishes:
    """The multi-source compose finishes (RGB/LRGB/HaRGB/HaLRGB live) — numpy
    twins of the compose + chain, plus the WGSL emission contract."""

    def _sources(self, seed=1, n=32):
        rng = np.random.default_rng(seed)
        combine = (rng.random((n, n, 3)) * 0.05).astype(np.float32)
        stars = np.zeros_like(combine)
        stars[::7, ::7] = 0.4                       # sparse point sources
        nb = (rng.random((n, n, 3)) * 0.04).astype(np.float32)
        lum = (rng.random((n, n, 1)) * 0.06).astype(np.float32)
        return combine, stars, nb, lum

    def test_star_intensity_algebra(self):
        """s=1 → the plain combine; s=0 → exactly the starless (combine −
        stars); the chain (linked MTF) is monotone so ordering survives."""
        from display.finishes import RGB_LIVE
        combine, stars, _, _ = self._sources()
        plain = RGB_LIVE.bake([combine, stars], {"star_s": 1.0})
        starless = RGB_LIVE.bake([combine, stars], {"star_s": 0.0})
        from display.ops import _linked_mtf_np
        v = RGB_LIVE.defaults()
        assert np.allclose(plain, _linked_mtf_np(combine, v), atol=1e-6)
        assert np.allclose(starless, _linked_mtf_np(combine - stars, v), atol=1e-6)

    def test_ha_k_zero_matches_rgb_and_k_raises_red_only(self):
        from display.finishes import HARGB_LIVE, RGB_LIVE
        combine, stars, nb, _ = self._sources()
        k0 = HARGB_LIVE.bake([combine, stars, nb], {"ha_k": 0.0})
        rgb = RGB_LIVE.bake([combine, stars])
        assert np.allclose(k0, rgb, atol=1e-6)
        k1 = HARGB_LIVE.bake([combine, stars, nb], {"ha_k": 1.0, "ha_q": 0.0})
        assert (k1[..., 0] >= k0[..., 0] - 1e-6).all(), "Hα must only raise R"
        assert np.allclose(k1[..., 1:], k0[..., 1:], atol=1e-6), "G/B untouched"

    def test_continuum_subtraction_clamps_at_zero(self):
        """q=1 with Hα ≤ R everywhere → the max(…, 0) clamp makes k a no-op."""
        from display.finishes import HARGB_LIVE
        combine, stars, _, _ = self._sources()
        nb = np.zeros_like(combine)
        nb[..., 1] = (combine[..., 0] * 0.5)         # Hα = half the R continuum
        out = HARGB_LIVE.bake([combine, stars, nb], {"ha_k": 1.0, "ha_q": 2.0})
        base = HARGB_LIVE.bake([combine, stars, nb], {"ha_k": 0.0})
        assert np.allclose(out, base, atol=1e-6)

    def test_lrgb_lab_own_lightness_is_identity(self):
        """The Lab invariant (mirror of the palettes.lrgb_vips test): identity
        lum stretch + aux == the composed RGB's own luminance Y + chroma=1
        reproduces the stretched RGB (grey (y,y,y) has exactly L*(Y=y))."""
        from display.finishes import LRGB_LIVE
        from display.ops import _linked_mtf_np
        combine, stars, _, _ = self._sources()
        v = LRGB_LIVE.defaults()
        stretched = _linked_mtf_np(combine, v)
        y = stretched @ np.array([0.2126729, 0.7151522, 0.0721750])
        # aux must land at y AFTER the aux stretch: identity stretch + pre-
        # inverting the CHAIN stretch on aux is fiddly, so drive aux through
        # the same linked MTF by construction: pick raw aux == raw combine's
        # channels' equivalent is impossible per-pixel — instead feed aux
        # POST-stretch by setting the lum stretch to identity and aux = y at
        # compose time via a pre-stretched source array.
        lum = y[..., None].astype(np.float32)
        out = LRGB_LIVE.bake(
            [stretched, np.zeros_like(stretched), lum],
            {"star_s": 1.0, "blackpoint": 0.0, "whitepoint": 1.0,
             "midtones": 0.5, "lum_black": 0.0, "lum_white": 1.0,
             "lum_mid": 0.5, "chroma": 1.0})
        assert np.allclose(out, np.clip(stretched, 0, 1), atol=2e-3), (
            np.abs(out - stretched).max())

    def test_wgsl_emission_contract(self):
        """Multi-source finishes emit displayFinish(src0..srcN, P), the compose
        text, the aux local, and one DisplayParams field per param — the exact
        shapes the frontend's generated fs_main splices against."""
        from display.finishes import HALRGB_LIVE, BLACK_WHITE_POINT
        w = HALRGB_LIVE.to_wgsl()
        assert "fn displayFinish(src0: vec4<f32>, src1: vec4<f32>, "
        assert "src3: vec4<f32>, P: DisplayParams)" in w
        assert "var aux: f32 = 0.0;" in w and "aux = src3.r;" in w
        for p in ("star_s", "ha_k", "ha_q", "blackpoint", "lum_mid", "chroma"):
            assert f"{p}: f32," in w, f"param {p} missing from DisplayParams"
        # Single-input finishes keep working under the same contract.
        w1 = BLACK_WHITE_POINT.to_wgsl()
        assert "fn displayFinish(src0: vec4<f32>, P: DisplayParams)" in w1
        assert "var c: vec3<f32> = src0.rgb;" in w1
