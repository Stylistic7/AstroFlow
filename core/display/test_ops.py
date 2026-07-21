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
