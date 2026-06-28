"""Unit tests for the additive background pedestal (seam equalization)."""
import numpy as np

from stacks.post_processing.background_pedestal import (
    DEFAULT_PEDESTAL,
    apply_background_pedestal,
)


def test_default_is_a_linear_black_point():
    # A linear black point, NOT a stretch-target background (0.1–0.25). See the
    # module comment: high pedestals lift the sky under every stretch.
    assert DEFAULT_PEDESTAL == 0.005


def test_shift_is_purely_additive_and_sets_sky_to_target():
    rng = np.random.default_rng(0)
    img = (0.30 + 0.001 * rng.standard_normal((128, 128))).astype(np.float32)
    img[40:60, 40:60] += 0.25  # a bright patch — must keep its height above sky

    out = apply_background_pedestal(img, target=0.005)

    # additive: every pixel moved by the same constant offset
    delta = out - img
    assert float(np.std(delta)) < 1e-6
    # sky (lower quartile) landed on the target
    sky = out[out <= np.quantile(out, 0.25)]
    assert abs(float(np.median(sky)) - 0.005) < 5e-4
    # multiplicative structure preserved (patch height above sky unchanged)
    assert abs((out.max() - 0.005) - (img.max() - float(np.median(img)))) < 2e-3


def test_mask_zeros_true_preserves_panel_boundary():
    img = np.full((64, 64), 0.30, dtype=np.float32)
    img[:, :12] = 0.0  # a zero "boundary" strip
    out = apply_background_pedestal(img, target=0.005, mask_zeros=True)
    assert np.all(out[:, :12] == 0.0)                       # zeros preserved
    assert abs(float(np.median(out[:, 12:])) - 0.005) < 5e-4


def test_mask_zeros_false_lifts_subzero_sky_no_holes():
    # GraXpert-like background-subtracted master: sky ~0 with many sub-zero px.
    rng = np.random.default_rng(1)
    img = (0.001 * rng.standard_normal((128, 128))).astype(np.float32)
    assert (img < 0).mean() > 0.3                           # genuinely sub-zero sky

    out = apply_background_pedestal(img, target=0.005, mask_zeros=False)
    assert np.all(out > 0)                                  # no edge-like sub-zero pixels
    sky = out[out <= np.quantile(out, 0.25)]
    assert abs(float(np.median(sky)) - 0.005) < 1e-3
