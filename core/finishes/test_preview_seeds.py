"""_stf_values — the Python twin of the viewer's stfSeed (2026-09-03
recalibration). Pins the two input shapes the canonical preview actually
sends (a 3-band composite through BOTH the per-channel and the pooled
LINKED path — the linked path with a 3-D array is the one the first
deploy broke) and the recipe's three properties: sky anchoring, the
gauge-aware white floor, and the trim_floor emission.
"""
import numpy as np

from finishes.preview import _stf_values

PED = 0.005


def _composite():
    rng = np.random.default_rng(5)
    h, w = 120, 160
    a = np.full((h, w, 3), PED, np.float32)
    a += rng.normal(0, 2e-5, a.shape).astype(np.float32)
    a[30:60, 40:90, :] += 4e-4                    # nebulosity
    for _ in range(40):                           # stars
        y, x = rng.integers(2, h - 2), rng.integers(2, w - 2)
        a[y, x, :] += rng.uniform(0.05, 0.5)
    return a


def test_linked_path_accepts_a_3band_array():
    """The pooled (suffixes=None) path with an (H, W, 3) array — the
    canonical preview's RGB branch and the star_mid probe both send
    this; a 2-D sky mask must mask per-pixel, not broadcast-fail."""
    v = _stf_values(_composite(), None)
    assert set(v) >= {"blackpoint", "whitepoint", "midtones", "trim_floor"}
    assert 0 < v["blackpoint"] < PED
    assert v["whitepoint"] > v["blackpoint"]


def test_sky_anchoring_and_bright_white():
    a = _composite()
    v = _stf_values(a, ("S", "H", "O"))
    for s in "SHO":
        bp, wp = v[f"black{s}"], v[f"white{s}"]
        # black anchors just below the SKY level, not dragged by signal
        assert PED - 3e-4 < bp < PED
        # white is a BRIGHT anchor — far above sky, so the background's
        # own noise cannot fill the display range
        assert wp > PED * 2
    # trim_floor lands below sky, above zero — the coverage-ramp cut
    assert 0 < v["trim_floor"] < PED


def test_white_does_not_collapse_on_a_sparse_field():
    """The 2026-09-03 regression: an astro frame is ~99% background, so a
    p99 white anchor sits a few sigma above sky, the span collapses, and
    everything brighter blows out (1.3% of a real broadband target
    saturated; the display 'popped' washed-out the moment the seed
    landed). The bright anchor must stay clear of the noise even when
    only a fraction of a percent of pixels carry signal."""
    rng = np.random.default_rng(11)
    h, w = 200, 200
    a = np.full((h, w, 3), PED, np.float32)
    a += rng.normal(0, 2e-5, a.shape).astype(np.float32)
    n_bright = int(0.002 * h * w)                 # 0.2% of the frame
    ys = rng.integers(0, h, n_bright); xs = rng.integers(0, w, n_bright)
    a[ys, xs, :] = 0.4
    v = _stf_values(a, None)
    sky_sigma = 1.4826 * float(np.median(np.abs(a - np.median(a))))
    assert v["whitepoint"] > np.median(a) + 20 * sky_sigma, (
        f"white {v['whitepoint']:.6f} collapsed onto the noise floor")
    # and the stretch must not be so tight that the background saturates
    assert v["whitepoint"] - v["blackpoint"] > 30 * sky_sigma


def test_display_referred_seeds_the_identity():
    v = _stf_values(_composite(), ("S", "H", "O"), display=True)
    for s in "SHO":
        assert (v[f"black{s}"], v[f"white{s}"], v[f"mid{s}"]) == (0.0, 1.0, 0.5)
    assert v["blackpoint"] == 0.0 and v["whitepoint"] == 1.0 and v["trim_floor"] == 0.0


def test_per_band_and_linked_keys_are_both_emitted():
    v = _stf_values(_composite(), ("S", "H", "O"))
    assert {"blackS", "whiteH", "midO", "blackpoint", "whitepoint", "midtones",
            "trim_floor", "star_white"} <= set(v)
    mono = _composite()[..., 1:2]
    w = _stf_values(mono, ("H",))
    assert "blackH" in w and "blackS" not in w
