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


def test_sky_anchoring_and_gauge_floor():
    a = _composite()
    v = _stf_values(a, ("S", "H", "O"))
    for s in "SHO":
        bp, wp = v[f"black{s}"], v[f"white{s}"]
        # black anchors just below the SKY level, not dragged by signal
        assert PED - 3e-4 < bp < PED
        # white is data-driven: below the old absolute bp+1e-3 floor
        # (which would have overridden p99 on this gauge), yet clearly
        # above sky — the scene holds it up
        assert wp < bp + 1e-3
        assert wp > PED + 2e-4
    # trim_floor lands below sky, above zero — the coverage-ramp cut
    assert 0 < v["trim_floor"] < PED


def test_full_white_keeps_white_at_one():
    v = _stf_values(_composite(), ("S", "H", "O"), full_white=True)
    for s in "SHO":
        assert v[f"white{s}"] == 1.0
