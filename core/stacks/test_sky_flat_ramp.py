"""The ramp measurement of a night-sky flat (`stacks/sky_flat_ramp.py`)."""
import numpy as np

from stacks.sky_flat_ramp import measure_ramp

H, W = 400, 600


def _flat(ramp_pct=0.0, angle_deg=0.0, vignetting=0.08, seed=0):
    yy, xx = np.mgrid[0:H, 0:W]
    hd = 0.5 * np.hypot(W, H)
    x, y = (xx - (W - 1) / 2) / hd, (yy - (H - 1) / 2) / hd
    r2 = x * x + y * y
    flat = 1.0 - vignetting * r2
    if ramp_pct:
        t = np.radians(angle_deg)
        u = x * np.cos(t) + y * np.sin(t)                    # rises along angle_deg
        span = np.ptp(u)
        flat = flat + (ramp_pct / 100.0) * (u - u.mean()) / span
    rng = np.random.default_rng(seed)
    flat = flat + 0.001 * rng.standard_normal(flat.shape)
    stars = rng.random(flat.shape) < 0.002                    # residual stars the clip must ignore
    flat[stars] += 0.2
    return flat


def test_centred_vignetting_alone_is_no_ramp():
    m = measure_ramp(_flat())
    assert m["ramp_pct"] < 0.05, m
    assert 7.0 < m["vignetting_pct"] < 9.0, m


def test_a_ramp_is_measured_with_its_size_and_direction():
    for angle in (0.0, 90.0, 225.0):
        m = measure_ramp(_flat(ramp_pct=1.0, angle_deg=angle))
        assert 0.8 < m["ramp_pct"] < 1.3, (angle, m)
        d = (m["angle_deg"] - angle + 180) % 360 - 180
        assert abs(d) < 10, (angle, m)


def test_nan_and_zero_pixels_are_ignored():
    f = _flat(ramp_pct=0.6, angle_deg=45.0)
    f[:20] = 0.0
    f[50:60, 100:200] = np.nan
    m = measure_ramp(f)
    assert 0.4 < m["ramp_pct"] < 0.8, m
