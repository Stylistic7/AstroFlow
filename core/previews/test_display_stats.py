"""Exact streaming display statistics (2026-10-04): every quantile equals the
numpy order statistic of the covered pixels, the sky subset is the pixels
below the luminance cut, and no-data (exact zero) never counts."""
import numpy as np
import pyvips
import pytest

from previews.display_stats import display_stats


def _img(a):
    a = np.ascontiguousarray(a, np.float32)
    return pyvips.Image.new_from_memory(a.data, a.shape[1], a.shape[0], a.shape[2], "float"), a


@pytest.fixture
def scene():
    rng = np.random.default_rng(3)
    h, w = 700, 530                       # several strips, a ragged last one
    a = 0.005 + rng.normal(0, 8e-6, (h, w, 3))
    a[100:300, 50:250] += 4e-4            # nebula
    stars = rng.random((h, w)) < 0.002
    a[stars] += rng.uniform(0.01, 0.5, (stars.sum(), 1))
    a[:40] = 0.0                          # uncovered rim
    a[400, 10, 1] = 0.0                   # one band missing -> uncovered
    return a


def test_quantiles_are_exact_order_statistics(scene):
    img, a = _img(scene)
    stats = display_stats(img, coarse=img)
    v = a.reshape(-1, 3)
    v = v[np.all(v > 0, axis=1)]
    lum = v.mean(axis=1)
    sky = v[lum < np.percentile(lum, 60)]
    for b, s in enumerate(stats):
        col, sc = np.sort(v[:, b]), np.sort(sky[:, b])
        k = lambda n, q: int(round(q * (n - 1)))
        for name, q in (("p1", .01), ("p25", .25), ("median", .5), ("p75", .75),
                        ("p95", .95), ("p99", .99), ("p999", .999)):
            assert s[name] == col[k(col.size, q)], name
        for name, q in (("skyP25", .25), ("skyMedian", .5), ("skyP75", .75)):
            assert s[name] == sc[k(sc.size, q)], name
        assert s["min"] == col[0] and s["max"] == col[-1]
        assert s["sigma"] == pytest.approx((s["p75"] - s["p25"]) / 1.349)
        assert s["skySigma"] == pytest.approx((s["skyP75"] - s["skyP25"]) / 1.349)
        assert s["high"] == max(s["p999"], s["median"] + 60 * s["sigma"])


def test_mono_and_empty():
    img, _ = _img(np.full((50, 60, 1), 0.007))
    s = display_stats(img)
    assert len(s) == 1 and s[0]["median"] == np.float32(0.007) and s[0]["skyMedian"] == np.float32(0.007)
    assert display_stats(_img(np.zeros((20, 20, 1)))[0]) == []
