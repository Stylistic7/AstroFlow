"""Synthetic test of the star-flat joint solve: a known response map, random
per-sub zero-points, a colour term, per-star noise, and one contaminated sub
(half its stars mis-matched by x0.3-x3). The solve must recover the map,
the colour term, and drop the contaminated sub."""
import numpy as np

from stacks.starflat import etalon_only, response_image, solve_star_flat, split_half

H, W = 640, 960          # small detector, same code paths
CELL = 64


def _truth(x, y):
    # vignetting bowl + a mid-scale ripple (the "etalon")
    r2 = ((x - W / 2) / W) ** 2 + ((y - H / 2) / H) ** 2
    return 1.0 - 0.08 * r2 + 0.006 * np.sin(x / 45.0) * np.cos(y / 37.0)


def _synth(seed=0, n_subs=24, stars_per_sub=1200, contaminated=(5,)):
    rng = np.random.default_rng(seed)
    rows = []
    c_true = 0.03
    for s in range(n_subs):
        x, y = rng.uniform(0, W, stars_per_sub), rng.uniform(0, H, stars_per_sub)
        G = rng.uniform(8, 12, stars_per_sub); col = rng.uniform(0.3, 2.0, stars_per_sub)
        K = rng.uniform(0.5, 2.0)
        scale = _truth(x, y) * K * np.exp(c_true * col) * np.exp(rng.normal(0, 0.03, stars_per_sub))
        if s in contaminated:
            bad = rng.random(stars_per_sub) < 0.5
            scale[bad] *= rng.uniform(0.3, 3.0, bad.sum())
        fwhm = 1.6 + rng.normal(0, 0.15, stars_per_sub)        # realistic scatter; a constant FWHM once broke the 97th-pct cut
        rows.append(np.column_stack([x, y, x * 0, y * 0, G, col, fwhm,
                                     scale * 1e4, scale, np.full(stars_per_sub, 100.0), np.full(stars_per_sub, s)]))
    return np.vstack(rows), c_true


def test_solve_recovers_response_and_drops_contaminated_sub():
    stars, c_true = _synth()
    sol = solve_star_flat(stars, (H, W), cell=CELL, min_stars_per_sub=50)
    assert sol.dropped_subs == [5], sol.dropped_subs
    assert abs(sol.colour_term - c_true) < 0.005, sol.colour_term
    # recovered log-gain map vs truth at cell centres
    ny, nx = sol.log_gain.shape
    yy, xx = np.mgrid[:ny, :nx] * CELL + CELL / 2
    truth = np.log(_truth(xx, yy)); truth -= np.nanmedian(truth)
    F = sol.log_gain - np.nanmedian(sol.log_gain)
    m = np.isfinite(F)
    r = np.corrcoef(F[m], truth[m])[0, 1]
    assert r > 0.95, r
    r_raw, r_blob = split_half(stars, sol)
    assert r_raw > 0.8, r_raw            # convergence metric is high and positive (0.85 on this synthetic set)


def test_response_image_sign_and_products():
    stars, _ = _synth(seed=1, contaminated=())
    sol = solve_star_flat(stars, (H, W), cell=CELL, min_stars_per_sub=50)
    resp = response_image(sol, smooth_sigma_cells=0.5)
    assert resp.shape == (H, W) and abs(float(np.median(resp)) - 1.0) < 1e-3
    # sign: the truth is BRIGHTER at centre than at the corners (vignetting); so must the response be
    assert resp[H // 2, W // 2] > resp[20, 20]
    et = etalon_only(resp, large_scale_px=200)
    assert et.shape == (H, W) and abs(float(np.median(et)) - 1.0) < 1e-3
    assert np.ptp(et) < np.ptp(resp)          # large scale removed
