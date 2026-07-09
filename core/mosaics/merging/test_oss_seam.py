"""Headless seam-flatness test for the OSS mosaic merger (`merge_images_oss`).

Feeds synthetic *placed* panels — already in the mosaic viewbox (full-canvas,
zero outside the panel, exactly what `place_image` emits) — straight into the
blend and checks the assembled result reads as one flat photograph across the
join. Pure numpy + the merge math: no PixInsight, no solve-field, no GraXpert,
so it runs anywhere (the headless replacement for the PhotometricMosaic apex
seam test).

The seam driver is a multiplicative transparency drift between panels — exactly
what `merge_images_oss`'s log-space flux least-squares (`_solve_flux_scales`,
clamped to [0.5, 2.0]) is meant to equalize. A negative control (naive
last-writer overlay, no equalization) confirms the merge is doing the work, not
that the input was already flat. Seam metric is shared with the PMM apex test
(`testing.seam_metrics`) so both judge seamlessness identically.
"""
from pathlib import Path

import numpy as np

from mosaics.merging.oss import _blend_many, _panel_bbox
from mosaics.placement.panel_io import write_placed_panel
from testing.seam_metrics import join_step, seam_column_profile


def _truth(height: int, width: int, rng, n_stars: int = 90) -> np.ndarray:
    """Shared ground-truth sky (flat background + point-source stars) that both
    panels image in their overlap — so a correct flux match makes them agree
    pixel-for-pixel there, and the only thing left to test is the merge."""
    img = np.full((height, width), 1000.0, np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    for _ in range(n_stars):
        cy, cx = rng.uniform(0, height), rng.uniform(0, width)
        img += rng.uniform(3000, 40000) * np.exp(
            -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.6 ** 2)))
    return img


def _panel(truth: np.ndarray, region, flux: float, rng) -> np.ndarray:
    """A placed panel: full-canvas zeros with `region` = truth·flux + shot-like
    noise. `flux` is the multiplicative transparency drift the merger equalizes."""
    height, width = truth.shape
    img = np.zeros((height, width), np.float32)
    r0, r1, c0, c1 = region
    sub = truth[r0:r1, c0:c1] * flux
    sub = sub + rng.normal(0, np.sqrt(np.maximum(sub, 1.0)), sub.shape)
    img[r0:r1, c0:c1] = np.clip(sub, 0.0, None).astype(np.float32)
    return img


def _write(tmp_path, name, arr) -> "Path":
    """Persist a full-canvas panel as bbox-on-disk — exactly what `place_image`
    now emits: the tight non-zero bbox pixels + their canvas offset in the
    header. `_blend_many` reads panel-sized arrays, not the full canvas."""
    p = tmp_path / f"{name}.tiff"
    r0, r1, c0, c1 = _panel_bbox(arr > 0)
    write_placed_panel(
        p, arr[r0:r1, c0:c1].astype(np.float32),
        offset_xy=(c0, r0),
        canvas_size=(arr.shape[1], arr.shape[0]),
    )
    return p


def test_merge_images_oss_equalizes_flux_drift_seamlessly(tmp_path):
    """Two panels with a 1.6x transparency drift (inside the [0.5, 2.0] flux
    clamp) — the classic seam driver — must blend into one flat image."""
    rng = np.random.default_rng(0x5EA11E55)
    height, width = 400, 800
    truth = _truth(height, width, rng)
    # ~25% overlap: A cols 0–500, B cols 300–800, join near x≈400.
    pa = _panel(truth, (0, height, 0, 500), flux=1.0, rng=rng)
    pb = _panel(truth, (0, height, 300, width), flux=1.6, rng=rng)
    fa = _write(tmp_path, "panel_a.fits", pa)
    fb = _write(tmp_path, "panel_b.fits", pb)

    merged = _blend_many([fa, fb]).astype(np.float64)
    js = join_step(seam_column_profile(merged))
    # Negative control: naive last-writer overlay, no flux equalization.
    naive = pa.copy()
    naive[pb > 0] = pb[pb > 0]
    jn = join_step(seam_column_profile(naive.astype(np.float64)))

    print(f"\n[oss-seam:2] merged join {js['step_over_rms']:.3f}x RMS "
          f"({js['step_adu']:.1f} ADU); naive {jn['step_adu']:.0f} ADU")

    assert np.isfinite(merged).all()
    assert (merged > 0).mean() > 0.95, "panels should tile the canvas"
    # Universal invariant: no visible seam at the join (measured 0.056x RMS).
    assert js["step_over_rms"] < 1.0, (
        f"visible seam after OSS merge: {js['step_over_rms']:.2f}x sky RMS")
    # The merge is genuinely doing the equalization — the un-corrected 1.6x
    # overlay's step is far larger than the merged one (measured ~500x).
    assert jn["step_adu"] > 30 * js["step_adu"], (
        f"control too weak (naive {jn['step_adu']:.0f} vs merged "
        f"{js['step_adu']:.1f} ADU) — drift isn't a real seam")


def test_merge_images_oss_three_panel_strip_seamless(tmp_path):
    """Variadic path: a 3-panel strip with drifts 1.0 / 1.4 / 0.7 and two
    overlaps must blend seamlessly (the global least-squares flux solve equalizes
    every overlapping pair at once, order-independent)."""
    rng = np.random.default_rng(0x3EA3)
    height, width = 400, 1000
    truth = _truth(height, width, rng, n_stars=110)
    regions = [(0, height, 0, 400), (0, height, 300, 700), (0, height, 600, width)]
    fluxes = [1.0, 1.4, 0.7]
    paths = [
        _write(tmp_path, f"p{i}.fits", _panel(truth, r, f, rng))
        for i, (r, f) in enumerate(zip(regions, fluxes))
    ]

    merged = _blend_many(paths).astype(np.float64)
    js = join_step(seam_column_profile(merged))
    print(f"\n[oss-seam:3] merged join {js['step_over_rms']:.3f}x RMS "
          f"({js['step_adu']:.1f} ADU)")

    assert np.isfinite(merged).all()
    assert (merged > 0).mean() > 0.95
    assert js["step_over_rms"] < 1.0, (
        f"visible seam across 3-panel strip: {js['step_over_rms']:.2f}x sky RMS")
