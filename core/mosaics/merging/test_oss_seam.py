"""Headless seam-flatness test for the OSS mosaic merger (`merge_images_oss`).

Feeds synthetic *placed* panels — already in the mosaic viewbox (full-canvas,
zero outside the panel, exactly what `place_image` emits) — straight into the
blend and checks the assembled result reads as one flat photograph across the
join. Pure numpy + the merge math: no PixInsight, no solve-field, no GraXpert.

CONTRACT (2026-09-01): panels arrive GAIN-TRUE — multiplicative calibration
happens upstream, absolutely, per master, against Gaia (post_stack_calibrate); the
merge's overlap flux solve was deleted with it. What the merge still owns is
the ADDITIVE mismatch: per-panel sky offsets, removed by the overlap level
solve (`_overlap_offsets` + `_solve_level_offsets`). The seam driver here is
therefore an injected OFFSET difference, and a negative control (naive
last-writer overlay) confirms the solve is doing the work. Seam metric shared
with the apex test (`testing.seam_metrics`).
"""
from pathlib import Path

import numpy as np

from mosaics.merging.oss import _blend_many, _panel_bbox
from mosaics.placement.panel_io import write_placed_panel
from testing.seam_metrics import join_step, seam_column_profile


def _truth(height: int, width: int, rng, n_stars: int = 90) -> np.ndarray:
    """Shared ground-truth sky (pedestal + point-source stars) that both
    panels image in their overlap — so with offsets solved they agree
    pixel-for-pixel there, and the only thing left to test is the merge."""
    img = np.full((height, width), 0.005, np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    for _ in range(n_stars):
        cy, cx = rng.uniform(0, height), rng.uniform(0, width)
        img += rng.uniform(0.01, 0.2) * np.exp(
            -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.6 ** 2)))
    return img


def _panel(truth: np.ndarray, region, offset: float, rng) -> np.ndarray:
    """A placed panel: full-canvas zeros with `region` = truth + offset +
    noise. `offset` is the additive sky-glow difference the level solve
    equalizes (gains arrive pre-calibrated — see module doc)."""
    height, width = truth.shape
    img = np.zeros((height, width), np.float32)
    r0, r1, c0, c1 = region
    sub = truth[r0:r1, c0:c1] + offset
    sub = sub + rng.normal(0, 5e-5, sub.shape)
    img[r0:r1, c0:c1] = np.clip(sub, 1e-6, None).astype(np.float32)
    return img


def _write(tmp_path, name, arr) -> "Path":
    """Persist a full-canvas panel as bbox-on-disk — exactly what
    `place_image` emits: the tight non-zero bbox pixels + their canvas
    offset in the header."""
    p = tmp_path / f"{name}.tiff"
    r0, r1, c0, c1 = _panel_bbox(arr > 0)
    write_placed_panel(
        p, arr[r0:r1, c0:c1].astype(np.float32),
        offset_xy=(c0, r0),
        canvas_size=(arr.shape[1], arr.shape[0]),
    )
    return p


def test_merge_images_oss_equalizes_offset_drift_seamlessly(tmp_path):
    """Two panels with a Cygnus-scale sky-offset difference (1.2e-4 on the
    0.005 pedestal) must blend into one flat image under the level solve."""
    rng = np.random.default_rng(0x5EA11E55)
    height, width = 400, 800
    truth = _truth(height, width, rng)
    # ~25% overlap: A cols 0-500, B cols 300-800, join near x~400.
    pa = _panel(truth, (0, height, 0, 500), offset=0.0, rng=rng)
    pb = _panel(truth, (0, height, 300, width), offset=1.2e-4, rng=rng)
    fa = _write(tmp_path, "panel_a", pa)
    fb = _write(tmp_path, "panel_b", pb)

    merged = _blend_many([fa, fb], level_solve="overlap").astype(np.float64)
    js = join_step(seam_column_profile(merged))
    # Negative control: naive last-writer overlay, no offset equalization.
    naive = pa.copy()
    naive[pb > 0] = pb[pb > 0]
    jn = join_step(seam_column_profile(naive.astype(np.float64)))

    print(f"\n[oss-seam:2] merged join {js['step_over_rms']:.3f}x RMS; "
          f"naive {jn['step_over_rms']:.1f}x RMS")

    assert np.isfinite(merged).all()
    assert (merged > 0).mean() > 0.95, "panels should tile the canvas"
    # Universal invariant: no visible seam at the join.
    assert js["step_over_rms"] < 1.0, (
        f"visible seam after OSS merge: {js['step_over_rms']:.2f}x sky RMS")
    # The solve is genuinely doing the work — the uncorrected overlay's
    # step is far larger than the merged one.
    assert jn["step_over_rms"] > 3 * js["step_over_rms"], (
        f"control too weak (naive {jn['step_over_rms']:.2f}x vs merged "
        f"{js['step_over_rms']:.2f}x) — drift isn't a real seam")


def test_merge_images_oss_three_panel_strip_seamless(tmp_path):
    """Variadic path: a 3-panel strip with distinct offsets and two overlaps
    must blend seamlessly (the global least-squares equalizes every
    overlapping pair at once, order-independent, loop-closure pinned)."""
    rng = np.random.default_rng(0x3EA3)
    height, width = 400, 1000
    truth = _truth(height, width, rng, n_stars=110)
    regions = [(0, height, 0, 400), (0, height, 300, 700),
               (0, height, 600, width)]
    offsets = [0.0, 1.3e-4, -0.8e-4]
    paths = [
        _write(tmp_path, f"p{i}", _panel(truth, r, o, rng))
        for i, (r, o) in enumerate(zip(regions, offsets))
    ]

    merged = _blend_many(paths, level_solve="overlap").astype(np.float64)
    js = join_step(seam_column_profile(merged))
    print(f"\n[oss-seam:3] merged join {js['step_over_rms']:.3f}x RMS")

    assert np.isfinite(merged).all()
    assert (merged > 0).mean() > 0.95
    assert js["step_over_rms"] < 1.0, (
        f"visible seam across 3-panel strip: {js['step_over_rms']:.2f}x "
        f"sky RMS")
