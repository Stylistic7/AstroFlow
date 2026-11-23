"""Unit tests for hot-pixel defect extraction.

These tests run on synthetic frames: a small image (large enough to
contain a few tiles) with a known set of seeded defects on a clean
noise background. They verify:

  1. seeded defects are recovered (high recall) and the strength column
     matches the seeded amplitude.
  2. a bright PSF (multi-pixel) does NOT register as a defect.
  3. the top-K cap limits the return size.
  4. raw vs. cosmetic-corrected frames behave as expected (corrected
     frames return ~empty).

Plus one full-resolution integration test against two real 6200MM
Orion frames in testdata/input — catches numerical edge cases at the
60 MP scale that the synthetic 512×768 cases miss (cv2.medianBlur
behavior over the full uint16 range, block reshape on a non-tile-
divisible (6388, 9576), cv2.dilate boundary behavior at frame edges).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.testdata  # reads core/testdata

from calibration.sensor_fingerprint import extract_defects
from file_io.generic import read_image
from file_io.types import FitsFilePath


def _noisy_background(
    h: int, w: int, *, mean: float = 1500.0, sigma: float = 25.0,
    seed: int = 0,
) -> np.ndarray:
    """Background noise levels approximate a real sky-limited frame: ~1500
    ADU sky offset with ~25 ADU shot+read noise after dark subtraction.
    A clean noise frame puts R-MAD around 25, so the k=6 MAD threshold
    sits at ~220 — close to what production frames produce.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(mean, sigma, size=(h, w)).clip(0, 65535).astype(np.uint16)


def _seed_defects(
    image: np.ndarray, positions: list[tuple[int, int]],
    *, amplitude: int = 4000,
) -> None:
    for y, x in positions:
        image[y, x] = min(65535, int(image[y, x]) + amplitude)


def _seed_star(
    image: np.ndarray, cy: int, cx: int,
    *, peak: int = 4000, sigma: float = 1.2,
) -> None:
    """Add a Gaussian PSF — bright neighbors should suppress the M signal."""
    radius = max(2, int(3 * sigma))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y, x = cy + dy, cx + dx
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                v = peak * np.exp(-(dy * dy + dx * dx) / (2 * sigma * sigma))
                image[y, x] = min(65535, int(image[y, x]) + int(v))


def _to_flat(positions: list[tuple[int, int]], w: int) -> set[int]:
    return {y * w + x for y, x in positions}


# Tile-divisible dimensions keep the test deterministic at the tile edges.
H, W = 512, 768


def test_seeded_defects_are_recovered():
    img = _noisy_background(H, W, seed=1)
    truth = [
        (30, 40), (50, 200), (100, 500), (200, 100), (300, 600),
        (350, 350), (400, 450), (450, 700), (480, 50), (490, 150),
    ]
    _seed_defects(img, truth, amplitude=4000)

    defects = extract_defects(img, top_k=200)
    recovered = set(defects.keys())
    expected = _to_flat(truth, W)

    # All seeded defects should be in the candidate set with comfortable margin.
    missing = expected - recovered
    assert not missing, f"Missed seeded defects: {missing}"


def test_broad_psf_star_is_not_a_defect():
    img = _noisy_background(H, W, seed=2)
    # A broad PSF (FWHM ≈ 5 px) — neighbors carry most of the flux, so M
    # collapses below threshold. Narrower PSFs CAN single-frame-resemble
    # defects and are removed by Stage 4 consensus (dithered star positions
    # don't recur across the cluster). This test covers the well-sampled
    # case that the per-frame primitive should kill on its own.
    _seed_star(img, cy=256, cx=384, peak=4000, sigma=4.0)

    defects = extract_defects(img, top_k=200)
    star_flat = 256 * W + 384
    assert star_flat not in defects, "Broad PSF center incorrectly flagged as defect"


def test_top_k_caps_return_size():
    img = _noisy_background(H, W, seed=3)
    truth = [(y, x) for y in range(20, 500, 30) for x in range(20, 700, 25)]
    _seed_defects(img, truth, amplitude=5000)

    defects = extract_defects(img, top_k=50)
    assert len(defects) <= 50


def test_clean_frame_returns_few_defects():
    img = _noisy_background(H, W, seed=4)
    defects = extract_defects(img, top_k=1000)
    # Pure Gaussian noise after MAD thresholding should yield at most a
    # few outliers; assert generously to avoid flakiness.
    assert len(defects) < 50, f"Clean frame produced {len(defects)} defects"


def test_strength_reflects_amplitude():
    img = _noisy_background(H, W, seed=5)
    weak = (100, 100)
    strong = (200, 200)
    _seed_defects(img, [weak], amplitude=2000)
    _seed_defects(img, [strong], amplitude=8000)

    defects = extract_defects(img, top_k=10)
    weak_strength = defects.get(weak[0] * W + weak[1])
    strong_strength = defects.get(strong[0] * W + strong[1])
    assert weak_strength is not None and strong_strength is not None
    assert strong_strength > weak_strength


def test_rejects_non_2d_input():
    img3d = np.zeros((100, 100, 3), dtype=np.uint16)
    with pytest.raises(ValueError):
        extract_defects(img3d)


def test_two_independent_seeded_sensors_disjoint():
    """Cross-camera overlap baseline. Two synthetic frames with disjoint
    defect sets should produce disjoint defect dicts."""
    rng = np.random.default_rng(42)
    pool = rng.choice(H * W, size=400, replace=False)
    sensor_a = [(int(p // W), int(p % W)) for p in pool[:200]]
    sensor_b = [(int(p // W), int(p % W)) for p in pool[200:]]

    img_a = _noisy_background(H, W, seed=10)
    img_b = _noisy_background(H, W, seed=11)
    _seed_defects(img_a, sensor_a, amplitude=5000)
    _seed_defects(img_b, sensor_b, amplitude=5000)

    defects_a = set(extract_defects(img_a, top_k=1000).keys())
    defects_b = set(extract_defects(img_b, top_k=1000).keys())

    overlap = defects_a & defects_b
    # Disjoint by construction; any overlap is pure noise from MAD-tail
    # outliers and should be tiny.
    assert len(overlap) <= 5, f"Unexpected cross-frame overlap: {len(overlap)}"


def test_irregular_dimensions_no_crash():
    """Frame shapes that aren't a clean multiple of the 256-tile size
    must still process cleanly. The optimized tile-MAD path uses block
    reshape on the largest aligned sub-region and replicates edge
    thresholds to the truncated remainder; this exercises that path."""
    img = _noisy_background(513, 769, seed=30)  # +1 row, +1 col past tile multiple
    truth = [(100, 100), (300, 500), (510, 760)]
    _seed_defects(img, truth, amplitude=5000)

    defects = extract_defects(img, top_k=100)
    expected = {y * 769 + x for y, x in truth}
    assert expected.issubset(set(defects.keys()))


def test_tile_local_threshold_adapts_to_local_noise():
    """A region with locally elevated noise (synthetic amp glow) should
    NOT spew false-positive defects — the tile-local MAD threshold
    scales with the local noise level. This guards against the
    optimization regressing into a global-MAD approximation that would
    pollute the candidate set near amp-glow corners.
    """
    img = _noisy_background(512, 768, seed=40)
    # Inject elevated noise in the top-left 256x256 tile, simulating amp glow.
    rng = np.random.default_rng(41)
    img[:256, :256] = img[:256, :256].astype(np.int32) + rng.integers(
        -200, 200, size=(256, 256)
    )
    img = np.clip(img, 0, 65535).astype(np.uint16)
    # Real defects only in the quiet region.
    truth = [(300, 100), (400, 500), (450, 700)]
    _seed_defects(img, truth, amplitude=5000)

    defects = set(extract_defects(img, top_k=200).keys())
    expected = {y * 768 + x for y, x in truth}
    assert expected.issubset(defects), "Defects missed in quiet region"

    # The noisy corner tile must contribute far fewer defects than its
    # area share (256*256 of 512*768 = 1/6 of pixels). If global MAD
    # were used, the noisy tile would either spam candidates (if global
    # MAD stayed low) or starve real ones (if it jumped). Tile-local MAD
    # gives both regions their own threshold.
    noisy_corner = {idx for idx in defects
                    if (idx // 768) < 256 and (idx % 768) < 256}
    assert len(noisy_corner) < 50, (
        f"Tile-local MAD failed: {len(noisy_corner)} defects in noisy corner"
    )


def test_same_sensor_two_frames_high_overlap():
    """Same seeded set across two noise realizations should produce
    near-identical fingerprints."""
    truth = [(30, 40), (50, 200), (100, 500), (200, 100), (300, 600),
             (350, 350), (400, 450), (450, 700), (480, 50), (490, 150),
             (60, 700), (120, 60), (180, 350), (250, 250), (380, 100)]

    img1 = _noisy_background(H, W, seed=20)
    _seed_defects(img1, truth, amplitude=5000)
    img2 = _noisy_background(H, W, seed=21)
    _seed_defects(img2, truth, amplitude=5000)

    d1 = set(extract_defects(img1, top_k=200).keys())
    d2 = set(extract_defects(img2, top_k=200).keys())

    expected = _to_flat(truth, W)
    # All truth pixels present in both, regardless of any noise outliers.
    assert expected.issubset(d1)
    assert expected.issubset(d2)


# ---------------------------------------------------------------------------
# Full-resolution integration test on real testdata frames.
# ---------------------------------------------------------------------------


_REAL_FRAME_A = Path(
    "testdata/input/Light_Orion_Pane 2_300.0s_Bin1_6200MM_H_20231222-220301_0001.fit"
)
_REAL_FRAME_B = Path(
    "testdata/input/Light_Orion_Pane 2_300.0s_Bin1_6200MM_H_20231222-221305_0003.fit"
)


@pytest.mark.skipif(
    not _REAL_FRAME_A.exists() or not _REAL_FRAME_B.exists(),
    reason="testdata input frames missing",
)
def test_full_resolution_real_frames_same_camera():
    """Two same-camera 60 MP Orion lights should produce defect sets
    with substantial overlap — the WHOLE POINT of the technique.

    Bounds rather than exact-equality so the test survives minor numeric
    refactors. A failure here is the canary for: cv2.medianBlur behaving
    differently at scale, block reshape mishandling the 6388×9576
    (non-tile-divisible) shape, or cv2.dilate boundary regression."""
    img_a = read_image(FitsFilePath(_REAL_FRAME_A))
    img_b = read_image(FitsFilePath(_REAL_FRAME_B))

    assert img_a.shape == (6388, 9576) and img_a.dtype == np.uint16
    assert img_b.shape == img_a.shape

    defects_a = extract_defects(img_a, top_k=1000)
    defects_b = extract_defects(img_b, top_k=1000)

    # Both frames must produce meaningful candidate sets. A regression
    # that bricks extraction would leave one or both empty.
    assert len(defects_a) >= 50, f"frame A defects: {len(defects_a)}"
    assert len(defects_b) >= 50, f"frame B defects: {len(defects_b)}"

    overlap = set(defects_a.keys()) & set(defects_b.keys())
    smaller = min(len(defects_a), len(defects_b))
    overlap_coef = len(overlap) / smaller

    # Combinatorial chance baseline for two 1000-pixel sets in 6388×9576
    # ≈ 1000²/61M ≈ 0.016 overlap_coef. Real same-camera overlap should
    # be dramatically higher; assert ≥ 0.20 with comfortable margin.
    assert overlap_coef >= 0.20, (
        f"Same-camera overlap_coef {overlap_coef:.3f} too low "
        f"(|A|={len(defects_a)}, |B|={len(defects_b)}, |A∩B|={len(overlap)})"
    )
