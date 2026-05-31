"""Per-camera hot-pixel fingerprinting.

The detector identifies single-pixel defects (hot pixels, stuck-low,
column-isolated bad pixels) from a single raw mono frame. The defect
SET — flat indices into the frame — is stable across re-captures by
the same physical sensor and statistically disjoint across different
sensors, even of the same model (≈ a·b/N ≈ 0.016 expected coincidences
between two 1000-pixel sets in a 61 MP IMX455).

This module covers Stage 1 of the five-stage pipeline (per-frame defect
extraction). Stages 2-4 (pairwise overlap → connected-components →
consensus EM) live in `core/scripts/cluster_cameras.py`; the live
ingestion path calls `extract_defects` only.

Algorithm:

  R(p) = I(p) − median₃ₓ₃(I)(p)          residual: hot pixel vs local bg
  M(p) = I(p) − max(8-neighbors of p)    isolation: hot pixel vs PSF wing

  candidate iff R(p) > k·1.4826·MAD(R_tile) AND M(p) > k·1.4826·MAD(M_tile)

  return top-K by R-strength as {flat_index: R_strength}

Tile-local MAD adapts the threshold to amp-glow corners (higher
dark-current shot noise) without per-camera tuning. Top-K caps the
return size so frames of different exposures/temperatures contribute
fingerprints of comparable size into Stages 2-4.

The detector deliberately rejects:
  * star cores (M ≤ 0 — PSF wings are also bright)
  * sky gradient (cancelled by R's 3×3 median)
  * amp glow (cancelled by tile-local MAD)
  * cosmic rays (random per-frame; Stage 4 consensus filters them out)
"""
from __future__ import annotations

import cv2
import numpy as np
from beartype import beartype


_TILE_SIZE = 256
_K_THRESHOLD = 6.0
_DEFAULT_TOP_K = 1000

# Tile MAD only needs noise statistics, not single-pixel resolution —
# subsampling the residual map by this stride before the MAD compute
# drops cost ~40× without changing the per-tile threshold (within ~5%
# vs the full-res computation on a typical 6200MM frame).
_MAD_STRIDE = 8

# 3x3 max kernel with the center zeroed → strict 8-neighbor max when
# passed to cv2.dilate (excludes self, which is required to detect
# single-pixel defects where self > all neighbors).
_CROSS_KERNEL = np.array(
    [[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8
)


@beartype
def extract_defects(
    image: np.ndarray,
    *,
    top_k: int = _DEFAULT_TOP_K,
    tile_size: int = _TILE_SIZE,
    k: float = _K_THRESHOLD,
) -> dict[int, float]:
    """Top-K single-pixel defect candidates from a raw mono frame.

    Returns {flat_index: R_strength} where flat_index = y * width + x.
    Caller is expected to supply a single-channel uint16 raw frame; any
    debayer, calibration, or cosmetic correction destroys the fingerprint
    signal. Returns empty dict for frames whose dimensions can't accommodate
    even a single tile (defensive; production frames are megapixel scale).
    """
    if image.ndim != 2:
        raise ValueError(
            f"extract_defects expects a 2D mono frame, got shape {image.shape}"
        )

    h, w = image.shape
    if h < tile_size or w < tile_size:
        return {}

    img32 = image.astype(np.float32, copy=False)

    # R: residual against 3x3 local median. The 3x3 median is the only
    # neighborhood that lets a SINGLE bright pixel be reliably excluded
    # from its own background estimate (the median of 9 values is the
    # 5th-largest, which is one of the 8 neighbors). Larger windows let
    # PSF wings leak into the bg estimate and kill the contrast.
    # cv2.medianBlur is ~280× faster than scipy.ndimage.median_filter for
    # 60 MP frames (5.8 s → 0.02 s).
    R = img32 - cv2.medianBlur(img32, 3)

    # M: per-pixel excess over the brightest 8-neighbor. A real PSF core
    # has bright neighbors; a single-pixel defect does not. cv2.dilate
    # with a 3x3 kernel whose center is zeroed gives a strict 8-neighbor
    # max (excluding self) — ~3× faster than the explicit shifted-pad
    # implementation.
    M = img32 - cv2.dilate(img32, _CROSS_KERNEL, borderType=cv2.BORDER_REPLICATE)

    # Tile-local MAD thresholds. Reshape into (tiles_y, tile_h, tiles_x,
    # tile_w) blocks so each tile gets its own scale; broadcast back to
    # per-pixel thresholds.
    R_thresh = _tile_mad_threshold(R, tile_size, k)
    M_thresh = _tile_mad_threshold(M, tile_size, k)

    candidates = (R > R_thresh) & (M > M_thresh)
    if not candidates.any():
        return {}

    ys, xs = np.where(candidates)
    strengths = R[ys, xs]

    # Top-K by R strength. argpartition is O(N) for a top-K cut.
    if strengths.size > top_k:
        cut = np.argpartition(-strengths, top_k - 1)[:top_k]
        ys = ys[cut]
        xs = xs[cut]
        strengths = strengths[cut]

    flat_indices = (ys.astype(np.int64) * w + xs.astype(np.int64))
    return {int(k): float(v) for k, v in zip(flat_indices, strengths)}


def _tile_mad_threshold(
    arr: np.ndarray, tile_size: int, k: float
) -> np.ndarray:
    """Per-pixel threshold = k * 1.4826 * MAD computed on a tile basis.

    The MAD only needs noise statistics, not single-pixel resolution, so
    we stride-decimate the input by _MAD_STRIDE before the median calls
    (a 60 MP frame becomes ~0.95 MP) and tile the result with a
    correspondingly smaller tile size. Empirically, this matches the
    full-res tile MAD within ~5% on real ASI6200MM frames while running
    ~40× faster.

    Output is per-pixel at the original input resolution, so the caller
    can compare directly against R/M.
    """
    if arr.shape[0] < tile_size or arr.shape[1] < tile_size:
        # Small frames — synthetic tests, debug. Use a simpler global MAD.
        mad = np.median(np.abs(arr - np.median(arr)))
        return np.full_like(arr, k * 1.4826 * mad)

    stride = _MAD_STRIDE
    if arr.shape[0] < tile_size * stride or arr.shape[1] < tile_size * stride:
        # Frame too small to subdivide after striding; fall back to a
        # global MAD on the strided view.
        sub = arr[::stride, ::stride]
        mad = np.median(np.abs(sub - np.median(sub)))
        return np.full_like(arr, k * 1.4826 * mad)

    sub = arr[::stride, ::stride]
    sub_tile = tile_size // stride
    h_sub, w_sub = sub.shape
    n_ty = h_sub // sub_tile
    n_tx = w_sub // sub_tile
    h_aligned = n_ty * sub_tile
    w_aligned = n_tx * sub_tile

    blocks = sub[:h_aligned, :w_aligned].reshape(
        n_ty, sub_tile, n_tx, sub_tile
    )
    tile_median = np.median(blocks, axis=(1, 3), keepdims=True)
    tile_mad = np.median(np.abs(blocks - tile_median), axis=(1, 3))
    tile_thresh = (k * 1.4826 * tile_mad).astype(arr.dtype, copy=False)

    # Broadcast (n_ty, n_tx) tile thresholds back to full per-pixel
    # resolution. Each tile covers tile_size×tile_size of the original
    # frame (sub_tile × sub_tile of the strided view, × stride).
    thresh_aligned_h = n_ty * tile_size
    thresh_aligned_w = n_tx * tile_size
    expanded = np.repeat(
        np.repeat(tile_thresh, tile_size, axis=0), tile_size, axis=1,
    )

    thresh = np.empty_like(arr)
    thresh[:thresh_aligned_h, :thresh_aligned_w] = expanded
    # Edge spillover: replicate the nearest tile's threshold along the
    # truncated remainder rows/cols at the bottom/right edges.
    if thresh_aligned_h < arr.shape[0]:
        thresh[thresh_aligned_h:, :thresh_aligned_w] = expanded[-1:, :]
    if thresh_aligned_w < arr.shape[1]:
        thresh[:, thresh_aligned_w:] = thresh[:, thresh_aligned_w - 1:thresh_aligned_w]
    return thresh
