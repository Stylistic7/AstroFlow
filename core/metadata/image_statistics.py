"""Brightness distribution + sky stats over a full image.

Feeds `ImageMetadata.statistics`, which is the canonical "what does this
image's pixel distribution look like" record. Consumers include the
web UI's auto-stretch optimizer, the structure-emphasis LUT builder,
the cross-panel seam metric, the per-mosaic QA dashboard, and the
panel comparison view.

Why this lives at the metadata layer rather than per-producer: every
artifact in AstroFlow gets `compute_image_metadata` chained after it
(via `@makes_image`), so dropping the stats here makes every image
self-describing without touching any producer. Subframes, masters,
post-processed masters, placed panels, mosaic merges — the auto-
stretch UI works on all of them out of the same code path.

The statistics are computed in the data's NATIVE units; consumers map
to display units via the LUT. That preserves comparability across
panels with different SFC scaling and avoids hardcoding any "data
should be in [0, 1]" assumption that breaks the day SFC's output
range shifts.
"""
from __future__ import annotations

import numpy as np

from metadata.classes import ImageStatistics


_HISTOGRAM_BINS = 1024

# Lower-quartile cut for sky-pixel selection. Matches the threshold
# used by `stacks/post_processing/background_pedestal.py` so the
# sky_median / sky_rms reported here are literally the values the
# pedestal step works with — same definition, no consumer surprises.
_SKY_QUANTILE = 0.25

# Clipping detection thresholds: pixels within 0.5 % of the bottom or
# top of the data's actual range count as clipped. UI surfaces these
# as "this image has N% of pixels saturated; consider a shorter
# exposure" diagnostics.
_CLIP_FRACTION = 0.005

# `compute_image_metadata` chains this op after every artifact, including
# gigapixel produced masters/mosaics/placed panels. Above `_MAX_EXACT_PIXELS`
# the sort-based statistics (percentiles, median, MAD, sky, histogram) come
# from a random `_SAMPLE_PIXELS` subsample; the scalar reductions (max, mean,
# non-zero count) stay exact via cheap streaming passes. At or below the
# threshold the full array is used, so small images (subframes, ordinary
# masters) keep their exact, unchanged output. 64M pixels bounds the exact
# path's transient float64 copies to ~1.5 GB; a 12M sample keeps percentile
# error under ~0.1 %. `_SAMPLE_SEED` makes the subsample deterministic, so the
# same image always yields the same statistics.
_MAX_EXACT_PIXELS = 64_000_000
_SAMPLE_PIXELS = 12_000_000
_SAMPLE_SEED = 0


def compute_image_statistics(image: np.ndarray) -> ImageStatistics:
    """Compute the full statistics block from a pixel array.

    `image` can be any shape and dtype. Multi-channel images are
    flattened across channels — these stats describe the overall
    brightness distribution, not per-channel.

    Pixels exactly equal to 0.0 are treated as a "no data" sentinel
    and dropped from every downstream computation: mosaics have an
    irregular footprint inside a rectangular bounding box and the
    bbox padding is `place_image`-zero, while `post_process_stack`'s
    pedestal step shifts every real sky pixel to a positive value.
    Without this exclusion the histogram, percentiles, and sky stats
    would all be ~50 % dominated by "no data" and the auto-stretch
    LUT would waste its display brightness on the rectangular border
    instead of on the actual image content.

    Runs on inputs from 1-megapixel subframes to 2.3-gigapixel mosaic
    masters (the same op is chained after every artifact). At or below
    `_MAX_EXACT_PIXELS` the full array is used (a 50 MP master ~50 ms,
    exact); above it the sort-based stats come from a random subsample
    to bound memory (see `_MAX_EXACT_PIXELS`). The cost lives one-time
    per file inside `compute_image_metadata`'s sidecar-cached path, so
    it's paid only on cold ingestion.
    """
    # Flatten to a 1-D view — channel breakdown isn't part of this view.
    # For large images, estimate the sort-based stats from a subsample and
    # take the scalar reductions exactly over the full array (streaming, no
    # float64 copy); for small images, upcast the whole thing to float64 for
    # stable percentiles exactly as before. See `_MAX_EXACT_PIXELS`.
    flat = np.ravel(image)
    if flat.size > _MAX_EXACT_PIXELS:
        rng = np.random.default_rng(_SAMPLE_SEED)
        # Sorted indices so the gather walks memory monotonically
        # (prefetch-friendly) instead of randomly across a multi-GB array.
        idx = np.sort(rng.integers(0, flat.size, size=_SAMPLE_PIXELS))
        pixels = flat[idx].astype(np.float64)
        pixels = pixels[pixels != 0.0]
        full_max = float(flat.max())
        full_nonzero = int(np.count_nonzero(flat))
        full_mean = (
            float(flat.sum(dtype=np.float64) / full_nonzero)
            if full_nonzero else 0.0
        )
        # Rescale the sampled histogram to full-image counts so
        # sum(histogram_counts) still reflects the image's non-zero pixel
        # count, matching the old full-array behavior consumers may rely on.
        count_scale = (full_nonzero / pixels.size) if pixels.size else 0.0
    else:
        # Float64 for stable percentile / mean / std regardless of input
        # dtype. Bit-identical to the original implementation.
        pixels = flat.astype(np.float64, copy=False)
        pixels = pixels[pixels != 0.0]
        full_max = full_mean = None
        count_scale = 1.0
    if pixels.size == 0:
        # Degenerate: empty / all-sentinel image. Return zeros so
        # the sidecar still deserializes cleanly.
        return ImageStatistics(
            histogram_bins=_HISTOGRAM_BINS,
            histogram_range=(0.0, 1.0),
            histogram_counts=tuple([0] * _HISTOGRAM_BINS),
            p0_1=0.0, p1=0.0, p5=0.0, p25=0.0, p50=0.0,
            p75=0.0, p95=0.0, p99=0.0, p99_9=0.0,
            sky_median=0.0, sky_rms=0.0,
            clipped_low_fraction=0.0, clipped_high_fraction=0.0,
            min_value=0.0, max_value=0.0, mean_value=0.0,
        )

    min_value = float(pixels.min())
    max_value = full_max if full_max is not None else float(pixels.max())
    mean_value = full_mean if full_mean is not None else float(pixels.mean())

    # Single percentile call returns every cut we need; cheaper than
    # nine separate calls because the sort happens once.
    quantiles = np.percentile(
        pixels,
        [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9],
    )
    p0_1, p1, p5, p25, p50, p75, p95, p99, p99_9 = (float(q) for q in quantiles)

    # Sky pixels = lower-quartile cut. Median + std on that subset
    # give the same sky_median / sky_rms the pedestal step uses.
    sky_mask = pixels < p25
    if sky_mask.any():
        sky_pixels = pixels[sky_mask]
        sky_median = float(np.median(sky_pixels))
        sky_rms = float(np.std(sky_pixels))
    else:
        # Degenerate: every pixel equals p25 (e.g., constant-fill
        # synthetic tile). Treat sky stats as 0.
        sky_median = p25
        sky_rms = 0.0

    # MAD = median(|pixel − median(all)|). Matches the `mdev()` helper
    # inside `previews.operations.autostretch` so the JS port produces
    # an identical STF black-point. Computed on the full pixel array
    # (zeros already excluded above), at pixel-level precision —
    # stacked-pedestaled masters have sub-bin sky noise that a 1024-bin
    # histogram would round to 0.
    mad = float(np.median(np.abs(pixels - p50)))

    # Clipping fractions relative to the data's actual range. For
    # uint16 the range is typically [0, 65535] so clipped_low picks up
    # bias pixels — that's intentional, the UI uses this to detect
    # under-exposed frames.
    span = max_value - min_value
    if span > 0:
        low_threshold = min_value + _CLIP_FRACTION * span
        high_threshold = max_value - _CLIP_FRACTION * span
        clipped_low_fraction = float((pixels <= low_threshold).mean())
        clipped_high_fraction = float((pixels >= high_threshold).mean())
    else:
        # Constant image — every pixel is at min == max, both
        # "clipped" by definition. Report 1.0 honestly.
        clipped_low_fraction = 1.0
        clipped_high_fraction = 1.0

    # Histogram over the actual data range, 1024 bins. tuple-of-int
    # for hashability / frozen-dataclass cleanliness.
    counts, _ = np.histogram(
        pixels,
        bins=_HISTOGRAM_BINS,
        range=(min_value, max_value) if span > 0 else (min_value, min_value + 1.0),
    )
    histogram_counts = (
        tuple(int(c) for c in counts) if count_scale == 1.0
        else tuple(int(round(c * count_scale)) for c in counts)
    )

    return ImageStatistics(
        histogram_bins=_HISTOGRAM_BINS,
        histogram_range=(min_value, max_value),
        histogram_counts=histogram_counts,
        p0_1=p0_1, p1=p1, p5=p5, p25=p25, p50=p50,
        p75=p75, p95=p95, p99=p99, p99_9=p99_9,
        sky_median=sky_median, sky_rms=sky_rms,
        mad=mad,
        clipped_low_fraction=clipped_low_fraction,
        clipped_high_fraction=clipped_high_fraction,
        min_value=min_value, max_value=max_value, mean_value=mean_value,
    )
