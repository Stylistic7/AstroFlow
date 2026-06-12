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


def compute_image_statistics(image: np.ndarray) -> ImageStatistics:
    """Compute the full statistics block from a pixel array.

    `image` can be any shape and dtype. Multi-channel images are
    flattened across channels — these stats describe the overall
    brightness distribution, not per-channel.

    For a 50-megapixel float32 master this runs in ~50 ms (most of
    that is the `np.percentile` calls); for a 1-megapixel float32
    subframe, ~1 ms. The cost lives one-time per file inside
    `compute_image_metadata`'s sidecar-cached path, so it's paid only
    on cold ingestion.
    """
    # Float64 for stable percentile / mean / std computation regardless
    # of input dtype. Flatten to 1-D — channel breakdown isn't part
    # of this view.
    pixels = image.astype(np.float64, copy=False).ravel()

    min_value = float(pixels.min())
    max_value = float(pixels.max())
    mean_value = float(pixels.mean())

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
    histogram_counts = tuple(int(c) for c in counts)

    return ImageStatistics(
        histogram_bins=_HISTOGRAM_BINS,
        histogram_range=(min_value, max_value),
        histogram_counts=histogram_counts,
        p0_1=p0_1, p1=p1, p5=p5, p25=p25, p50=p50,
        p75=p75, p95=p95, p99=p99, p99_9=p99_9,
        sky_median=sky_median, sky_rms=sky_rms,
        clipped_low_fraction=clipped_low_fraction,
        clipped_high_fraction=clipped_high_fraction,
        min_value=min_value, max_value=max_value, mean_value=mean_value,
    )
