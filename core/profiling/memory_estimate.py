"""Profile-driven Ray memory estimation for @node(auto_memory=True).

`estimate_memory(fn, args, kwargs)` returns an int byte estimate
suitable for Ray's ``memory=`` resource. It buckets profile samples
by inferred input pixel count (sum of `width_pixels × height_pixels`
across every `ImageMetadata` in the call's args), then returns the
p95 of the matching bucket's recent peak-RSS samples.

The bucketing matters because the same op's RSS scales with input
size — `stack_files` on full-res frames consumes ~16 GB, on 8x
downscaled (1/64 the pixels) ~250 MB. Pre-split-op, the per-op
buckets blended both into one p95; now each pixel-size bucket
keeps a separate p95.

Buckets are log2-MP rounded down: 0 → <1 MP, 1 → 1–2 MP, 2 → 2–4 MP,
... 10 → 1024–2048 MP, etc. Samples in the same bucket are within
2× of each other in pixel count — a coarse-but-stable
classification that doesn't fragment the data.

When no samples in the matching bucket exist, falls back to a
per-op default. The estimate is capped at 0.9 × the largest node's
declared `memory` so the autoscaler can always at least try.
"""
import math
from typing import Any, Callable

import psutil

from hashing import hash_source_code

_GB = 1024 ** 3
_MAX_MEMORY_FRACTION = 0.9

# Per-op cold-start default (no profile samples yet). Keyed on op
# `__name__`: heavy ops (full-res WBPP / PhotometricMosaic) get 16 GB,
# everything else 4 GB. After the first sample lands in the matching
# pixel bucket, this default is bypassed.
_DEFAULT_MEMORY_BY_OP = {
    "stack_files":                     16 * _GB,
    "post_process_stack":              16 * _GB,
    "merge_target_channel":            16 * _GB,
    "store_mosaic_color_channel_xisf": 16 * _GB,
    "merge_images":                    16 * _GB,
}
_DEFAULT_MEMORY_UNKNOWN = 16 * _GB


def _pixel_count(args, kwargs) -> int:
    """Sum `width_pixels × height_pixels` across every ImageMetadata
    instance reachable in args/kwargs. Walks one level into tuples,
    lists, and dicts (so `masters=(ImageMetadata, ...)` data deps
    are caught). Returns 0 if no images are visible — falls back to
    op-name default in that case."""
    total = 0
    seen = set()

    def visit(v):
        nonlocal total
        if id(v) in seen:
            return
        seen.add(id(v))
        wp = getattr(v, "width_pixels", None)
        hp = getattr(v, "height_pixels", None)
        if isinstance(wp, int) and isinstance(hp, int):
            total += wp * hp
            return
        # Also accept WithSidecar-wrapped ImageMetadata: the .path or
        # .metadata might carry the dims.
        inner = getattr(v, "path", None)
        if inner is not None and inner is not v:
            visit(inner)
        if isinstance(v, dict):
            for x in v.values():
                visit(x)
        elif isinstance(v, (list, tuple)) and not isinstance(v, str):
            for x in v:
                visit(x)

    for v in (*args, *kwargs.values()):
        visit(v)
    return total


def pixel_bucket(pixels: int) -> int:
    """Map a pixel count to a log2-MP bucket label. Same-bucket
    samples are within 2× of each other in pixel count."""
    if pixels <= 0:
        return -1
    mp = pixels / 1_000_000
    if mp < 1:
        return 0
    return int(math.floor(math.log2(mp))) + 1


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered)) - 1)))
    return ordered[idx]


def _max_memory_bytes() -> int:
    """Cap auto-memory estimates at 0.9 × the largest node's declared
    `memory` resource.

    A request bigger than any node's capacity is fatal to the
    autoscaler ("no available node types can fulfill"); the cap keeps
    every estimate submittable. If a task genuinely needs more than
    0.9 × max-node memory, it'll OOM at runtime — much louder and
    more traceable than a silent stall waiting on a node that can
    never appear.

    Reads `ray.nodes()` to track joins/leaves. Falls back to the
    local machine's projected Ray memory (`total × 0.7`, matching
    `state/ray_control._ray_memory_bytes`) if Ray isn't initialized
    yet — keeps the cap meaningful during cold start.
    """
    try:
        import ray
        if ray.is_initialized():
            alive = [n for n in ray.nodes() if n.get("Alive")]
            mems = [int(n["Resources"].get("memory", 0)) for n in alive]
            mems = [m for m in mems if m > 0]
            if mems:
                return int(_MAX_MEMORY_FRACTION * max(mems))
    except Exception:
        pass
    return int(_MAX_MEMORY_FRACTION * psutil.virtual_memory().total * 0.7)


def _trim_high_outliers(values: list[float]) -> list[float]:
    """Drop high outliers via the Tukey fence (samples > Q3 + 1.5 × IQR).

    Profile samples are easily contaminated when other processes' RSS
    growth lands in the measurement window for a different task. A
    single 50 GiB sample for a function whose typical peak is 5 GiB
    poisons the p95 indefinitely (the bad sample will be the 95th
    percentile of any reasonably-small sample list). Trimming before
    the statistic catches the obvious bad ones non-parametrically.

    No-op below 5 samples — the IQR is too unstable to be useful, and
    a freshly-populated cache with one or two samples shouldn't have
    its only signal thrown away.
    """
    if len(values) < 5:
        return values
    ordered = sorted(values)
    q1 = ordered[len(ordered) // 4]
    q3 = ordered[3 * len(ordered) // 4]
    upper = q3 + 1.5 * (q3 - q1)
    return [v for v in values if v <= upper]


def estimate_memory(fn: Callable, args: tuple, kwargs: dict) -> int:
    """Return a bytes estimate for Ray scheduling of ``fn(*args, **kwargs)``.

    Buckets samples by input pixel count and reads p95 from the
    bucket matching this call. Falls back to an adjacent bucket if
    the exact one is empty (next-larger first, then next-smaller),
    and finally to the per-op default for cold start. Always capped
    at 0.9 × the largest node's declared `memory`.
    """
    from context import context

    fn_hash = hash_source_code(fn)
    cap = _max_memory_bytes()
    bucket = pixel_bucket(_pixel_count(args, kwargs))

    key = ("profile", fn_hash, fn.__name__)
    samples = context.get(key, [])

    # Group samples by their stored bucket; missing field implies a
    # legacy sample (pre-bucket recording) — treat as bucket -1
    # (unknown). Legacy samples are still useful as a last-resort
    # fallback when nothing better exists.
    #
    # Prefer the absolute "peak_rss_b" over the legacy
    # "peak_rss_delta_b" — see measure_performance.py for why deltas
    # underestimate in reused workers.
    by_bucket: dict[int, list[float]] = {}
    for s in samples:
        b = s.get("pixel_bucket", -1)
        rss = s.get("peak_rss_b", s.get("peak_rss_delta_b", 0))
        by_bucket.setdefault(b, []).append(rss)

    # Try this call's bucket, then nearby buckets (larger first —
    # over-estimating memory is safer than under).
    for candidate in [bucket, bucket + 1, bucket - 1, bucket + 2, bucket - 2, -1]:
        rsss = by_bucket.get(candidate)
        if rsss:
            rsss = _trim_high_outliers(rsss)
            rounded = int(math.ceil(_p95(rsss) / _GB) * _GB)
            return min(rounded, cap)

    default = _DEFAULT_MEMORY_BY_OP.get(fn.__name__, _DEFAULT_MEMORY_UNKNOWN)
    return min(default, cap)
