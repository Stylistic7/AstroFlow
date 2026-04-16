"""Profile-driven Ray memory estimation for @node(auto_memory=True).

Exports a single function, ``estimate_memory(fn, args, kwargs)``, which
looks up recent peak-RSS samples in the context's profile cache and
returns an int byte estimate suitable for Ray's ``memory=`` resource.
Falls back to a quality-aware default when no samples are available
for the current ``(fn_hash, quality)`` — Master tasks get 16 GB,
DownscaledMaster/Subframe get 2 GB, unknown gets 4 GB.
"""
import math
from typing import Any, Callable

from hashing import hash_source_code
from profiling.measure_performance import _infer_quality

_GB = 1024 ** 3
_HEADROOM_MULTIPLIER = 1.1

# Quality-aware defaults for the first run of a function (no profile
# samples yet). Based on observed peak-RSS distributions across the
# pipeline. Master/PostProcessed tasks are heavy; everything else is
# light. A blanket 32 GB default deadlocked scheduling because two
# cold-start tasks couldn't coexist on a 64 GB Mac.
_DEFAULT_MEMORY_BY_QUALITY = {
    "Master": 16 * _GB,
    "PostProcessed": 16 * _GB,
    "DownscaledMaster": 2 * _GB,
    "Subframe": 2 * _GB,
}
_DEFAULT_MEMORY_UNKNOWN = 4 * _GB


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered)) - 1)))
    return ordered[idx]


def estimate_memory(fn: Callable, args: tuple, kwargs: dict) -> int:
    """Return a bytes estimate for Ray scheduling of ``fn(*args, **kwargs)``.

    Looks up the function's recent peak-RSS samples in the profile cache
    and returns p95 × 1.1, rounded up to the nearest whole GB. Falls
    back to a quality-aware default when no samples exist for the current
    (fn_hash, quality).
    """
    from context import context

    fn_hash = hash_source_code(fn)
    quality = _infer_quality(args, kwargs)

    prefix = ("profile", fn_hash, fn.__name__, quality)
    matches = context.get_matching(prefix)

    rsss: list[float] = []
    for samples in matches.values():
        for sample in samples:
            rsss.append(sample["peak_rss_delta_b"])

    if not rsss:
        return _DEFAULT_MEMORY_BY_QUALITY.get(quality, _DEFAULT_MEMORY_UNKNOWN)

    estimate = _p95(rsss) * _HEADROOM_MULTIPLIER
    return int(math.ceil(estimate / _GB) * _GB)
