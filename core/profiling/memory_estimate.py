"""Profile-driven Ray memory estimation for @node(auto_memory=True).

Exports a single function, ``estimate_memory(fn, args, kwargs)``, which
looks up recent peak-RSS samples in the context's profile cache and
returns an int byte estimate suitable for Ray's ``memory=`` resource.
Falls back to a conservative default (32 GB) when no samples are
available for the current ``(fn_hash, quality)`` — so the first task
after a code change serializes, and every task thereafter auto-sizes.
"""
import math
from typing import Any, Callable

from hashing import hash_source_code
from profiling.measure_performance import _infer_quality

_GB = 1024 ** 3
DEFAULT_MEMORY = 32 * _GB
# No headroom multiplier. peak_rss_delta_b is a direct measurement of
# what the task needed; padding it on top of Ray's 30% physical-RAM
# reserve double-counts the safety margin and blocks tasks that
# actually fit. If a task's true peak later creeps above the p95, the
# next run records it and the estimate tracks — no buffer required.
_HEADROOM_MULTIPLIER = 1.0


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered)) - 1)))
    return ordered[idx]


def estimate_memory(fn: Callable, args: tuple, kwargs: dict) -> int:
    """Return a bytes estimate for Ray scheduling of ``fn(*args, **kwargs)``.

    Looks up the function's recent peak-RSS samples in the profile cache
    and returns p95 * 1.3, rounded up to the nearest whole GB. Returns
    ``DEFAULT_MEMORY`` (32 GB) when no samples exist for the current
    (fn_hash, quality) — which happens on cold start and after every
    code change to ``fn``.
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
        return DEFAULT_MEMORY

    estimate = _p95(rsss) * _HEADROOM_MULTIPLIER
    return int(math.ceil(estimate / _GB) * _GB)
