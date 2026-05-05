"""Profile-driven Ray memory estimation for @node(auto_memory=True).

Exports a single function, ``estimate_memory(fn, args, kwargs)``, which
looks up recent peak-RSS samples in the context's profile cache and
returns an int byte estimate suitable for Ray's ``memory=`` resource.
Falls back to a quality-aware default when no samples are available
for the current ``(fn_hash, quality)`` — Master tasks get 16 GB,
DownscaledMaster/Subframe get 2 GB, unknown gets 4 GB.

The estimate is just a hint to Ray so it doesn't co-schedule N
heavy tasks before realizing they're heavy; once samples exist,
Ray's runtime memory tracking handles real availability. The
estimate is capped at 0.9× the largest node's declared `memory`,
because requests beyond any node's capacity make the autoscaler
reject them as "no node type can fulfill" (a permanent stall)
rather than queueing.
"""
import math
from typing import Any, Callable

import psutil

from hashing import hash_source_code
from profiling.measure_performance import _infer_quality

_GB = 1024 ** 3
_MAX_MEMORY_FRACTION = 0.9

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

    Looks up the function's recent peak-RSS samples in the profile cache,
    drops contaminated high outliers, and returns the p95 of what's
    left, rounded up to the nearest whole GB. Falls back to a
    quality-aware default when no samples exist for the current
    (fn_hash, quality). Always capped at 0.9 × the largest node's
    declared `memory` so the autoscaler can always at least try to
    schedule the result.
    """
    from context import context

    fn_hash = hash_source_code(fn)
    quality = _infer_quality(args, kwargs)

    key = ("profile", fn_hash, fn.__name__, quality)
    samples = context.get(key, [])

    rsss: list[float] = [s["peak_rss_delta_b"] for s in samples]

    cap = _max_memory_bytes()

    if not rsss:
        default = _DEFAULT_MEMORY_BY_QUALITY.get(quality, _DEFAULT_MEMORY_UNKNOWN)
        return min(default, cap)

    rsss = _trim_high_outliers(rsss)
    rounded = int(math.ceil(_p95(rsss) / _GB) * _GB)
    return min(rounded, cap)
