"""1 Hz cluster-utilization sampler for the Ray status chip's
sparkline.

Runs as a gevent greenlet inside the SSE-stream gunicorn worker
(which is the only worker that loads `gunicorn_config.py:post_fork`)
and persists samples to diskcache via `local_context` so the OTHER
gunicorn worker — the main Flask app, which actually serves
`/api/ray/status` — can read them out across the process boundary.

The diskcache hit per write is tiny (3 × 60 floats × 8 B ≈ 1.5 KB
per second) — well under any reasonable budget.

History is intentionally short (60 samples at 1 Hz = the last
minute). The chip's job is "is the cluster busy or idle right now?",
not "graph the last hour". If the user wants deeper history, that
lives in a different surface.

Per-node real-time CPU is not exposed cheaply by Ray's public API —
`available_resources()` is cluster-wide, not per-node, and the
dashboard endpoint (`:8265/nodes`) requires the dashboard to be
enabled. The cluster-aggregate signal carries most of the
"is-anything-happening" weight for now; per-node breakdowns would
need either dashboard mode or a small psutil-sampler running on
each worker — both deferrable.
"""
from __future__ import annotations

import time
from typing import Any


_HISTORY_LEN = 60
_SAMPLE_INTERVAL_S = 1.0

# Diskcache keys. Three lockstep streams of last-60 samples.
_CPU_KEY = "ray_metrics_cpu_history"
_MEM_KEY = "ray_metrics_mem_history"
_OBJ_KEY = "ray_metrics_obj_history"


def _push_sample(key: str, value: float) -> None:
    """Append `value` to the rolling history at `key`, trim to the
    last `_HISTORY_LEN` samples, write back. Each write is a single
    diskcache transaction so concurrent readers see a consistent
    list (never a half-trimmed one)."""
    from context import local_context as context
    prev = context.get(key, None) or []
    if not isinstance(prev, list):
        prev = []
    prev.append(value)
    if len(prev) > _HISTORY_LEN:
        prev = prev[-_HISTORY_LEN:]
    context[key] = prev


def _read_history(key: str) -> list[float]:
    from context import local_context as context
    val = context.get(key, None)
    return val if isinstance(val, list) else []


def _utilization_pct(total: dict, avail: dict, key: str) -> float | None:
    """`(total - available) / total` as a 0-100 percentage. None when
    Ray hasn't told us about this resource (cluster not up yet, or
    the key just isn't tracked)."""
    total_val = float(total.get(key, 0))
    if total_val <= 0:
        return None
    used = total_val - float(avail.get(key, 0))
    return max(0.0, min(100.0, (used / total_val) * 100.0))


def sample_once() -> None:
    """Sample current cluster CPU / memory / object-store utilization
    and append to history.

    Memory and object-store numbers track Ray's SCHEDULED resources
    (what tasks have declared via `memory=...`), not actual RSS on
    each node. For most workloads that's a reliable proxy for
    pressure — Ray scheduling near 100% means no room for new tasks
    — but it isn't the same as "my Macs are about to swap". If we
    grow a real-RSS view, that's a separate per-node sampler.

    Best-effort: swallows every exception, so Ray-not-yet-initialized
    just produces no samples this tick. The deques keep their prior
    contents and the sparklines flatline visibly when the cluster
    goes away."""
    try:
        import ray
        total = ray.cluster_resources() or {}
        avail = ray.available_resources() or {}
        cpu = _utilization_pct(total, avail, "CPU")
        if cpu is not None:
            _push_sample(_CPU_KEY, cpu)
        mem = _utilization_pct(total, avail, "memory")
        if mem is not None:
            _push_sample(_MEM_KEY, mem)
        obj = _utilization_pct(total, avail, "object_store_memory")
        if obj is not None:
            _push_sample(_OBJ_KEY, obj)
    except Exception as e:
        import os
        if os.environ.get("ASTROFLOW_DEBUG_RAY_METRICS"):
            print(f"### ray_metrics sample failed: {e}")


def get_cpu_history() -> list[float]:
    """Snapshot the current CPU history. The diskcache read is on
    the hot path of `/api/ray/status`, but it's a single small list
    lookup — sub-millisecond."""
    return _read_history(_CPU_KEY)


def get_mem_history() -> list[float]:
    """Same shape as `get_cpu_history` but for scheduled cluster
    memory utilization."""
    return _read_history(_MEM_KEY)


def get_object_store_history() -> list[float]:
    """Same shape but for Ray's plasma object store. Worth its own
    series because object-store pressure is what triggers spilling
    to disk (slow) and downstream OOMs."""
    return _read_history(_OBJ_KEY)


def start_polling() -> None:
    """Spawn the 1 Hz sampler greenlet. Called from gunicorn's
    `post_fork` so it runs once per worker, after gevent's monkey
    patching is in place."""
    import gevent
    gevent.spawn(_poll_loop)


def _poll_loop() -> None:
    """Tight loop: sample, then `gevent.sleep` so other greenlets
    (notably the SSE announcer's `event_loop`) get scheduled. Also
    announces `ray_status` once per second so the frontend's SSE
    handler refetches `/api/ray/status` and the chip's sparkline
    keeps marching."""
    from web_server.backend.announcer import announce
    while True:
        sample_once()
        try:
            announce("ray_status")
        except Exception:
            pass
        import gevent
        gevent.sleep(_SAMPLE_INTERVAL_S)
