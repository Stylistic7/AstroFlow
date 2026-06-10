"""1 Hz cluster-utilization sampler for the Ray status chip's
sparkline.

Runs as a gevent greenlet inside the gunicorn worker — the same
process the Flask app serves from — so the `/api/ray/status`
endpoint can read the rolling history directly out of module-level
state without crossing a process or a diskcache boundary.

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

import collections
import time
from typing import Any


_HISTORY_LEN = 60
_SAMPLE_INTERVAL_S = 1.0


# Module-level ring buffers. Read concurrently by the Flask endpoint
# off the gevent event loop; the GIL plus gevent's cooperative
# scheduling means appends + slicing don't need an explicit lock.
_cpu_history: collections.deque[float] = collections.deque(maxlen=_HISTORY_LEN)


def sample_once() -> None:
    """Sample current cluster CPU utilization and append to history.

    Best-effort — swallows every exception. Ray not initialized,
    transient cluster glitches, etc. just produce no sample this
    tick; the deque keeps its prior contents and the sparkline
    visibly flatlines if the cluster goes away."""
    try:
        import ray
        total = ray.cluster_resources() or {}
        avail = ray.available_resources() or {}
        total_cpu = float(total.get("CPU", 0))
        if total_cpu <= 0:
            return
        used_cpu = total_cpu - float(avail.get("CPU", 0))
        pct = max(0.0, min(100.0, (used_cpu / total_cpu) * 100.0))
        _cpu_history.append(pct)
    except Exception as e:
        # Don't even log under normal "ray not yet initialized" —
        # noise dominates signal. Only log if the user wires
        # ASTROFLOW_DEBUG_RAY_METRICS.
        import os
        if os.environ.get("ASTROFLOW_DEBUG_RAY_METRICS"):
            print(f"### ray_metrics sample failed: {e}")


def get_cpu_history() -> list[float]:
    """Snapshot the current history as a plain list — what the
    Flask endpoint serializes into the JSON response. Returns the
    samples oldest-first; the sparkline renders left-to-right
    accordingly."""
    return list(_cpu_history)


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
