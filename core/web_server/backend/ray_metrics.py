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


def _try_import_ray_remote():
    """Define `_sample_node_psutil` lazily so the module imports cleanly
    in processes where Ray isn't available (tests, the SSE worker, etc.).
    Returns the `@ray.remote`-wrapped function or None on failure."""
    try:
        import ray

        @ray.remote(num_cpus=0.01)
        def _sample_node_psutil():
            import psutil
            # Blocking 0.5 s so the first call from a fresh worker
            # process returns accurate CPU% (the non-blocking variant
            # would return 0 the first time it's called).
            cpu = float(psutil.cpu_percent(interval=0.5))
            mem = float(psutil.virtual_memory().percent)
            return (cpu, mem)

        return _sample_node_psutil
    except Exception:
        return None


_sample_node_psutil = None  # populated on first sample_once call


def sample_once() -> None:
    """Sample EACH Ray node's psutil CPU / memory and aggregate by max.

    "Aggregate by max" answers "is any Mac in the cluster working
    hard right now?" — the same question you'd answer by glancing at
    Activity Monitor on each machine. Mean would hide a single hot
    node behind a cluster average; max keeps it honest.

    Implementation: fire a small `@ray.remote(num_cpus=0.01)` task per
    alive node, pinned via `NodeAffinitySchedulingStrategy`. Each task
    runs psutil locally on its host and returns the (cpu%, mem%) pair.
    The head collects, takes the max of each, writes to diskcache.

    Object-store sampling stays Ray-side — that's the one metric Ray
    reports honestly (it's the size of the plasma store relative to
    its budget, not a scheduled-vs-available game).

    Best-effort: any exception leaves the deques untouched."""
    global _sample_node_psutil
    try:
        import ray
        if _sample_node_psutil is None:
            _sample_node_psutil = _try_import_ray_remote()
        if _sample_node_psutil is None:
            return

        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        nodes = ray.nodes()
        alive = [n for n in nodes if n.get("Alive")]
        if not alive:
            return

        refs = []
        for n in alive:
            strategy = NodeAffinitySchedulingStrategy(
                node_id=n["NodeID"], soft=False,
            )
            refs.append(
                _sample_node_psutil.options(scheduling_strategy=strategy).remote()
            )
        # Per-ref ray.get so one bad node doesn't blank the whole sample.
        # Total wall time bounded by max(node sampling latency) + network —
        # the tasks block for 0.5 s in parallel, so a 2-node cluster is
        # still ~0.5 s + a bit.
        cpus, mems = [], []
        for ref in refs:
            try:
                cpu, mem = ray.get(ref, timeout=4.0)
                cpus.append(float(cpu))
                mems.append(float(mem))
            except Exception as e:
                print(f"### ray_metrics per-node sample failed: {type(e).__name__}: {e}")
        if cpus:
            _push_sample(_CPU_KEY, max(0.0, min(100.0, max(cpus))))
        if mems:
            _push_sample(_MEM_KEY, max(0.0, min(100.0, max(mems))))

        # Object-store sampling stays Ray-side.
        try:
            total = ray.cluster_resources() or {}
            avail = ray.available_resources() or {}
            total_obj = float(total.get("object_store_memory", 0))
            if total_obj > 0:
                used = total_obj - float(avail.get("object_store_memory", 0))
                obj = max(0.0, min(100.0, (used / total_obj) * 100.0))
                _push_sample(_OBJ_KEY, obj)
        except Exception:
            pass
    except Exception as e:
        print(f"### ray_metrics sample failed: {type(e).__name__}: {e}")


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


# ── Per-node MEASURED free RAM, for run_graph's admission gate ───────────────
# Distinct from the chip sampler above (which aggregates cpu%/mem% by max for a
# sparkline). This returns per-node *available bytes* so the scheduler can place
# a heavy task only on a node that actually has room — see
# dependency_graph.run_graph. Same NodeAffinity fan-out idiom, but keyed per node
# and with no CPU sleep.
_sample_node_mem_fn = None  # lazily-defined @ray.remote


def _try_import_node_mem_remote():
    """Define `_sample_node_mem` lazily so this module still imports in
    Ray-less processes. Returns the wrapped fn or None."""
    try:
        import ray

        # num_cpus=0: a measurement probe must never queue behind compute — on a
        # CPU-saturated node a fractional-CPU task would wait, and we'd lose
        # memory visibility exactly when the node is busiest. A 0-CPU pinned task
        # schedules immediately regardless of load.
        @ray.remote(num_cpus=0)
        def _sample_node_mem():
            import psutil
            import ray as _ray
            vm = psutil.virtual_memory()
            node_id = _ray.get_runtime_context().get_node_id()
            return (node_id, int(vm.available), int(vm.total))

        return _sample_node_mem
    except Exception:
        return None


def sample_node_memory(timeout: float = 4.0) -> dict:
    """Measure REAL free RAM on every alive Ray node.

    Fires one tiny task per node, pinned via `NodeAffinitySchedulingStrategy` so
    it runs ON that node — its `psutil` reading therefore reflects EVERY process
    there (PixInsight included), not just Ray's workers, which is the whole point
    of measuring instead of trusting Ray's declared reservations.

    Returns `{node_id: (available_bytes, total_bytes)}`. A node that fails or
    times out is simply omitted (per-ref, so one flaky node can't blank the
    batch); the caller drops its stale snapshot entry, so it falls out of
    scheduling until it answers again. Whole fan-out is milliseconds — unlike the
    chip sampler there is no 0.5 s CPU-percent sleep."""
    global _sample_node_mem_fn
    out: dict[str, tuple[int, int]] = {}
    try:
        import ray
        if _sample_node_mem_fn is None:
            _sample_node_mem_fn = _try_import_node_mem_remote()
        if _sample_node_mem_fn is None:
            return out

        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        alive = [n for n in ray.nodes() if n.get("Alive")]
        refs = []
        for n in alive:
            strategy = NodeAffinitySchedulingStrategy(node_id=n["NodeID"], soft=False)
            refs.append(_sample_node_mem_fn.options(scheduling_strategy=strategy).remote())
        for ref in refs:
            try:
                node_id, avail, total = ray.get(ref, timeout=timeout)
                out[node_id] = (int(avail), int(total))
            except Exception as e:
                print(f"### sample_node_memory per-node sample failed: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"### sample_node_memory failed: {type(e).__name__}: {e}")
    return out


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
