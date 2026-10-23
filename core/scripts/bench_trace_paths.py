"""Measure paths_for_cart_item wall-time against real context.

Usage (head node, with the live app's Ray cluster up):
    PYTHONPATH=. python -m scripts.bench_trace_paths

Picks the first cart item from context, or falls back to the first
(mosaic, "Siril + GraXpert + OSS Mosaic") pair if no cart entry exists.
Times 10 traces, reports min/median/max + path count.
"""
import time
import sys

from ray_infra.ray_init import connect_to_ray_cluster
from context import local_context as context
from cache.context_ref import use_local_context_for_refs
from dependency_graph.trace_paths import paths_for_cart_item


def _pick_item():
    cart = context.shopping_cart or []
    if cart:
        item = cart[0]
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            return tuple(item[:3])

    mosaics = context.mosaics or {}
    if mosaics:
        mosaic_id = next(iter(mosaics.keys()))
        return ("mosaic", mosaic_id, "Siril + GraXpert + OSS Mosaic")

    targets = context.targets or {}
    if targets:
        target_id = next(iter(targets.keys()))
        return ("target", target_id, "PixInsight")

    return None


def main():
    connect_to_ray_cluster()
    use_local_context_for_refs()

    item = _pick_item()
    if item is None:
        print("no cart item, mosaic, or target in context — nothing to trace")
        sys.exit(1)

    print(f"tracing item: {item}")

    # First read: snapshot hit (~0.1ms) if this item was built, else a cold
    # inline trace (recompute) if it's absent from the current graph.
    t0 = time.perf_counter()
    paths = paths_for_cart_item(item)
    cold = time.perf_counter() - t0
    print(f"first trace:  {cold*1000:.1f} ms ({len(paths)} paths)")

    # Subsequent reads: build-time snapshot hit (`context.get`), no trace.
    N = 5
    warm_times = []
    for i in range(N):
        t0 = time.perf_counter()
        paths = paths_for_cart_item(item)
        elapsed = time.perf_counter() - t0
        warm_times.append(elapsed)
        print(f"warm trace {i+1}/{N}: {elapsed*1000:.1f} ms ({len(paths)} paths)")

    warm_times.sort()
    print()
    print(f"first (cold or warm):  {cold*1000:.1f} ms")
    print(f"warm min/median/max:   {warm_times[0]*1000:.1f} / "
          f"{warm_times[N//2]*1000:.1f} / {warm_times[-1]*1000:.1f} ms")


if __name__ == "__main__":
    main()
