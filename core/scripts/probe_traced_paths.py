"""Call traced_paths_for + traced_path_for directly against the live
context. Surfaces the actual exception (vs. Flask's opaque 500).

Usage:
    PYTHONPATH=. python -m scripts.probe_traced_paths mosaic <mosaic_id> "PixInsight Downscaled"
"""
import sys
import time
import traceback

from ray_infra.ray_init import connect_to_ray_cluster
from context import local_context as context
from cache.context_ref import use_local_context_for_refs


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    object_type, object_id, pipeline = sys.argv[1], sys.argv[2], sys.argv[3]
    item = (object_type, object_id, pipeline)

    connect_to_ray_cluster()
    use_local_context_for_refs()

    print(f"item = {item}")

    from dependency_graph.trace_paths import (
        traced_paths_for, _resolve_pipeline_call, _traced_paths,
    )
    from hashing import hash_source_code
    from hashing.hashing import hash_call
    from metadata.image_metadata import load_metadata
    import ray

    print("\n--- _resolve_pipeline_call ---")
    fn, entity, config = _resolve_pipeline_call(item)
    print(f"  fn={fn}")
    print(f"  entity stable_hash={getattr(entity, 'stable_hash', None)}")
    print(f"  config content_hash={getattr(config, 'content_hash', None)}")
    if fn is None:
        print("  -> resolve returned None; bailing")
        return

    pipeline_source_hash = hash_source_code(fn)
    print(f"  pipeline_source_hash={pipeline_source_hash}")

    print("\n--- load_metadata (called twice, content_hash should be stable) ---")
    for i in range(3):
        t0 = time.perf_counter()
        md = ray.get(load_metadata.remote())
        dt = time.perf_counter() - t0
        print(f"  call {i+1}: {dt*1000:.1f} ms, content_hash={md.content_hash}")
    metadata_ref = md

    args = ("PixInsight Downscaled", entity, config, metadata_ref, pipeline_source_hash)

    print("\n--- is_cached BEFORE any trace ---")
    print(f"  {_traced_paths.is_cached(*args)}")

    print("\n--- traced_paths_for (call 1) ---")
    t0 = time.perf_counter()
    paths1 = traced_paths_for(item)
    dt1 = time.perf_counter() - t0
    print(f"  took {dt1*1000:.1f} ms, returned {len(paths1)} entries")

    print(f"\n  is_cached AFTER call 1: {_traced_paths.is_cached(*args)}")

    print("\n--- traced_paths_for (call 2) ---")
    t0 = time.perf_counter()
    paths2 = traced_paths_for(item)
    dt2 = time.perf_counter() - t0
    print(f"  took {dt2*1000:.1f} ms, returned {len(paths2)} entries")

    print(f"\n  is_cached AFTER call 2: {_traced_paths.is_cached(*args)}")

    print("\n--- traced_paths_for (call 3) ---")
    t0 = time.perf_counter()
    paths3 = traced_paths_for(item)
    dt3 = time.perf_counter() - t0
    print(f"  took {dt3*1000:.1f} ms, returned {len(paths3)} entries")

    print(f"\nSummary: call1={dt1*1000:.1f}ms, call2={dt2*1000:.1f}ms, call3={dt3*1000:.1f}ms")
    print(f"\npath equality (call1 vs call2): "
          f"{paths1 == paths2}")
    print(f"path equality (call2 vs call3): "
          f"{paths2 == paths3}")

    # If is_cached says cached but call is still slow, the cache key is
    # different inside the Ray worker than what we computed here.
    # Compare the call_hash directly.
    from cache.cached_decorator import _bind_args
    bound = _bind_args(_traced_paths._function, args, {}, None)

    cache_key_narrow = getattr(_traced_paths, "_cache_key_narrow", None)
    fn_hash = getattr(_traced_paths, "_fn_hash", None)
    print(f"\n_traced_paths._fn_hash = {fn_hash}")
    print(f"_traced_paths._cache_key_narrow = {cache_key_narrow}")

    # Compute the expected call_hash ourselves
    fixed_args = bound.args
    if cache_key_narrow is not None:
        fixed_args, _ = cache_key_narrow(fixed_args, {})
    expected_call_hash = hash_call(fn_hash, *fixed_args)
    print(f"expected call_hash = {expected_call_hash}")
    expected_key = ("_traced_paths", expected_call_hash)
    print(f"expected cache key = {expected_key}")

    from context import get_cache
    cache = get_cache()
    print(f"key present in cache? {expected_key in cache}")


if __name__ == "__main__":
    main()
