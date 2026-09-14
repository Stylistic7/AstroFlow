import ray
import numpy as np
import pytest

from context import context, get_cache
from cache.cached_decorator import make_cached_decorator
from cache.types import NeverRun
from hashing import hash_call
from cache.context_ref import ContextRef
from dependency_graph import Graph, run_graph, print_graph
from ray.exceptions import ObjectFreedError

cached = make_cached_decorator(
    cache_provider=get_cache,
)


@cached.remote
def bar(obj_ref) -> int:
    obj = obj_ref.get()
    return len(obj)


@cached.remote
def baz(obj_ref) -> int:
    obj = obj_ref.get()
    return len(obj)


def foo(obj_ref) -> Graph:
    return Graph({(bar, obj_ref): {(baz, obj_ref)}})


@cached.remote
def make_large_array(shape: tuple[int]) -> ContextRef:
    arr = np.random.rand(*shape)
    return ContextRef.store_large(arr)


@cached.remote
def add_to_large_array(value: float, obj: ContextRef) -> ContextRef:
    arr = obj.get() + value
    return ContextRef.store_large(arr)


@cached.remote
def consume_large_array(obj: ContextRef) -> None:
    arr = obj.get()
    print(f"### arr.shape {arr.shape}")


@cached.remote
def build_large_array_graph(arr: ContextRef) -> Graph:
    return Graph({consume_large_array: {(add_to_large_array, 10, arr)}})


@cached.remote
def fail_remote() -> int:
    raise ValueError("intentional failure")


@cached.remote
def make_small_image() -> np.ndarray:
    return np.ones((10, 10))


def test_exception_handling(test_context) -> None:
    with pytest.raises(ray.exceptions.RayTaskError) as excinfo:
        ray.get(fail_remote.remote())

    assert "intentional failure" in str(excinfo.value)


def test_dry_run(test_context) -> None:
    result = ray.get(make_small_image.remote(dry_run=True))
    assert result == NeverRun

    result = ray.get(make_small_image.remote(dry_run=False))
    assert isinstance(result, np.ndarray)


def test_large_object(test_context) -> None:
    # This first try will result in this large object being cached.
    obj_ref = ContextRef.store_large("foo is big")
    assert isinstance(obj_ref.get(), str)

    graph = foo(obj_ref)
    print_graph(graph)

    captured = {}
    def on_node_done(node, result, logs):
        captured[node.task._function.__name__] = result

    graph.node_status_fn = on_node_done
    run_graph(graph)
    assert all(r == 10 for r in captured.values())

    # Now, forcibly remove the object from the Ray object store. Running the same
    # graph again should still work, but it will force the ContextRef to reload
    # its contents from the cache.
    ray._private.internal_api.free([obj_ref.ref], local_only=False)
    with pytest.raises(ObjectFreedError):
        ray.get(obj_ref.ref)

    graph = foo(obj_ref)
    print_graph(graph)

    captured = {}
    graph.node_status_fn = on_node_done
    run_graph(graph)
    assert all(r == 10 for r in captured.values())

    # Finally, running the graph a third time with an entirely different
    # obj_ref should pass that new value through the entire graph.
    obj_ref = ContextRef.store_large("bar is bigger")
    assert isinstance(obj_ref.get(), str)

    graph = foo(obj_ref)
    print_graph(graph)

    captured = {}
    graph.node_status_fn = on_node_done
    run_graph(graph)
    assert all(r == 13 for r in captured.values())


def test_cache_key_narrow_hides_irrelevant_args(test_context) -> None:
    """`cache_key_narrow` lets a function declare which parts of its args
    actually affect its result. The function still receives the full
    args at runtime; only the cache key uses the narrowed view.
    """
    def narrow(args, kwargs):
        # Only the first arg (`subject`) feeds the cache key. The second
        # (`noise`) is dropped from hashing — different `noise` values
        # for the same `subject` should hit the cache.
        return (args[0],), {}

    @cached.remote(cache_key_narrow=narrow)
    def narrowed_f(subject, noise) -> int:
        return subject + noise

    # First call: cache miss, runs and stores under the narrowed key.
    assert ray.get(narrowed_f.remote(10, 100)) == 110

    # Same `subject`, different `noise`: with `narrow` dropping `noise`
    # from the key, this is a cache hit and returns the previously-stored
    # result (110), NOT 10+999=1009 — the function did not re-run.
    assert ray.get(narrowed_f.remote(10, 999)) == 110

    # `is_cached` confirms cache state at the narrowed-key level: the
    # original call was stored, and any (10, *) call hits.
    assert narrowed_f.is_cached(10, 100)
    assert narrowed_f.is_cached(10, 999)
    assert narrowed_f.is_cached(10, 0)
    # A different `subject` is a fresh key — not cached yet.
    assert not narrowed_f.is_cached(11, 100)


def test_cache_key_narrow_default_falls_back_to_full_args(test_context) -> None:
    """Without `cache_key_narrow`, the cache key is on the full args —
    different args means different keys (the existing behavior)."""
    @cached.remote
    def plain_f(a, b) -> int:
        return a + b

    ray.get(plain_f.remote(1, 2))
    assert plain_f.is_cached(1, 2)
    assert not plain_f.is_cached(1, 3)  # different `b` → fresh key


def test_filename_fn_and_cache_layer_share_canonical_call_hash(test_context) -> None:
    """The on-disk filename's call_hash MUST equal the cache layer's
    call_hash. Under the consolidated `@node` design they both close
    over the same `canonical_call_hash` closure that `@node` built, so
    drift is structurally impossible. This test asserts the structural
    invariant: every image-producer wrapper has `_fn_hash` set and its
    `filename_fn`'s output stem matches what the cache key would use.

    Regression for the bug found 2026-05-29 — the old design had two
    independent call_hash computations that silently disagreed.
    """
    from products import registry as _registry
    _registry._ensure_producers_imported()
    failures: list[str] = []
    seen: set[int] = set()
    for ops in _registry._producers.values():
        for op in ops:
            if id(op) in seen:
                continue
            seen.add(id(op))
            # @image_producer ops are non-Ray, non-cached lightweight
            # presentations (synchronous ndarray-returning). They have
            # neither a filename_fn nor a cache layer, so the
            # canonical_call_hash invariant doesn't apply.
            if not hasattr(op, "filename_fn"):
                continue
            if not hasattr(op, "_fn_hash"):
                failures.append(f"{op.__name__}: _fn_hash missing")
    assert not failures, "\n".join(failures)




# ---------------------------------------------------------------------------
# The cache handle stays in the process that resolved it
# ---------------------------------------------------------------------------

_PROVIDER_CALLS: list = []


def _counting_provider():
    """Module-level (so cloudpickle ships it by reference, like the real
    `_head_aware_cache`) and counting, so a re-resolution is observable."""
    _PROVIDER_CALLS.append(1)
    return get_cache()


def test_the_resolved_cache_handle_never_rides_in_the_pickled_task(test_context) -> None:
    """Ray pickles a decorated op's task body by value, closure included. A
    cache handle memoized in that closure went to every worker as a
    `diskcache.Cache` bound to the head's PATH — which a worker then opened
    on its own disk (2026-09-14: 170k @node results in a MacBook-local
    cache the head never saw). The handle must be resolved per process."""
    from ray import cloudpickle
    from ray.remote_function import RemoteFunction
    from cache import cached_decorator as cd

    dec = make_cached_decorator(cache_provider=_counting_provider)

    @dec.remote
    def echo(x: int) -> int:
        return x

    task = echo.options.__self__                          # the RemoteFunction Ray exports
    assert isinstance(task, RemoteFunction)
    body = task._function
    _PROVIDER_CALLS.clear()
    assert echo.cached_value(1) == (False, None)          # resolves the handle here
    assert len(_PROVIDER_CALLS) == 1

    blob = cloudpickle.dumps(body)
    assert get_cache().directory.encode() not in blob, "a diskcache path travelled with the task"
    assert b"diskcache" not in blob

    cd._RESOLVED.pop(_counting_provider, None)             # the receiving process starts empty
    assert cloudpickle.loads(blob)(2) == 2
    assert len(_PROVIDER_CALLS) == 2, "the unpickled body must resolve through the provider, not a shipped handle"
