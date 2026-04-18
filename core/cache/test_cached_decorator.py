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
    call_hasher=hash_call,
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


