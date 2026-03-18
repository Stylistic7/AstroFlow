import ray
from pathlib import Path
from PIL import Image
import tempfile
import numpy as np
import pytest

from context import context, get_cache
from cache.cached_decorator import make_cached_decorator, Injected
from cache.types import NeverRun
from hashing import hash_call, hash_list
from cache.context_ref import ContextRef
from dependency_graph import Graph, run_graph, print_result_dict, print_graph
from testing.sidecar_hasher import SidecarHasher
from cache.content_hashed import ContentHashed
from decorators.content_hasher import ContentHasher
from ray.exceptions import ObjectFreedError

cached = make_cached_decorator(
    cache_provider=get_cache,
    content_hasher=ContentHasher(),
    call_hasher=hash_call,
)

sidecar_cached = make_cached_decorator(
    cache_provider=get_cache,
    content_hasher=SidecarHasher(),
    call_hasher=hash_call,
    debug=True,
)

_FAKE_SOURCE_CODE = "abc123"


def fake_source_hash_call(serialized_func: bytes, *args, **kwargs) -> str:
    kwargs_list = sorted(kwargs.items())
    return hash_list([_FAKE_SOURCE_CODE, *args, *kwargs_list])


fake_source_code_hashed = make_cached_decorator(
    cache_provider=get_cache,
    content_hasher=ContentHasher(),
    call_hasher=fake_source_hash_call,
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


@ray.remote
def graph_done() -> None:
    pass


@cached.remote
def make_small_image() -> np.ndarray:
    return np.ones((10, 10))


@fake_source_code_hashed.remote
def task_returns_list() -> list[ContentHashed]:
    return [
        ContentHashed(content_hash="a"),
        ContentHashed(content_hash="b"),
        ContentHashed(content_hash="c"),
    ]


@sidecar_cached.remote
def make_large_image_remote(
    shape: tuple[int, ...], output_dir: Injected[Path]
) -> Path:
    context.num_make_large_image_runs = context.num_make_large_image_runs + 1
    filename = output_dir.value / "foo.png"
    img = Image.new("L", shape, color=128)
    img.save(filename)
    print(f"Wrote image to filename {filename}")
    return filename


@sidecar_cached.remote
def consume_large_image_remote(path: Path) -> None:
    context.num_consume_large_image_runs = (
        context.num_consume_large_image_runs + 1
    )
    with Image.open(path) as img:
        print(f"img.size {img.size}")


@sidecar_cached.remote
def build_graph() -> Graph:
    context.num_build_graph_runs = context.num_build_graph_runs + 1
    return Graph({consume_large_image_remote: {make_large_image_remote}})


@sidecar_cached.remote
def build_mixed_graph() -> Graph:
    context.num_build_graph_runs = context.num_build_graph_runs + 1
    return Graph(
        {
            graph_done: {consume_large_image_remote},
            consume_large_image_remote: {make_large_image_remote},
        }
    )


def test_remote_functions(test_context) -> None:
    context.num_make_large_image_runs = 0
    context.num_consume_large_image_runs = 0

    with tempfile.TemporaryDirectory() as output_dir:
        injected_context = {"output_dir": Path(output_dir)}

        assert not make_large_image_remote.is_cached(
            (100, 100), injected_context=injected_context
        )

        # These functions should only run the first time.
        filename = ray.get(
            make_large_image_remote.remote(
                (100, 100), injected_context=injected_context
            )
        )
        ray.get(
            consume_large_image_remote.remote(
                filename, injected_context=injected_context
            )
        )

        assert isinstance(filename, Path)
        assert context.num_make_large_image_runs == 1
        assert context.num_consume_large_image_runs == 1

        assert make_large_image_remote.is_cached(
            (100, 100), injected_context=injected_context
        )

        # The second calls should just return cached results.
        filename = ray.get(
            make_large_image_remote.remote(
                (100, 100), injected_context=injected_context
            )
        )
        ray.get(
            consume_large_image_remote.remote(
                filename, injected_context=injected_context
            )
        )

        assert isinstance(filename, Path)
        assert context.num_make_large_image_runs == 1
        assert context.num_consume_large_image_runs == 1


def test_exception_handling(test_context) -> None:
    with pytest.raises(ray.exceptions.RayTaskError) as excinfo:
        ray.get(fail_remote.remote())

    assert "intentional failure" in str(excinfo.value)


def test_graph_caching(test_context) -> None:
    context.num_build_graph_runs = 0
    _ = ray.get(build_graph.remote())
    assert build_graph.is_cached()
    _ = ray.get(build_graph.remote())
    assert context.num_build_graph_runs == 1


def test_mixed_graph_caching(test_context) -> None:
    context.num_build_graph_runs = 0
    _ = ray.get(build_mixed_graph.remote())
    assert build_mixed_graph.is_cached()
    _ = ray.get(build_mixed_graph.remote())
    assert context.num_build_graph_runs == 1


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

    result, _ = run_graph(graph)
    print_result_dict(result)
    assert all(r == 10 for r in result.values())

    # Now, forcibly remove the object from the Ray object store. Running the same
    # graph again should still work, but it will force the ContextRef to reload
    # its contents from the cache.
    ray._private.internal_api.free([obj_ref.ref], local_only=False)
    with pytest.raises(ObjectFreedError):
        ray.get(obj_ref.ref)

    graph = foo(obj_ref)
    print_graph(graph)

    result, _ = run_graph(graph)
    print_result_dict(result)
    assert all(r == 10 for r in result.values())

    # Finally, running the graph a third time with an entirely different
    # obj_ref should pass that new value through the entire graph.
    obj_ref = ContextRef.store_large("bar is bigger")
    assert isinstance(obj_ref.get(), str)

    graph = foo(obj_ref)
    print_graph(graph)

    result, _ = run_graph(graph)
    print_result_dict(result)
    assert all(r == 13 for r in result.values())


def test_list_of_content_hashed_objects(test_context) -> None:
    result = ray.get(task_returns_list.remote())
    assert result[0].content_hash == hash_list([fake_source_hash_call("a"), 0])
    assert result[1].content_hash == hash_list([fake_source_hash_call("b"), 1])
    assert result[2].content_hash == hash_list([fake_source_hash_call("c"), 2])
