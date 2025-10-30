import ray
from pathlib import Path
from PIL import Image
import tempfile
import numpy as np
import pytest

from context import context, get_cache
from cache.cached_decorator import make_cached_decorator, Injected
from cache.types import NeverRun
from hashing import hash_call
from hashing.types import LargeObject
from dependency_graph import Graph, run_graph, print_result_dict
from testing.sidecar_hasher import SidecarHasher

sidecar_hasher = SidecarHasher()
cached = make_cached_decorator(cache_provider=get_cache,
                               content_hasher=sidecar_hasher,
                               call_hasher=hash_call,
                               debug=True)

@cached.remote
def make_large_array(shape: tuple[int]) -> LargeObject:
  arr = np.random.rand(*shape)
  return LargeObject(arr)

@cached.remote
def add_to_large_array(value: float, obj: LargeObject) -> LargeObject:
  arr = obj.get() + value
  return LargeObject(arr)

@cached.remote
def consume_large_array(obj: LargeObject) -> None:
  arr = obj.get()
  print(f'### arr.shape {arr.shape}')

@cached.remote
def build_large_array_graph(arr: LargeObject) -> Graph:
  return Graph({consume_large_array: {(add_to_large_array, 10, arr)}})


@cached.remote
def make_large_image_remote(shape: tuple[int],
                            output_dir: Injected[Path]) -> Path:
  context.num_make_large_image_runs = context.num_make_large_image_runs + 1
  filename = output_dir / "foo.png"
  img = Image.new('L', shape, color=128)
  img.save(filename)
  print(f'Wrote image to filename {filename}')
  return filename


@cached.remote
def consume_large_image_remote(path: Path) -> None:
  context.num_consume_large_image_runs = context.num_consume_large_image_runs + 1
  with Image.open(path) as img:
    print(f'img.size {img.size}')


@cached.remote
def fail_remote() -> int:
  raise ValueError("intentional failure")


@ray.remote
def graph_done() -> None:
  pass


@cached.remote
def build_graph() -> Graph:
  context.num_build_graph_runs = context.num_build_graph_runs + 1
  return Graph({consume_large_image_remote: {make_large_image_remote}})


@cached.remote
def build_mixed_graph() -> Graph:
  context.num_build_graph_runs = context.num_build_graph_runs + 1
  return Graph({
      graph_done: {consume_large_image_remote},
      consume_large_image_remote: {make_large_image_remote}
  })


@cached.remote
def make_small_image() -> np.ndarray:
  return np.ones((10, 10))


def test_remote_functions(test_context) -> None:
  context.num_make_large_image_runs = 0
  context.num_consume_large_image_runs = 0

  with tempfile.TemporaryDirectory() as output_dir:
    injected_context = {"output_dir": Path(output_dir)}

    assert not make_large_image_remote.is_cached(
        (100, 100), injected_context=injected_context)

    # These functions should only run the first time.
    filename = ray.get(
        make_large_image_remote.remote((100, 100),
                                       injected_context=injected_context))
    ray.get(
        consume_large_image_remote.remote(filename,
                                          injected_context=injected_context))

    assert isinstance(filename, Path)
    assert context.num_make_large_image_runs == 1
    assert context.num_consume_large_image_runs == 1

    assert make_large_image_remote.is_cached((100, 100),
                                             injected_context=injected_context)

    # The second calls should just return cached results.
    filename = ray.get(
        make_large_image_remote.remote((100, 100),
                                       injected_context=injected_context))
    ray.get(
        consume_large_image_remote.remote(filename,
                                          injected_context=injected_context))

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
  arr = ray.get(make_large_array.remote((1000, 1000)))
  graph = ray.get(build_large_array_graph.remote(arr))
  results, _ = run_graph(graph)
  print_result_dict(results)


  # The first time the graph is run, the cache is cold, and the LargeObjects
  # should contain live ray.ObjectRefs.
  # results, _ = run_graph(graph)
  # result = results[add_task]
  # assert type(result) is LargeObject
  # assert result.ref is not None
  # arr = result.get()
  # assert type(arr) is np.ndarray
  # assert arr.shape == shape

  # # The second time the graph is run, the cache is warm, and the LargeObjects
  # # should load their values into memory from the diskcache only when get() is
  # # called.
  # results, _ = run_graph(graph)
  # result = results[add_task]
  # assert type(result) is LargeObject
  # assert result.ref is None
  # arr = result.get()
  # assert type(arr) is np.ndarray
  # assert arr.shape == shape