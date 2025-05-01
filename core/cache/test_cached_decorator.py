from ray_infra import ray_local

import ray
import unittest
from pathlib import Path
from PIL import Image
import tempfile
from typing import Any

from context import context
from cache.cached_decorator import make_cached_decorator, Injected
from hashing import hash_list
from dependency_graph import Graph
from testing.sidecar_hasher import SidecarHasher


def _compute_content_hash(serialized_func: bytes, *args, **kwargs) -> str:
  kwargs_list = sorted(kwargs.items())
  return hash_list([serialized_func, *args, *kwargs_list])


sidecar_hasher = SidecarHasher()
task_with_context = make_cached_decorator(cache=context,
                                          content_hasher=sidecar_hasher,
                                          call_hasher=_compute_content_hash)


@task_with_context.remote
def make_large_image_remote(shape: tuple[int],
                            output_dir: Injected[Path]) -> Path:
  context.num_make_large_image_runs = context.num_make_large_image_runs + 1
  filename = output_dir / "foo.png"
  img = Image.new('L', shape, color=128)
  img.save(filename)
  print(f'Wrote image to filename {filename}')
  return filename


@task_with_context.remote
def consume_large_image_remote(path: Path) -> None:
  context.num_consume_large_image_runs = context.num_consume_large_image_runs + 1
  with Image.open(path) as img:
    print(f'img.size {img.size}')


@task_with_context.remote
def fail_remote() -> int:
  raise ValueError("intentional failure")


@task_with_context.remote
def make_graph_remote() -> Graph:
  context.num_graph_runs = context.num_graph_runs + 1
  return Graph()


class TestTask(unittest.TestCase):

  def test_remote_functions(self):
    context.num_make_large_image_runs = 0
    context.num_consume_large_image_runs = 0

    with tempfile.TemporaryDirectory() as output_dir:
      injected_context = {"output_dir": Path(output_dir)}

      self.assertFalse(
          make_large_image_remote.is_cached((100, 100),
                                            injected_context=injected_context))

      # These functions should only run the first time.
      filename = ray.get(
          make_large_image_remote.remote((100, 100),
                                         injected_context=injected_context))
      ray.get(
          consume_large_image_remote.remote(filename,
                                            injected_context=injected_context))

      self.assertIsInstance(filename, Path)
      self.assertEqual(context.num_make_large_image_runs, 1)
      self.assertEqual(context.num_consume_large_image_runs, 1)

      self.assertTrue(
          make_large_image_remote.is_cached((100, 100),
                                            injected_context=injected_context))

      # The second calls should just return cached results.
      filename = ray.get(
          make_large_image_remote.remote((100, 100),
                                         injected_context=injected_context))
      ray.get(
          consume_large_image_remote.remote(filename,
                                            injected_context=injected_context))

      self.assertIsInstance(filename, Path)
      self.assertEqual(context.num_make_large_image_runs, 1)
      self.assertEqual(context.num_consume_large_image_runs, 1)

  def test_exception_handling(self):
    with self.assertRaises(ray.exceptions.RayTaskError) as cm:
      ray.get(fail_remote.remote())

    self.assertIn("intentional failure", str(cm.exception))

  def test_excluded_from_cache(self):
    context.num_graph_runs = 0
    self.assertFalse(make_graph_remote.is_cached())

    ray.get(make_graph_remote.remote())
    self.assertFalse(make_graph_remote.is_cached())

    ray.get(make_graph_remote.remote())
    self.assertFalse(make_graph_remote.is_cached())

    # Should have run both times since Graph is excluded from cache
    self.assertEqual(context.num_graph_runs, 2)
