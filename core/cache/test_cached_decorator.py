from ray_infra import ray_local

import ray
import unittest
from pathlib import Path
from PIL import Image
import tempfile
from typing import Any

from context import context
from cache.cached_decorator import make_cached_decorator, Injected
from hashing import hash_call
from dependency_graph import Graph
from testing.sidecar_hasher import SidecarHasher

sidecar_hasher = SidecarHasher()
cached = make_cached_decorator(cache=context,
                               content_hasher=sidecar_hasher,
                               call_hasher=hash_call,
                               debug=True)


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
def make_graph_remote() -> Graph:
  context.num_graph_runs = context.num_graph_runs + 1
  return Graph()


class TestCachedDecorator(unittest.TestCase):

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
