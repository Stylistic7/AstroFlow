from ray_infra import ray_local

import ray
import unittest
from pathlib import Path
from PIL import Image
import tempfile
import time

from context import context
from task.task import make_task_decorator, Injected
from hashing import hash_list
from dependency_graph import Graph


class SidecarHash:

  def get(self, path: Path) -> str:
    with open(path.with_suffix('.hash'), 'r') as f:
      content_hash = f.read()
      print(f'got hash {content_hash}')
      return content_hash

  def set(self, path: Path, content_hash: str):
    print(f'setting hash {content_hash} for path {path}')
    with open(path.with_suffix('.hash'), 'w') as f:
      f.write(content_hash)


def _compute_content_hash(serialized_func: bytes, *args, **kwargs) -> str:
  kwargs_list = sorted(kwargs.items())
  return hash_list([serialized_func, *args, *kwargs_list])


sidecar = SidecarHash()
task_with_context = make_task_decorator(context,
                                        sidecar,
                                        call_hasher=_compute_content_hash,
                                        no_cache_types=[Graph])


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


class TestTask(unittest.TestCase):

  def test_remote_functions(self):
    context.num_make_large_image_runs = 0
    context.num_consume_large_image_runs = 0

    with tempfile.TemporaryDirectory() as output_dir:
      injected_context = {"output_dir": Path(output_dir)}

      # These functions should only run the first time.
      filename = ray.get(
          make_large_image_remote.remote((100, 100),
                                         injected_context=injected_context))
      print(f'### context contents 0 {context.get_cache_keys()}')
      ray.get(
          consume_large_image_remote.remote(filename,
                                            injected_context=injected_context))
      print(f'### context contents 1 {context.get_cache_keys()}')

      self.assertIsInstance(filename, Path)
      self.assertEqual(context.num_make_large_image_runs, 1)
      self.assertEqual(context.num_consume_large_image_runs, 1)

      # The second calls should just return cached results.
      filename = ray.get(
          make_large_image_remote.remote((100, 100),
                                         injected_context=injected_context))
      print(f'### context contents 2 {context.get_cache_keys()}')
      ray.get(
          consume_large_image_remote.remote(filename,
                                            injected_context=injected_context))
      print(f'### context contents 3 {context.get_cache_keys()}')

      self.assertIsInstance(filename, Path)
      self.assertEqual(context.num_make_large_image_runs, 1)
      self.assertEqual(context.num_consume_large_image_runs, 1)

  # def test_update_performance_actor_latency(self):
  #   fn_name = "expensive_task"
  #   n = 1000
  #   samples = []

  #   # Clear any previous performance data
  #   context[("performance_history", "127.0.0.1", fn_name)] = []

  #   for _ in range(n):
  #     t0 = time.perf_counter()
  #     _update_performance(fn_name, 0.123, context)
  #     t1 = time.perf_counter()
  #     samples.append(t1 - t0)

  #   avg_time = sum(samples) / len(samples)
  #   worst_time = max(samples)
  #   best_time = min(samples)

  #   print(f"\n_update_performance() over {n} calls:")
  #   print(f"  average: {avg_time:.6f} sec")
  #   print(f"  best:    {best_time:.6f} sec")
  #   print(f"  worst:   {worst_time:.6f} sec")

  #   # You can set a generous upper bound depending on expectations
  #   self.assertLess(avg_time, 0.01, "Average performance tracking too slow")


# TODO: Add tests of exception handling.

# TODO: Add tests of excluding Graphs

# TODO: Add tests of performance measurement.
