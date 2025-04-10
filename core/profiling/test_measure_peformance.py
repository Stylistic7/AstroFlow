from ray_infra import ray_local

import ray
import unittest
from pathlib import Path
from PIL import Image
import tempfile
from typing import Any

from context import context
from task.task import make_task_decorator, Injected
from hashing import hash_list
from dependency_graph import Graph
from profiling.measure_performance import measure_performance


class SidecarHasher:

  def has(self, obj: Any) -> bool:
    return isinstance(obj, Path)

  def get(self, obj: Any) -> str:
    if isinstance(obj, Path):
      with open(obj.with_suffix('.hash'), 'r') as f:
        content_hash = f.read()
        print(f'got hash {content_hash}')
        return content_hash

  def set(self, obj: Any, content_hash: str) -> Any:
    print(f'setting hash {content_hash} for obj {obj}')
    if isinstance(obj, Path):
      with open(obj.with_suffix('.hash'), 'w') as f:
        f.write(content_hash)
    return obj


def _compute_content_hash(serialized_func: bytes, *args, **kwargs) -> str:
  kwargs_list = sorted(kwargs.items())
  return hash_list([serialized_func, *args, *kwargs_list])


sidecar_hasher = SidecarHasher()
task_with_context = make_task_decorator(cache=context,
                                        content_hasher=sidecar_hasher,
                                        call_hasher=_compute_content_hash,
                                        no_cache_types=[Graph])


@task_with_context.remote
@measure_performance(context)
def make_large_image_remote(shape: tuple[int],
                            output_dir: Injected[Path]) -> Path:
  context.num_make_large_image_runs = context.num_make_large_image_runs + 1
  filename = output_dir / "foo.png"
  img = Image.new('L', shape, color=128)
  img.save(filename)
  print(f'Wrote image to filename {filename}')
  return filename


class TestTask(unittest.TestCase):

  def test_decorator(self):
    from ray.util import get_node_ip_address

    context.num_make_large_image_runs = 0
    fn_name = "make_large_image_remote"
    node_ip = get_node_ip_address()
    perf_key = ('performance_history', node_ip, fn_name)

    context[perf_key] = []

    with tempfile.TemporaryDirectory() as output_dir:
      injected_context = {"output_dir": Path(output_dir)}
      ray.get(
          make_large_image_remote.remote((50, 50),
                                         injected_context=injected_context))

    history = context.get(perf_key, [])
    self.assertTrue(len(history) > 0)
    self.assertIsInstance(history[0], float)
