import unittest
from pathlib import Path
from PIL import Image
import tempfile

from task.task import make_task_decorator, Injected


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


cache = {}
sidecar = SidecarHash()
task = make_task_decorator(cache, sidecar)

num_make_large_image_runs = 0
num_consume_large_image_runs = 0


@task
def make_large_image(shape: tuple[int], output_dir: Injected[Path]) -> Path:
  global num_make_large_image_runs
  num_make_large_image_runs += 1
  filename = output_dir / "foo.png"
  img = Image.new('L', shape, color=128)
  img.save(filename)
  print(f'Wrote image to filename {filename}')
  return filename


@task
def consume_large_image(path: Path) -> None:
  global num_consume_large_image_runs
  num_consume_large_image_runs += 1
  with Image.open(path) as img:
    print(f'img.size {img.size}')


class TestTask(unittest.TestCase):

  def test_caching(self):
    with tempfile.TemporaryDirectory() as output_dir:
      context = {"output_dir": Path(output_dir)}

      # These functions should only run the first time.
      filename = make_large_image((100, 100), context=context).unwrap()
      consume_large_image(filename, context=context)

      self.assertEqual(num_make_large_image_runs, 1)
      self.assertEqual(num_consume_large_image_runs, 1)

      # The second calls should just return cached results.
      filename = make_large_image((100, 100), context=context).unwrap()
      consume_large_image(filename, context=context)

      self.assertEqual(num_make_large_image_runs, 1)
      self.assertEqual(num_consume_large_image_runs, 1)
