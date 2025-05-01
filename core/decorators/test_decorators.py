from ray_infra import ray_local

import ray
import numpy as np
import datetime
import unittest
import datetime
from pathlib import Path

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from testing import make_test_context, copy_testdata_frames
from file_io.types import TiffFilePath
from file_io.tiff_file_io import write_tiff_image
from stacks.composite_stack import CompositeStack
from decorators import cached, measure_performance, create_thumbnail

from products.registry import makes_image, clear_registry, MediaType

clear_registry(MediaType.IMAGE)


@makes_image(CompositeStack).as_master('master_{name}.tiff')
@cached.remote
@measure_performance
@create_thumbnail
def stack_files(stack: CompositeStack) -> TiffFilePath:
  '''Runs WBPP to stack raw images into a TIFF.'''
  output_path = TiffFilePath(stack_files.filename_fn(stack)).resolve_at(
      context.output_dir)
  image = np.zeros((100, 100))
  write_tiff_image(output_path, image)

  return output_path


class TestDecorators(unittest.TestCase):

  def setUp(self) -> None:
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def tearDown(self) -> None:
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_single_stack(self) -> None:
    copy_testdata_frames(prefix='light_run_0',
                         target_name='NGC7635',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 16, 29, 43),
                         dest_dir=context.input_dirs[0])

    run_graph(build_compute_graph())

    stacks = context.stacks
    self.assertEqual(len(stacks), 1)
    stack = next(iter(stacks.values()))

    filename = ray.get(stack_files.remote(stack))
    self.assertTrue(isinstance(filename, Path))

    thumbnail = stack_files.thumbnail_fn(stack)
    self.assertIsNotNone(thumbnail)
    self.assertTrue(isinstance(thumbnail, bytes))


if __name__ == "__main__":
  unittest.main()
