from ray_infra import ray_local

import os
import ray
import numpy as np
import unittest
from dependency_graph import run_graph

from decorators import cached
from products import makes_image, get_producer, MediaType
from compute_graph import build_compute_graph
from testing import make_test_context, copy_testdata_frames
from context import context
from file_io.types import TiffFilePath, XisfFilePath
from file_io.tiff_file_io import write_tiff_image
from stacks.composite_stack import CompositeStack
from targets.types import Target
from mosaics.types import Mosaic
from previews.types import QualityLevel

from products.registry import clear_registry

clear_registry(MediaType.IMAGE)


@makes_image(CompositeStack).as_subframe('preview_{name}.tiff')
@cached.remote
def stack_preview(stack: CompositeStack) -> TiffFilePath:
  '''Loads the best individual FITS image from stack and saves as TIFF.'''
  output_path = stack_preview.filename_fn(stack).resolve_at(context.output_dir)
  image = np.zeros((100, 100))
  write_tiff_image(output_path, image)

  return output_path


@makes_image(CompositeStack).as_master('master_{name}.tiff')
@cached.remote
def stack_files(stack: CompositeStack) -> TiffFilePath:
  '''Runs WBPP to stack raw images into a TIFF.'''
  output_path = stack_files.filename_fn(stack).resolve_at(context.output_dir)
  image = np.zeros((100, 100))
  write_tiff_image(output_path, image)

  return output_path


class TestRegistry(unittest.TestCase):

  def setUp(self) -> None:
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def tearDown(self) -> None:
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_single_stack(self) -> None:
    copy_testdata_frames(target_name='NGC7635', dest_dir=context.input_dirs[0])
    run_graph(build_compute_graph())
    stack = next(iter(context.stacks.values()))

    filename = ray.get(stack_files.remote(stack))
    self.assertTrue(os.path.exists(filename))

  def test_registry(self) -> None:
    copy_testdata_frames(target_name='NGC7635', dest_dir=context.input_dirs[0])
    run_graph(build_compute_graph())
    stack = next(iter(context.stacks.values()))

    fn = get_producer(MediaType.IMAGE, CompositeStack, QualityLevel.SUBFRAME)
    self.assertIsNotNone(fn)
    self.assertFalse(fn.is_cached(stack))

    filename = ray.get(fn.remote(stack))
    self.assertTrue(os.path.exists(filename))
    self.assertTrue(fn.is_cached(stack))
