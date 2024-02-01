import ray_local

import unittest
import tempfile
import ray
from cache import cache

from make_test_frames import make_synthetic_frames, copy_testdata_frames
import ray_diskcache
from context import Context
import file_io

from image_metadata import scatter_compute_image_metadata
from spatial_association import find_components_by_cosine_similarity_ray
from stack_identification import identify_stacks_ray
from calibration_frames import associate_calibration_frames
from image_handling import scatter_stack_thumbnails


@ray.remote
def _build_compute_graph(new_filenames):
  context = Context()
  context.image_hashes = ray.put({})
  context.fits_headers = ray.put({})
  context.plate_solutions = ray.put({})

  context = scatter_compute_image_metadata.bind(context, new_filenames)
  context = find_components_by_cosine_similarity_ray.bind(context)
  context = identify_stacks_ray.bind(context)
  context = associate_calibration_frames.bind(context)
  context = scatter_stack_thumbnails.bind(context)

  return ray.workflow.continuation(context)


class TestCalibrationFrames(unittest.TestCase):

  def setUp(self):
    ray_diskcache.clear()
    cache.clear()

  def test_flats_from_different_camera(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      # There are lights flats from two different cameras. This should result
      # in two stacks, with mutually-exclusive sets of flat frames.
      #
      # The NGC7635 images were taken in October 2023, while the M42 images
      # were taken in November 2023. The stacks should be grouped in
      # chronological order.
      copy_testdata_frames(target_name='NGC7635', dest_dir=tmp_dir)
      make_synthetic_frames(number=5,
                            image_type='Dark',
                            fill_value=100,
                            instrument='ZWO ASI6200MM Pro',
                            width_pixels=9576,
                            height_pixels=6388,
                            temperature=-15,
                            dest_dir=tmp_dir)
      make_synthetic_frames(number=7,
                            image_type='Bias',
                            fill_value=50,
                            instrument='ZWO ASI6200MM Pro',
                            width_pixels=9576,
                            height_pixels=6388,
                            temperature=-15,
                            dest_dir=tmp_dir)
      make_synthetic_frames(number=11,
                            image_type='Flat',
                            fill_value=4096,
                            exposure=1,
                            instrument='ZWO ASI6200MM Pro',
                            width_pixels=9576,
                            height_pixels=6388,
                            temperature=-15,
                            dest_dir=tmp_dir)

      copy_testdata_frames(target_name='M42', dest_dir=tmp_dir)
      make_synthetic_frames(number=13,
                            image_type='Dark',
                            fill_value=100,
                            instrument='ZWO ASI2600MM Pro',
                            width_pixels=6248,
                            height_pixels=4176,
                            temperature=-20,
                            dest_dir=tmp_dir)
      make_synthetic_frames(number=17,
                            image_type='Bias',
                            fill_value=50,
                            instrument='ZWO ASI2600MM Pro',
                            width_pixels=6248,
                            height_pixels=4176,
                            temperature=-20,
                            dest_dir=tmp_dir)
      make_synthetic_frames(number=19,
                            image_type='Flat',
                            fill_value=4096,
                            exposure=1,
                            instrument='ZWO ASI2600MM Pro',
                            width_pixels=6248,
                            height_pixels=4176,
                            temperature=-20,
                            dest_dir=tmp_dir)

      filenames = file_io.find_files(tmp_dir)
      self.assertEqual(len(filenames), 78)

      context = ray.workflow.run(_build_compute_graph.bind(filenames))

      stacks = ray.get(context.stacks)
      self.assertEqual(len(stacks), 2)
      stack_iterator = iter(stacks)

      # TODO: Don't depend on the order of insertion here.
      stack = stacks[next(stack_iterator)]
      print('### First stack is', stack.target)
      self.assertEqual(len(stack.light_frames), 3)
      self.assertEqual(len(stack.dark_frames), 5)
      self.assertEqual(len(stack.bias_frames), 7)
      self.assertEqual(len(stack.flat_frames), 11)
      self.assertTrue(('stack_thumbnail_image', stack.id,
                       stack.content_hash) in cache)
      self.assertTrue(('stack_thumbnail_video', stack.id,
                       stack.content_hash) in cache)

      stack = stacks[next(stack_iterator)]
      print('### Second stack is', stack.target)
      self.assertEqual(len(stack.light_frames), 3)
      self.assertEqual(len(stack.dark_frames), 13)
      self.assertEqual(len(stack.bias_frames), 17)
      self.assertEqual(len(stack.flat_frames), 19)
      self.assertTrue(('stack_thumbnail_image', stack.id,
                       stack.content_hash) in cache)
      self.assertTrue(('stack_thumbnail_video', stack.id,
                       stack.content_hash) in cache)

  def test_dark_scaling(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      # The light frames are 180 seconds, but the dark frames are 60 or 300
      # seconds. AstroFlow should pick the closest darks that are equal to or
      # longer than the lights.
      copy_testdata_frames(target_name='NGC7635', dest_dir=tmp_dir)
      make_synthetic_frames(number=5,
                            image_type='Dark',
                            fill_value=100,
                            exposure=60,
                            instrument='ZWO ASI6200MM Pro',
                            width_pixels=9576,
                            height_pixels=6388,
                            temperature=-15,
                            dest_dir=tmp_dir)
      make_synthetic_frames(number=7,
                            image_type='Dark',
                            fill_value=100,
                            exposure=300,
                            instrument='ZWO ASI6200MM Pro',
                            width_pixels=9576,
                            height_pixels=6388,
                            temperature=-15,
                            dest_dir=tmp_dir)
      make_synthetic_frames(number=11,
                            image_type='Bias',
                            fill_value=50,
                            instrument='ZWO ASI6200MM Pro',
                            width_pixels=9576,
                            height_pixels=6388,
                            temperature=-15,
                            dest_dir=tmp_dir)
      make_synthetic_frames(number=13,
                            image_type='Flat',
                            fill_value=4096,
                            exposure=1,
                            instrument='ZWO ASI6200MM Pro',
                            width_pixels=9576,
                            height_pixels=6388,
                            temperature=-15,
                            dest_dir=tmp_dir)

      filenames = file_io.find_files(tmp_dir)
      self.assertEqual(len(filenames), 39)

      context = ray.workflow.run(_build_compute_graph.bind(filenames))

      stacks = ray.get(context.stacks)
      self.assertEqual(len(stacks), 1)

      stack = stacks[next(iter(stacks))]
      self.assertEqual(len(stack.light_frames), 3)
      self.assertEqual(len(stack.dark_frames), 7)
      self.assertEqual(len(stack.bias_frames), 11)
      self.assertEqual(len(stack.flat_frames), 13)

      self.assertTrue(('stack_thumbnail_image', stack.id,
                       stack.content_hash) in cache)
      self.assertTrue(('stack_thumbnail_video', stack.id,
                       stack.content_hash) in cache)

  def test_temperature_mismatch(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      # The light frames are at temperature -15, but the darks are at -18. This
      # difference is small enough for the frames to still be associated.
      copy_testdata_frames(target_name='NGC7635', dest_dir=tmp_dir)
      make_synthetic_frames(number=5,
                            image_type='Dark',
                            fill_value=100,
                            exposure=300,
                            instrument='ZWO ASI6200MM Pro',
                            width_pixels=9576,
                            height_pixels=6388,
                            temperature=-18,
                            dest_dir=tmp_dir)

      filenames = file_io.find_files(tmp_dir)
      self.assertEqual(len(filenames), 8)

      context = ray.workflow.run(_build_compute_graph.bind(filenames))

      stacks = ray.get(context.stacks)
      self.assertEqual(len(stacks), 1)

      stack = stacks[next(iter(stacks))]
      self.assertEqual(len(stack.light_frames), 3)
      self.assertEqual(len(stack.dark_frames), 5)
      self.assertEqual(len(stack.bias_frames), 0)
      self.assertEqual(len(stack.flat_frames), 0)

      self.assertTrue(('stack_thumbnail_image', stack.id,
                       stack.content_hash) in cache)
      self.assertTrue(('stack_thumbnail_video', stack.id,
                       stack.content_hash) in cache)


if __name__ == "__main__":
  unittest.main()
