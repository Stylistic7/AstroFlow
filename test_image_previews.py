import ray_local

import datetime
import unittest
import datetime

from testing.make_test_frames import make_synthetic_frames, copy_testdata_frames
from context import context
from compute_graph import build_compute_graph
from dependency_graph import run_graph
from file_io.find_image_files import find_image_files
from testing.make_test_context import make_test_context


class TestImagePreviews(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()
    context.regroup = True

  def tearDown(self):
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_stack_dependency_graph(self):
    # These runs of dark and bias frames will be used by both runs of light
    # frames.
    make_synthetic_frames(prefix='dark',
                          number=5,
                          image_type='Dark',
                          fill_value=100,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=context.input_dirs[0])
    make_synthetic_frames(prefix='bias',
                          number=7,
                          image_type='Bias',
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=context.input_dirs[0])

    # The first run of light frames and its associated flats.
    copy_testdata_frames(prefix='light_run_0',
                         target_name='NGC7635',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 16, 29, 43),
                         dest_dir=context.input_dirs[0])
    make_synthetic_frames(prefix='flat_run_0',
                          number=11,
                          image_type='Flat',
                          start_datetime=datetime.datetime(
                              2024, 2, 5, 18, 29, 43),
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=context.input_dirs[0])

    # The second run of light frames and its associated flats.
    copy_testdata_frames(prefix='light_run_1',
                         target_name='NGC7635',
                         start_datetime=datetime.datetime(
                             2024, 2, 7, 16, 29, 43),
                         dest_dir=context.input_dirs[0])
    make_synthetic_frames(prefix='flat_run_1',
                          number=13,
                          image_type='Flat',
                          start_datetime=datetime.datetime(
                              2024, 2, 7, 18, 29, 43),
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=context.input_dirs[0])

    filenames = find_image_files(context.input_dirs[0])
    self.assertEqual(len(filenames), 42)

    graph = build_compute_graph()
    run_graph(graph)

    stacks = context.stacks
    self.assertEqual(len(stacks), 7)

    light_stacks = [s for s in stacks.values() if s.image_type == 'light']
    self.assertEqual(len(light_stacks), 3)

    keys = context.get_cache_keys()
    for k in keys:
      print(f'### cache key {k}')

    # TODO: More testing is needed, to make sure the calibration frames are
    # actually applied correctly (e.g. the subtraction has been done).
    self.assertTrue(
        context.contains(
            ('stack_subframe_thumbnail_image', '8e3471f8f0666240')))
    self.assertTrue(
        context.contains(
            ('stack_subframe_thumbnail_video', '8e3471f8f0666240')))


if __name__ == "__main__":
  unittest.main()
