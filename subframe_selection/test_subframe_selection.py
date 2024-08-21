from ray_infra import ray_local

import datetime
import unittest
import ray

from context import context
from testing.test_context import make_test_context
from testing.test_frames import make_synthetic_frames
from file_io.find_image_files import find_image_files
from stacks.identification import identify_stacks
from clustering import cluster_by_image_center
import subframe_selection
from metadata.actor import get_metadata

# TODO: These tests don't actually test the subframe selection at all. The
# stack identification step no longer eliminates bad frames from the stack.
# Instead, we're going to do subframe selection right before stacking.


class TestSubframeSelection(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()
    context.regroup = True

  def tearDown(self):
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_star_count_fraction(self):
    make_synthetic_frames(prefix='light_clear',
                          start_datetime=datetime.datetime(
                              2023, 12, 6, 16, 29, 43),
                          number=5,
                          image_type='Light',
                          fill_value=100,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          num_stars=100,
                          dest_dir=context.input_dirs[0])
    make_synthetic_frames(prefix='light_cloudy',
                          start_datetime=datetime.datetime(
                              2023, 12, 6, 16, 59, 43),
                          number=5,
                          image_type='Light',
                          fill_value=100,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          num_stars=10,
                          dest_dir=context.input_dirs[0])

    filenames = find_image_files(context.input_dirs[0])
    self.assertEqual(len(filenames), 10)

    ray.get(cluster_by_image_center.remote())
    ray.get(identify_stacks.remote())

    stacks = context.stacks
    self.assertEqual(len(stacks), 2)
    stack = next(iter(s for s in stacks.values() if s.stable_hash))
    self.assertEqual(len(stack.all_frames()), 10)

  def test_split_run(self):
    make_synthetic_frames(prefix='light',
                          start_datetime=datetime.datetime(
                              2023, 12, 6, 16, 29, 43),
                          number=15,
                          image_type='Light',
                          fill_value=100,
                          instrument='ZWO ASI6200MM Pro',
                          exposure=600,
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          num_stars=100,
                          dest_dir=context.input_dirs[0])

    # The middle frame has too few stars, and should split the run into two.
    metadata = get_metadata()

    for i in metadata:
      if i.filename.endswith('_00007.fit'):
        middle_frame = i
    print(f'middle_frame {middle_frame.filename}')
    middle_frame.plate_solution.num_stars = 10
    context.set(('image_metadata', middle_frame.content_hash), middle_frame)

    filenames = find_image_files(context.input_dirs[0])
    self.assertEqual(len(filenames), 15)

    ray.get(cluster_by_image_center.remote())
    ray.get(identify_stacks.remote())

    stacks = context.stacks
    self.assertEqual(len(stacks), 2)
    stack = next(iter(s for s in stacks.values() if s.stable_hash))
    self.assertEqual(len(stack.all_frames()), 15)


if __name__ == "__main__":
  unittest.main()
