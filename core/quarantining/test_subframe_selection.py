from ray_infra import ray_local

import datetime
import unittest
import ray

from context import context
from testing.test_context import make_test_context
from testing.test_frames import make_synthetic_frames
from file_io.find_image_files import find_image_files
from dependency_graph import run_graph
from stacks import CompositeStack
from compute_graph import build_compute_graph


class TestSubframeSelection(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()
    context.regroup = True

  def tearDown(self):
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_subframe_selection_subgraph(self):
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
                          number=7,
                          image_type='Light',
                          fill_value=100,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          num_stars=10,
                          dest_dir=context.input_dirs[0])

    filenames = find_image_files(context.input_dirs[0])
    self.assertEqual(len(filenames), 12)

    run_graph(build_compute_graph(run_main_processing_graph=False))

    # The max number of stars in the stack is 100. The default 75% acceptance
    # threshold means that only those frames with more than 75 stars will be
    # accepted. There are 5 such frames.
    stacks = context.stacks
    self.assertEqual(len(stacks), 2)
    stack = next(iter(s for s in stacks.values() if type(s) is CompositeStack))
    self.assertEqual(len(stack.all_frames()), 5)


if __name__ == "__main__":
  unittest.main()
