import ray_local

import ray
import unittest
import datetime
import os

from orion_mosaic import add_orion_mosaic_images
from file_io.find_image_files import find_image_files
from context import context
import get_stacks
from testing.make_test_frames import copy_testdata_frames, make_synthetic_frames
import compute_graph
from dependency_graph import run_graph, print_graph
from testing.make_test_context import make_test_context


class TestComputeGraph(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()
    context.mock_stacking = True
    context.regroup = True

  def tearDown(self):
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_single_light_stack(self):
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

    copy_testdata_frames(prefix='light_run_1',
                         target_name='NGC7635',
                         start_datetime=datetime.datetime(
                             2024, 3, 5, 16, 29, 43),
                         dest_dir=context.input_dirs[0])
    make_synthetic_frames(prefix='flat_run_2',
                          number=13,
                          image_type='Flat',
                          start_datetime=datetime.datetime(
                              2024, 3, 5, 18, 29, 43),
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=context.input_dirs[0])

    filenames = find_image_files(context.input_dirs[0])
    self.assertEqual(len(filenames), 42)

    graph = compute_graph.build_compute_graph()
    run_graph(graph)

    masters = os.listdir(context.output_dir)
    self.assertEqual(len(masters), 0)

    stacks = context.stacks
    toplevel_stacks = get_stacks.get_toplevel_stacks(stacks)
    self.assertEqual(len(toplevel_stacks), 1)
    toplevel_stack = toplevel_stacks[0]

    context.shopping_cart = [('stack', toplevel_stack.stable_hash)]

    processing_graph = ray.get(
        compute_graph.build_main_processing_graph.remote())
    print_graph(processing_graph)

    self.assertEqual(len(processing_graph), 2)


if __name__ == '__main__':
  unittest.main()
