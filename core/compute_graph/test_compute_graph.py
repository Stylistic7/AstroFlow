from ray_infra import ray_local

import ray
import unittest
import datetime
import os

from file_io.find_image_files import find_image_files
from context import context
from stacks.get_stacks import get_toplevel_stacks
from testing.test_frames import copy_testdata_frames, make_synthetic_frames
import compute_graph
from dependency_graph import run_graph, print_graph
from dependency_graph.debugging import node_to_string
from testing.test_context import make_test_context
from mosaics.orion_mosaic import add_orion_mosaic_images
from compute_graph import build_compute_graph


class TestComputeGraph(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()
    context.mock_stacking = True

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

    graph = compute_graph.build_compute_graph(regroup=True,
                                              make_previews=False,
                                              run_main_processing_graph=False)
    run_graph(graph)

    masters = os.listdir(context.output_dir)
    self.assertEqual(len(masters), 0)

    toplevel_stacks = get_toplevel_stacks(context.stacks)
    self.assertEqual(len(toplevel_stacks), 1)
    toplevel_stack = next(iter(toplevel_stacks))

    context.shopping_cart = [('stack', toplevel_stack.stable_hash)]

    _, graph = ray.get(compute_graph.build_main_processing_graph.remote())
    print_graph(graph)
    results, _ = run_graph(graph, trace=True)

    expected_names = [
        "build_stack_subgraph", "customize_stack",
        "create_stack_subframe_thumbnails", "build_stacking_subgraph",
        "stack_files", "compute_image_metadata", "post_process_stack",
        "compute_image_metadata", "gather_compute_image_metadata",
        "main_processing_graph_done"
    ]
    for node, expected_name in zip(results.trace, expected_names):
      self.assertEqual(node.task._function.__name__, expected_name)

  def test_orion_mosaic(self) -> None:
    # This test uses a random seed because it checks the specific values of
    # hashes, so all of the frames will have predictable contents.
    add_orion_mosaic_images()
    results, _ = run_graph(build_compute_graph(regroup=True,
                                               make_previews=True,
                                               run_main_processing_graph=False),
                           trace=True)

    self.assertEqual(len(context.stacks), 4)
    self.assertEqual(len(context.calibration_stacks), 3)
    self.assertEqual(len(context.targets), 4)
    self.assertEqual(len(context.mosaics), 1)

    for node in results.trace:
      print('### trace', node_to_string(node))


if __name__ == '__main__':
  unittest.main()
