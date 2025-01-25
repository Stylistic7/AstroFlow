from ray_infra import ray_local

import ray
import unittest
import datetime
import os
from typing import Any

from compute_graph import build_compute_graph
from context import context
from dependency_graph import Node, run_graph
from testing import make_test_context, copy_testdata_frames, make_synthetic_frames

from file_io.find_image_files import find_image_files
from stacks.get_stacks import get_toplevel_stacks
from dependency_graph.debugging import node_to_string
from mosaics.orion_mosaic import add_orion_mosaic_images
from compute_graph.compute_graph import build_compute_graph
from metadata.image_metadata import compute_image_metadata, load_metadata
from customization.compute_graph import customize_stack
from stacks.stacking import stack_files


def find_compatible_nodes(trace: list[Node], template: Any) -> list[int]:
  return [i for i, node in enumerate(trace) if node.compatible_with(template)]


# Things to look for in these tests:
# - Customize each stack only once
# - Stack all stacks involved in a Target before the Target's post-processing
# - Run all Target post-processing before any Mosaic merging
# - If all Stacks in a Mosaic's color channel are available, merge the masters
#   - Otherwise, only merge the subframes
# - Stack files of any objects in the shopping cart
# - Create thumbnails for both subframes and masters, when available


class TestComputeGraph(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()

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
    make_synthetic_frames(prefix='flat_run_1',
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

    graph = build_compute_graph(regroup=True,
                                make_previews=False,
                                run_main_processing_graph=False)
    results, _ = run_graph(graph, trace=True)

    print('### First run of graph:')
    for node in results.trace:
      print('### trace', node_to_string(node))

    expected_task_names = [
        "scatter_compute_image_metadata", "compute_image_metadata",
        "compute_image_metadata", "compute_image_metadata",
        "compute_image_metadata", "compute_image_metadata",
        "compute_image_metadata", "load_metadata", "build_regroup_graph",
        "cluster_by_image_center", "scatter_spatial_groups",
        "build_spatial_group_graph", "identify_stacks", "validate_stacks",
        "store_stacks", "group_stacks_into_targets", "gather_spatial_groups",
        "store_targets", "identify_mosaics", "store_mosaics",
        "identify_calibration_stacks", "store_calibration_stacks",
        "build_customize_graph", "scatter_customize_stacks",
        "scatter_customize_mosaics", "scatter_customize_targets",
        "customize_stack", "gather_customize_stacks", "customization_done"
    ]

    for node, expected_name in zip(results.trace, expected_task_names):
      self.assertEqual(node.task._function.__name__, expected_name)

    # Running the graph the first time should compute metadata for the 6 light
    # frames, and customize just a single stack (composed of all 6 light
    # frames).
    self.assertEqual(
        len(find_compatible_nodes(results.trace, compute_image_metadata)), 6)
    self.assertEqual(len(find_compatible_nodes(results.trace, load_metadata)),
                     1)
    self.assertEqual(len(find_compatible_nodes(results.trace, customize_stack)),
                     1)

    # No stacks should have been stacked into masters.
    self.assertEqual(len(find_compatible_nodes(results.trace, stack_files)), 0)
    masters = os.listdir(context.output_dir)
    self.assertEqual(len(masters), 0)

    # There should be one composite stack, and four calibration stacks.
    self.assertEqual(len(context.stacks), 1)
    self.assertEqual(len(context.calibration_stacks), 4)
    self.assertEqual(len(context.targets), 1)
    self.assertEqual(len(context.mosaics), 0)
    toplevel_stacks = get_toplevel_stacks(context.stacks)
    self.assertEqual(len(toplevel_stacks), 1)
    self.assertEqual(len(context.calibration_stacks), 4)
    toplevel_stack = next(iter(toplevel_stacks))

    # Running the graph a second time should not compute any new metadata;
    # there will be no need to load metadata either.
    results, _ = run_graph(graph, trace=True)

    print('### Second run of graph:')
    for node in results.trace:
      print('### trace', node_to_string(node))

    expected_task_names = [
        "scatter_compute_image_metadata", "build_regroup_graph",
        "cluster_by_image_center", "scatter_spatial_groups",
        "build_spatial_group_graph", "identify_stacks", "validate_stacks",
        "store_stacks", "group_stacks_into_targets", "gather_spatial_groups",
        "store_targets", "identify_mosaics", "store_mosaics",
        "identify_calibration_stacks", "store_calibration_stacks",
        "build_customize_graph", "scatter_customize_stacks",
        "scatter_customize_mosaics", "scatter_customize_targets",
        "customize_stack", "gather_customize_stacks", "customization_done"
    ]

    for node, expected_name in zip(results.trace, expected_task_names):
      self.assertEqual(node.task._function.__name__, expected_name)

    self.assertEqual(
        len(find_compatible_nodes(results.trace, compute_image_metadata)), 0)
    self.assertEqual(len(find_compatible_nodes(results.trace, load_metadata)),
                     0)
    self.assertEqual(len(find_compatible_nodes(results.trace, customize_stack)),
                     1)

    context.shopping_cart = [('stack', toplevel_stack.stable_hash)]
    graph = compute_graph.build_compute_graph(regroup=False,
                                              make_previews=True,
                                              run_main_processing_graph=True)
    results, _ = run_graph(graph, trace=True)

    print('### First run of graph with previews and main processing:')
    for node in results.trace:
      print('### trace', node_to_string(node))

    expected_names = [
        "scatter_compute_image_metadata", "identify_calibration_stacks",
        "store_calibration_stacks", "build_customize_graph",
        "scatter_customize_stacks", "scatter_customize_mosaics",
        "scatter_customize_targets", "customize_stack",
        "gather_customize_stacks", "customization_done", "build_preview_graph",
        "scatter_stack_thumbnails", "create_flat_correction_thumbnails",
        "create_stack_subframe_thumbnails", "gather", "preview_graph_done",
        "build_main_processing_graph", "build_stacking_subgraph", "stack_files",
        "stack_files", "stack_files", "stack_files", "stack_files",
        "compute_image_metadata", "post_process_stack",
        "compute_image_metadata", "load_metadata", "main_processing_graph_done"
    ]
    for node, expected_name in zip(results.trace, expected_names):
      self.assertEqual(node.task._function.__name__, expected_name)

  def test_orion_mosaic(self) -> None:
    add_orion_mosaic_images()

    # TODO: Run this graph twice, and check which tasks disappear from the
    # trace due to caching.
    results, _ = run_graph(build_compute_graph(regroup=True,
                                               make_previews=True,
                                               run_main_processing_graph=False),
                           trace=True)

    self.assertEqual(len(context.stacks), 4)
    self.assertEqual(len(context.calibration_stacks), 3)
    self.assertEqual(len(context.targets), 4)
    self.assertEqual(len(context.mosaics), 1)

    # TODO: Add some expectations of what's in the trace, what order, etc.
    for node in results.trace:
      print('### trace', node_to_string(node))

    mosaic = next(iter(context.mosaics.values()))
    context.shopping_cart = [('mosaic', mosaic.stable_hash)]

    results, _ = run_graph(build_compute_graph(regroup=True,
                                               make_previews=True,
                                               run_main_processing_graph=True),
                           trace=True)

    # TODO: Add some expectations of what's in the trace, what order, etc.
    for node in results.trace:
      print('### trace', node_to_string(node))


if __name__ == '__main__':
  unittest.main()
