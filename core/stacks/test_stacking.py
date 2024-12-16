from ray_infra import ray_local

import unittest
import datetime
from pathlib import Path

from testing.test_context import make_test_context
from context import context
from compute_graph import build_compute_graph
from dependency_graph import run_graph
from testing.test_frames import copy_testdata_frames, make_synthetic_frames


class TestStacking(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()
    context.regroup = True
    context.mock_stacking = False

  def tearDown(self):
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_orion_pane(self):
    copy_testdata_frames(target_name='Orion Pane 1',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 16, 29, 43),
                         dest_dir=context.input_dirs[0],
                         add_noise=False)
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

    # Running the graph the first time, with an empty shopping cart, should
    # produce all the subframe previews for the stack.
    graph = build_compute_graph(regroup=True,
                                make_previews=True,
                                run_main_processing_graph=True)
    run_graph(graph, debug=True)

    self.assertEqual(len(context.stacks), 4)
    self.assertEqual(len(context.targets), 1)

    for stack in context.stacks.values():
      print(f'### stack {stack} type {type(stack)}')

    context.shopping_cart = [('stack', 'e1f7d909ac125ebe')]

    graph = build_compute_graph()
    run_graph(graph, debug=True)

    filenames = list(
        str(f) for f in Path(context.output_dir).rglob("*post_processed.xisf"))
    self.assertEqual(len(filenames), 1)


if __name__ == "__main__":
  unittest.main()
