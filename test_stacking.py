import ray_local

import unittest
import datetime
from pathlib import Path

from make_test_context import make_test_context
from context import context
from compute_graph import build_compute_graph
from dependency_graph import run_graph
from make_test_frames import copy_testdata_frames


class TestStacking(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()
    context.regroup = True

  def tearDown(self):
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_orion_pane(self):
    copy_testdata_frames(target_name='Orion Pane 1',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 16, 29, 43),
                         dest_dir=context.input_dirs[0],
                         add_noise=False)

    # Running the graph the first time, with an empty shopping cart, should
    # produce all the subframe previews for the stack.
    graph = build_compute_graph()
    run_graph(graph, debug=True)

    self.assertEqual(len(context.stacks), 2)
    self.assertEqual(len(context.targets), 1)

    for stack_id, stack in context.stacks.items():
      print(f'### {stack}')

    context.shopping_cart = [('stack', '4475e17d7341cbd0')]

    graph = build_compute_graph()
    run_graph(graph, debug=True)


if __name__ == "__main__":
  unittest.main()
