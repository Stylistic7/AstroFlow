from ray_infra import ray_local

import ray
import os
import unittest
import datetime
from pathlib import Path
import time

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph, Graph
from state import dispatch, enqueue_action, main_event_loop
from testing import make_test_context, make_synthetic_frames, copy_testdata_frames

from file_io.find_image_files import find_image_files


@ray.remote
def my_task(input):
  time.sleep(5)


class TestState(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def tearDown(self) -> None:
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_shopping_cart(self):
    state = dispatch(dict(type='add_to_shopping_cart', item=('stack', 'foo')))
    self.assertEqual(len(context.shopping_cart), 1)

    state = dispatch(dict(type='add_to_shopping_cart', item=('stack', 'foo')))
    self.assertEqual(len(context.shopping_cart), 1)

    state = dispatch(
        dict(type='remove_from_shopping_cart', item=('stack', 'foo')))
    self.assertEqual(len(context.shopping_cart), 0)

  def test_single_compute_graph(self):
    copy_testdata_frames(prefix='light_run_0',
                         target_name='NGC7635',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 16, 29, 43),
                         dest_dir=context.input_dirs[0])
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

    filenames = find_image_files(context.input_dirs[0])
    self.assertEqual(len(filenames), 26)

    graph = build_compute_graph()
    run_graph(graph)

    masters = list(Path(context.output_dir).rglob('*.xisf'))
    self.assertEqual(len(masters), 0)

    stacks = context.stacks
    self.assertEqual(len(stacks), 1)
    toplevel_stack = next(iter(stacks.values()))

    context.shopping_cart = [('stack', toplevel_stack.stable_hash)]

    graph = build_compute_graph()
    run_graph(graph)

    masters = list(Path(context.output_dir).rglob('*.xisf'))
    for master in masters:
      print(f'master {master}')
    self.assertEqual(len(masters), 6)


if __name__ == "__main__":
  unittest.main()
