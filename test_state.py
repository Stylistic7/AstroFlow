import ray_local

import ray
import os
import unittest
import datetime
from pathlib import Path

from context import context
from state_actor import dispatch
import get_stacks
import file_io
from compute_graph import build_compute_graph, build_processing_graph
from dependency_graph import run_graph
from make_test_context import make_test_context
from make_test_frames import make_synthetic_frames, copy_testdata_frames


class TestState(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()
    context.mock_stacking = True

  def test_shopping_cart(self):
    state = dispatch(dict(type='add_to_shopping_cart', item=('stack', 'foo')))
    self.assertEqual(len(state['shopping_cart']), 1)

    state = dispatch(dict(type='add_to_shopping_cart', item=('stack', 'foo')))
    self.assertEqual(len(state['shopping_cart']), 1)

    state = dispatch(
        dict(type='remove_from_shopping_cart', item=('stack', 'foo')))
    self.assertEqual(len(state['shopping_cart']), 0)

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

    filenames = file_io.find_image_files(context.input_dirs[0])
    self.assertEqual(len(filenames), 26)

    graph = build_compute_graph()
    run_graph(graph)

    masters = os.listdir(context.output_dir)
    self.assertEqual(len(masters), 0)

    stacks = context.stacks
    toplevel_stacks = get_stacks.get_toplevel_stacks(stacks)
    self.assertEqual(len(toplevel_stacks), 1)
    toplevel_stack = toplevel_stacks[0]

    context.shopping_cart = [('stack', toplevel_stack.geometric_hash)]

    processing_graph = ray.get(build_processing_graph.remote())
    self.assertEqual(len(processing_graph), 1)

    graph = build_compute_graph()
    run_graph(graph)

    masters = list(Path(context.output_dir).rglob('*.xisf'))
    self.assertEqual(len(masters), 4)


if __name__ == "__main__":
  unittest.main()
