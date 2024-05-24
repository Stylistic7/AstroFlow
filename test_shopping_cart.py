import ray_local

import ray
import unittest
import datetime

import file_io
from compute_graph import build_compute_graph
from dependency_graph import run_graph
from make_test_context import make_test_context
from make_test_frames import make_synthetic_frames, copy_testdata_frames


class TestShoppingCart(unittest.TestCase):

  def setUp(self):
    self.context = make_test_context()

  def test_one_stack_in_shopping_cart(self):
    copy_testdata_frames(prefix='light_run_0',
                         target_name='NGC7635',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 16, 29, 43),
                         dest_dir=self.context.input_dirs[0])
    make_synthetic_frames(prefix='dark',
                          number=5,
                          image_type='Dark',
                          fill_value=100,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=self.context.input_dirs[0])
    make_synthetic_frames(prefix='bias',
                          number=7,
                          image_type='Bias',
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=self.context.input_dirs[0])
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
                          dest_dir=self.context.input_dirs[0])

    filenames = file_io.find_image_files(self.context.input_dirs[0])
    self.assertEqual(len(filenames), 42)

    graph = build_compute_graph()
    self.context, _ = run_graph(graph, self.context)

    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 7)

    light_stacks = [s for s in stacks.values() if s.image_type == 'light']
    self.assertEqual(len(light_stacks), 3)


if __name__ == "__main__":
  unittest.main()
