import ray_local

import ray
import unittest
import datetime

from cache import cache
from context import context
from testing.make_test_frames import make_synthetic_frames, copy_testdata_frames
from testing.make_test_context import make_test_context
from stacks.identification import identify_stacks
from spatial_association import find_components_by_cosine_similarity
from get_stacks import get_toplevel_stacks, get_content_stacks_of_type
from dependency_graph import run_graph
from compute_graph import build_compute_graph


class TestFlatCorrection(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()
    context.regroup = True

  def tearDown(self):
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_one_perfect_stack(self):
    copy_testdata_frames(prefix='light_run',
                         target_name='NGC7635',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 16, 29, 43),
                         dest_dir=context.input_dirs[0])
    make_synthetic_frames(prefix='bias_run_0',
                          number=5,
                          image_type='Bias',
                          start_datetime=datetime.datetime(
                              2024, 2, 5, 18, 29, 43),
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-20,
                          dest_dir=context.input_dirs[0])
    make_synthetic_frames(prefix='dark_run_0',
                          number=7,
                          image_type='Dark',
                          start_datetime=datetime.datetime(
                              2024, 2, 5, 18, 29, 43),
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-20,
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
                          temperature=-20,
                          dest_dir=context.input_dirs[0])
    make_synthetic_frames(prefix='flat_run_1',
                          number=11,
                          image_type='Flat',
                          start_datetime=datetime.datetime(
                              2024, 2, 4, 18, 29, 43),
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-20,
                          dest_dir=context.input_dirs[0])

    graph = build_compute_graph()
    run_graph(graph)

    for k in cache.iterkeys():
      print(k)
      if 'flat_correction_thumbnails' in k:
        print('type(cache[k])', type(cache[k]))
        print('key', k, 'has keys', cache[k].keys())

    stacks = context.stacks
    self.assertEqual(len(stacks), 6)

    for stack in stacks.values():
      print(stack)

    # key = ('flat_correction_thumbnails', stack.stable_hash)


if __name__ == '__main__':
  unittest.main()
