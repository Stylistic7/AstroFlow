import ray_local

import ray
import unittest
import numpy as np

from context import context
from stack import Stack
from make_test_context import make_test_context
from make_test_frames import make_synthetic_frames, copy_testdata_frames
from dependency_graph import run_graph, make_graph_from_list
from image_metadata import scatter_compute_image_metadata
from spatial_association import find_components_by_cosine_similarity
from stack_identification import identify_stacks
from calibration_frames import identify_calibration_stacks, associate_with_stacks
from make_test_context import make_test_context
from hashing import hash_list


def _build_compute_graph():
  return make_graph_from_list([(scatter_compute_image_metadata, 'all'),
                               find_components_by_cosine_similarity,
                               identify_stacks, identify_calibration_stacks,
                               associate_with_stacks])


class TestStack(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def test_all_frames(self):
    stacks = {
        'toplevel_stack':
            Stack(content_hash='toplevel_stack',
                  contents=['stack_foo', 'stack_bar', 'stack_baz']),
        'stack_foo':
            Stack(content_hash='stack_bar', frames=['a', 'b', 'c']),
        'stack_bar':
            Stack(content_hash='stack_bar', frames=['d', 'e', 'f']),
        'stack_baz':
            Stack(content_hash='stack_baz', frames=['g', 'h', 'i']),
    }

    self.assertEqual(sorted(stacks['toplevel_stack'].all_frames(stacks)),
                     ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])

  def test_one_dark_stack(self):
    make_synthetic_frames(prefix='dark',
                          number=5,
                          image_type='Dark',
                          fill_value=100,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          exposure=300,
                          dest_dir=context.input_dirs[0])

    graph = _build_compute_graph()
    run_graph(graph)

    stacks = context.stacks
    self.assertEqual(len(stacks), 1)
    stack = next(iter(stacks.values()))

    self.assertNotEqual(stack.content_hash, None)
    self.assertEqual(len(stack.frames), 5)
    self.assertEqual(len(stack.contents), 0)
    self.assertEqual(len(stack.dependencies), 0)
    self.assertEqual(stack.image_type, 'dark')
    self.assertNotEqual(stack.oldest_date, None)
    self.assertNotEqual(stack.newest_date, None)
    self.assertEqual(stack.target, None)
    self.assertEqual(stack.camera, 'ZWO ASI6200MM Pro')
    self.assertEqual(stack.mean_temperature, -15.0)
    self.assertEqual(stack.exposure_s, 300)
    self.assertEqual(stack.integration_s, 5 * 300)
    self.assertEqual(stack.width_pixels, 9576)
    self.assertEqual(stack.height_pixels, 6388)

  def test_one_light_one_dark(self):
    copy_testdata_frames(target_name='NGC7635', dest_dir=context.input_dirs[0])
    make_synthetic_frames(prefix='dark',
                          number=5,
                          image_type='Dark',
                          fill_value=100,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-10,
                          exposure=300,
                          dest_dir=context.input_dirs[0])

    graph = _build_compute_graph()
    run_graph(graph)

    stacks = context.stacks
    self.assertEqual(len(stacks), 3)

    toplevel_stack = next(
        stack for stack in stacks.values() if stack.geometric_hash)
    light_stack = next(
        stack for stack in stacks.values() if stack.image_type == 'light')
    dark_stack = next(
        stack for stack in stacks.values() if stack.image_type == 'dark')

    self.assertEqual(toplevel_stack.image_type, 'light')
    self.assertNotEqual(toplevel_stack.geometric_hash, None)
    self.assertNotEqual(toplevel_stack.content_hash, None)
    self.assertEqual(len(toplevel_stack.contents), 1)
    self.assertEqual(len(toplevel_stack.dependencies), 0)
    self.assertNotEqual(toplevel_stack.oldest_date, None)
    self.assertNotEqual(toplevel_stack.newest_date, None)
    self.assertEqual(toplevel_stack.camera, 'ZWO ASI6200MM Pro')
    self.assertEqual(toplevel_stack.filter, 'H')
    self.assertAlmostEqual(toplevel_stack.mean_temperature, -12.9, places=3)
    self.assertEqual(toplevel_stack.exposure_s, [300])
    self.assertEqual(toplevel_stack.integration_s, 3 * 300)
    self.assertEqual(toplevel_stack.width_pixels, 9576)
    self.assertEqual(toplevel_stack.height_pixels, 6388)
    self.assertEqual(toplevel_stack.target, 'Bubble nebula')
    self.assertTrue(
        np.allclose(toplevel_stack.center, [350.183329899, 61.2229195667]))
    self.assertAlmostEqual(toplevel_stack.image_scale, 0.38, places=2)
    self.assertAlmostEqual(toplevel_stack.rotation_angle, 1.55, places=2)

    self.assertEqual(light_stack.image_type, 'light')
    self.assertEqual(light_stack.geometric_hash, None)
    self.assertNotEqual(light_stack.content_hash, None)
    self.assertEqual(len(light_stack.frames), 3)
    self.assertEqual(len(light_stack.contents), 0)
    self.assertEqual(light_stack.dependencies,
                     {'dark': dark_stack.content_hash})
    self.assertNotEqual(light_stack.oldest_date, None)
    self.assertNotEqual(light_stack.newest_date, None)
    self.assertEqual(light_stack.target, None)
    self.assertEqual(light_stack.camera, 'ZWO ASI6200MM Pro')
    self.assertAlmostEqual(light_stack.mean_temperature, -12.9, places=3)
    self.assertEqual(light_stack.exposure_s, 300)
    self.assertEqual(light_stack.integration_s, 3 * 300)
    self.assertEqual(light_stack.width_pixels, 9576)
    self.assertEqual(light_stack.height_pixels, 6388)

    self.assertEqual(dark_stack.image_type, 'dark')
    self.assertEqual(dark_stack.geometric_hash, None)
    self.assertNotEqual(dark_stack.content_hash, None)
    self.assertEqual(len(dark_stack.frames), 5)
    self.assertEqual(len(dark_stack.contents), 0)
    self.assertEqual(len(dark_stack.dependencies), 0)
    self.assertNotEqual(dark_stack.oldest_date, None)
    self.assertNotEqual(dark_stack.newest_date, None)
    self.assertEqual(dark_stack.target, None)
    self.assertEqual(dark_stack.camera, 'ZWO ASI6200MM Pro')
    self.assertEqual(dark_stack.mean_temperature, -10.0)
    self.assertEqual(dark_stack.exposure_s, 300)
    self.assertEqual(dark_stack.integration_s, 5 * 300)
    self.assertEqual(dark_stack.width_pixels, 9576)
    self.assertEqual(dark_stack.height_pixels, 6388)


if __name__ == "__main__":
  unittest.main()
