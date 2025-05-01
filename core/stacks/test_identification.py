from ray_infra import ray_local

import ray
import unittest

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from testing import make_test_context, make_synthetic_frames
from metadata.image_metadata import load_metadata


class TestStackIdentification(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()
    context.regroup = True

  def tearDown(self):
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_one_perfect_stack(self):
    make_synthetic_frames(ra=100,
                          dec=50,
                          prefix='foo',
                          number=10,
                          dither=0,
                          dest_dir=context.input_dirs[0])

    run_graph(build_compute_graph())

    stacks = context.stacks
    self.assertEqual(len(stacks), 1)
    toplevel_stack = next(iter(stacks.values()))
    self.assertEqual(toplevel_stack.center, [100, 50])
    self.assertEqual(toplevel_stack.image_scale, 1.0)

    self.assertEqual(len(toplevel_stack.contents), 1)
    stack = next(iter(toplevel_stack.contents.values()))
    self.assertEqual(len(stack.frames), 10)

  def test_two_stacks_different_image_scales(self):
    make_synthetic_frames(ra=100,
                          dec=50,
                          prefix='foo',
                          number=10,
                          dither=0,
                          dest_dir=context.input_dirs[0])
    make_synthetic_frames(ra=100,
                          dec=50,
                          prefix='bar',
                          number=10,
                          dither=0,
                          image_scale=2.0,
                          dest_dir=context.input_dirs[0])

    run_graph(build_compute_graph())

    stacks = context.stacks
    self.assertEqual(len(stacks), 2)

    for stack in stacks:
      print(f'### toplevel stack {stack}')

    stack = stacks['963b3bad57928528']
    frames = stack.all_frames()
    metadata = stack.all_metadata()
    filenames = [m.filename for m in metadata.values()]
    self.assertTrue(all('foo' in f for f in filenames))
    self.assertEqual(len(frames), 10)

    stack = stacks['31520d846ae3463d']
    frames = stack.all_frames()
    metadata = stack.all_metadata()
    filenames = [m.filename for m in metadata.values()]
    self.assertTrue(all('bar' in f for f in filenames))
    self.assertEqual(len(frames), 10)

  def test_two_stacks_different_image_sizes(self):
    # Note that these two groups have the same generic camera name, which is
    # typical of some capture software.
    make_synthetic_frames(ra=100,
                          dec=50,
                          prefix='foo',
                          instrument='ASI Camera',
                          number=10,
                          dither=0,
                          width_pixels=3000,
                          height_pixels=2000,
                          dest_dir=context.input_dirs[0])
    make_synthetic_frames(ra=100,
                          dec=50,
                          prefix='bar',
                          instrument='ASI Camera',
                          number=10,
                          dither=0,
                          width_pixels=6000,
                          height_pixels=4000,
                          dest_dir=context.input_dirs[0])

    run_graph(build_compute_graph())

    stacks = context.stacks
    self.assertEqual(len(stacks), 2)

    keys = sorted(stacks.keys())

    stack = stacks[keys[0]]
    self.assertEqual(stack.center, [100, 50])
    self.assertEqual(stack.width_pixels, 6000)
    self.assertEqual(stack.height_pixels, 4000)
    self.assertEqual(stack.image_scale, 1.0)

    stack = stacks[keys[1]]
    self.assertEqual(stack.center, [100, 50])
    self.assertEqual(stack.width_pixels, 3000)
    self.assertEqual(stack.height_pixels, 2000)
    self.assertEqual(stack.image_scale, 1.0)

  def test_multiple_exposures(self):
    make_synthetic_frames(ra=100,
                          dec=50,
                          prefix='foo',
                          number=5,
                          dither=0,
                          exposure=300,
                          dest_dir=context.input_dirs[0])
    make_synthetic_frames(ra=100,
                          dec=50,
                          prefix='bar',
                          number=7,
                          dither=0,
                          exposure=120,
                          dest_dir=context.input_dirs[0])

    run_graph(build_compute_graph())

    stacks = context.stacks
    self.assertEqual(len(stacks), 1)

    toplevel_stack = next(iter(stacks.values()))
    print(f'### toplevel_stack {toplevel_stack}')
    print(f'### toplevel_stack.contents {toplevel_stack.contents}')
    self.assertEqual(toplevel_stack.center, [100, 50])
    self.assertEqual(len(toplevel_stack.contents), 2)

    stack = list(toplevel_stack.contents.values())[0]
    self.assertTrue(all('bar' in f.filename for f in stack.metadata.values()))
    self.assertEqual(len(stack.frames), 7)
    self.assertEqual(stack.exposure_s, [120.0])
    self.assertEqual(stack.integration_s, 7 * 120)

    stack = list(toplevel_stack.contents.values())[1]
    self.assertTrue(all('foo' in f.filename for f in stack.metadata.values()))
    self.assertEqual(len(stack.frames), 5)
    self.assertEqual(stack.exposure_s, [300.0])
    self.assertEqual(stack.integration_s, 5 * 300)


if __name__ == '__main__':
  unittest.main()
