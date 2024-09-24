from ray_infra import ray_local

import ray
import unittest

from testing.test_context import make_test_context
from context import context
from dependency_graph import run_graph
from testing.test_frames import make_synthetic_frames
from stacks.get_stacks import get_toplevel_stacks, get_content_stacks_of_type
from metadata.actor import get_metadata
from compute_graph import build_compute_graph


class TestStackIdentification(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()
    context.regroup = True

  def tearDown(self):
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_one_perfect_stack(self):
    make_synthetic_frames(ra=100, dec=50, prefix='foo', number=10, dither=0)

    run_graph(
        build_compute_graph(regroup=True,
                            make_previews=False,
                            run_main_processing_graph=False))

    stacks = context.stacks
    self.assertEqual(len(stacks), 2)

    toplevel_stacks = get_toplevel_stacks(stacks)
    self.assertEqual(len(toplevel_stacks), 1)
    stack = toplevel_stacks[0]
    self.assertEqual(stack.center, [100, 50])
    self.assertEqual(stack.image_scale, 1.0)

    content_stacks = get_content_stacks_of_type(stacks, 'light')
    self.assertEqual(len(content_stacks), 1)
    stack = content_stacks[0]
    self.assertEqual(len(stack.frames), 10)

  def test_two_stacks_different_image_scales(self):
    make_synthetic_frames(ra=100, dec=50, prefix='foo', number=10, dither=0)
    make_synthetic_frames(ra=100,
                          dec=50,
                          prefix='bar',
                          number=10,
                          dither=0,
                          image_scale=2.0)

    run_graph(
        build_compute_graph(regroup=True,
                            make_previews=False,
                            run_main_processing_graph=False))
    stacks = context.stacks
    self.assertEqual(len(stacks), 4)

    toplevel_stacks = get_toplevel_stacks(stacks)
    self.assertEqual(len(toplevel_stacks), 2)
    stack = toplevel_stacks[0]
    self.assertEqual(stack.center, [100, 50])
    self.assertEqual(stack.image_scale, 1.0)

    content_stacks = get_content_stacks_of_type(stacks, 'light')
    self.assertEqual(len(content_stacks), 2)
    stack = content_stacks[0]
    filenames = get_metadata(stack.frames, transform_fn=lambda i: i.filename)
    self.assertTrue(all('foo' in f for f in filenames))
    self.assertEqual(len(stack.frames), 10)

    stack = content_stacks[1]
    filenames = get_metadata(stack.frames, transform_fn=lambda i: i.filename)
    self.assertTrue(all('bar' in f for f in filenames))
    self.assertEqual(len(stack.frames), 10)

  def test_two_stacks_different_image_sizes(self):
    # Note that these two groups have the same generic camera name, which is
    # typical of some capture software.
    make_synthetic_frames(
        ra=100,
        dec=50,
        prefix='foo',
        instrument='ASI Camera',
        number=10,
        dither=0,
        width_pixels=3000,
        height_pixels=2000,
    )
    make_synthetic_frames(ra=100,
                          dec=50,
                          prefix='bar',
                          instrument='ASI Camera',
                          number=10,
                          dither=0,
                          width_pixels=6000,
                          height_pixels=4000)

    run_graph(
        build_compute_graph(regroup=True,
                            make_previews=False,
                            run_main_processing_graph=False))

    stacks = context.stacks
    self.assertEqual(len(stacks), 4)

    toplevel_stacks = get_toplevel_stacks(stacks)
    self.assertEqual(len(toplevel_stacks), 2)

    self.assertEqual(toplevel_stacks[0].center, [100, 50])
    self.assertEqual(toplevel_stacks[0].width_pixels, 3000)
    self.assertEqual(toplevel_stacks[0].height_pixels, 2000)
    self.assertEqual(toplevel_stacks[0].image_scale, 1.0)

    self.assertEqual(toplevel_stacks[1].center, [100, 50])
    self.assertEqual(toplevel_stacks[1].width_pixels, 6000)
    self.assertEqual(toplevel_stacks[1].height_pixels, 4000)
    self.assertEqual(toplevel_stacks[1].image_scale, 1.0)

    content_stacks = get_content_stacks_of_type(stacks, 'light')
    self.assertEqual(len(content_stacks), 2)

  def test_multiple_exposures(self):
    make_synthetic_frames(ra=100,
                          dec=50,
                          prefix='foo',
                          number=5,
                          dither=0,
                          exposure=300)
    make_synthetic_frames(ra=100,
                          dec=50,
                          prefix='bar',
                          number=7,
                          dither=0,
                          exposure=120)

    run_graph(
        build_compute_graph(regroup=True,
                            make_previews=False,
                            run_main_processing_graph=False))

    stacks = context.stacks
    self.assertEqual(len(stacks), 3)

    toplevel_stacks = get_toplevel_stacks(stacks)
    self.assertEqual(len(toplevel_stacks), 1)
    stack = toplevel_stacks[0]
    print(f'### toplevel_stack {stack}')
    print(f'### toplevel_stack.contents {stack.contents}')
    self.assertEqual(stack.center, [100, 50])
    self.assertEqual(len(stack.contents), 2)

    content_stacks = get_content_stacks_of_type(stacks, 'light')
    self.assertEqual(len(content_stacks), 2)
    stack = content_stacks[0]
    filenames = get_metadata(stack.frames, transform_fn=lambda i: i.filename)
    self.assertTrue(all('bar' in f for f in filenames))
    self.assertEqual(len(stack.frames), 7)
    self.assertEqual(stack.exposure_s, [120.0])
    self.assertEqual(stack.integration_s, 7 * 120)

    stack = content_stacks[1]
    filenames = get_metadata(stack.frames, transform_fn=lambda i: i.filename)
    self.assertTrue(all('foo' in f for f in filenames))
    self.assertEqual(len(stack.frames), 5)
    self.assertEqual(stack.exposure_s, [300.0])
    self.assertEqual(stack.integration_s, 5 * 300)


if __name__ == '__main__':
  unittest.main()
