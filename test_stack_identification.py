import ray_local

import ray
import unittest
import datetime

from context import context
from make_test_frames import make_synthetic_frames
from stack_identification import identify_stacks
from spatial_association import find_components_by_cosine_similarity
from get_stacks import get_toplevel_stacks, get_content_stacks_of_type

# TODO: I may need to separate the concepts of the diskcache-based global
# context, and the context used in a Graph.
#
# What belongs in the global context? The stacks, targets, mosaics, and user
# settings.
#
# What belongs in the local context for the Graph? I'm not sure.


def set_default_context():
  context.cosine_similarity_threshold = 0.99999
  context.min_component_members = 3
  context.max_run_gap = datetime.timedelta(minutes=15)
  context.min_frames_in_run = 3
  context.min_frames_in_stack = 3
  context.quarantined_files = []


class TestStackIdentification(unittest.TestCase):

  def setUp(self):
    set_default_context()

  def test_one_perfect_stack(self):
    image_hashes, fits_headers, plate_solutions = make_synthetic_frames(
        ra=100, dec=50, prefix='foo', number=10, dither=0)

    context.image_hashes = image_hashes
    context.fits_headers = fits_headers
    context.plate_solutions = plate_solutions

    ray.get(find_components_by_cosine_similarity.remote())
    ray.get(identify_stacks.remote())

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
    image_hashes, fits_headers, plate_solutions = make_synthetic_frames(
        ra=100, dec=50, prefix='foo', number=10, dither=0)
    image_hashes, fits_headers, plate_solutions = make_synthetic_frames(
        ra=100,
        dec=50,
        prefix='bar',
        number=10,
        dither=0,
        image_scale=2.0,
        image_hashes=image_hashes,
        fits_headers=fits_headers,
        plate_solutions=plate_solutions)

    context.image_hashes = image_hashes
    context.fits_headers = fits_headers
    context.plate_solutions = plate_solutions

    ray.get(find_components_by_cosine_similarity.remote())
    ray.get(identify_stacks.remote())

    image_hashes = context.image_hashes
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
    self.assertTrue(all('foo' in image_hashes[f] for f in stack.frames))
    self.assertEqual(len(stack.frames), 10)

    stack = content_stacks[1]
    self.assertTrue(all('bar' in image_hashes[f] for f in stack.frames))
    self.assertEqual(len(stack.frames), 10)

  def test_two_stacks_different_image_sizes(self):
    # Note that these two groups have the same generic camera name, which is
    # typical of some capture software.
    image_hashes, fits_headers, plate_solutions = make_synthetic_frames(
        ra=100,
        dec=50,
        prefix='foo',
        instrument='ASI Camera',
        number=10,
        dither=0,
        width_pixels=3000,
        height_pixels=2000,
    )
    image_hashes, fits_headers, plate_solutions = make_synthetic_frames(
        ra=100,
        dec=50,
        prefix='bar',
        instrument='ASI Camera',
        number=10,
        dither=0,
        width_pixels=6000,
        height_pixels=4000,
        image_hashes=image_hashes,
        fits_headers=fits_headers,
        plate_solutions=plate_solutions)

    context.image_hashes = image_hashes
    context.fits_headers = fits_headers
    context.plate_solutions = plate_solutions

    ray.get(find_components_by_cosine_similarity.remote())
    ray.get(identify_stacks.remote())

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

    self.assertTrue(
        all('foo' in image_hashes[f] for f in content_stacks[0].frames))
    self.assertEqual(len(content_stacks[0].frames), 10)

    self.assertTrue(
        all('bar' in image_hashes[f] for f in content_stacks[1].frames))
    self.assertEqual(len(content_stacks[1].frames), 10)

  def test_multiple_exposures(self):
    set_default_context()

    image_hashes, fits_headers, plate_solutions = make_synthetic_frames(
        ra=100, dec=50, prefix='foo', number=5, dither=0, exposure=300)
    image_hashes, fits_headers, plate_solutions = make_synthetic_frames(
        ra=100,
        dec=50,
        prefix='bar',
        number=7,
        dither=0,
        exposure=120,
        image_hashes=image_hashes,
        fits_headers=fits_headers,
        plate_solutions=plate_solutions)

    context.image_hashes = image_hashes
    context.fits_headers = fits_headers
    context.plate_solutions = plate_solutions

    ray.get(find_components_by_cosine_similarity.remote())
    ray.get(identify_stacks.remote())

    image_hashes = context.image_hashes
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
    self.assertTrue(all('bar' in image_hashes[f] for f in stack.frames))
    self.assertEqual(len(stack.frames), 7)
    self.assertEqual(stack.exposure_s, 120.0)
    self.assertEqual(stack.integration_s, 7 * 120)

    stack = content_stacks[1]
    self.assertTrue(all('foo' in image_hashes[f] for f in stack.frames))
    self.assertEqual(len(stack.frames), 5)
    self.assertEqual(stack.exposure_s, 300.0)
    self.assertEqual(stack.integration_s, 5 * 300)


if __name__ == '__main__':
  unittest.main()
