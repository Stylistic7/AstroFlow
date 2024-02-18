import unittest
import ray

from context import Context
from make_test_frames import make_synthetic_frames
from stack_identification import identify_stacks
from spatial_association import find_components_by_cosine_similarity


class TestStackIdentification(unittest.TestCase):

  def test_one_perfect_stack(self):
    image_hashes, fits_headers, plate_solutions = make_synthetic_frames(
        ra=100, dec=50, prefix='foo', number=10, dither=0)

    context = Context(image_hashes=ray.put(image_hashes),
                      fits_headers=ray.put(fits_headers),
                      plate_solutions=ray.put(plate_solutions))

    context = ray.get(find_components_by_cosine_similarity.remote(context))
    context = ray.get(identify_stacks.remote(context))

    stacks = ray.get(context.stacks)
    self.assertEqual(len(stacks), 1)

    stack = stacks[0]
    self.assertEqual(len(stack.runs), 1)
    self.assertEqual(len(stack.runs[0].frames), 10)
    self.assertEqual(stack.center, [100, 50])
    self.assertEqual(stack.image_scale, 1.0)

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

    context = Context(image_hashes=ray.put(image_hashes),
                      fits_headers=ray.put(fits_headers),
                      plate_solutions=ray.put(plate_solutions))

    context = ray.get(find_components_by_cosine_similarity.remote(context))
    context = ray.get(identify_stacks.remote(context))

    stacks = ray.get(context.stacks)
    self.assertEqual(len(stacks), 2)

    stack = stacks[0]
    self.assertEqual(len(stack.runs), 1)
    run = stack.runs[0]
    self.assertTrue(all('foo' in f for f in run.frames))
    self.assertEqual(len(run.frames), 10)
    self.assertEqual(stack.center, [100, 50])
    self.assertEqual(stack.image_scale, 1.0)

    stack = stacks[1]
    self.assertEqual(len(stack.runs), 1)
    run = stack.runs[0]
    self.assertTrue(all('bar' in f for f in run.frames))
    self.assertEqual(len(run.frames), 10)
    self.assertEqual(stack.center, [100, 50])
    self.assertEqual(stack.image_scale, 2.0)

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

    context = Context(image_hashes=ray.put(image_hashes),
                      fits_headers=ray.put(fits_headers),
                      plate_solutions=ray.put(plate_solutions))

    context = ray.get(find_components_by_cosine_similarity.remote(context))
    context = ray.get(identify_stacks.remote(context))

    stacks = ray.get(context.stacks)
    self.assertEqual(len(stacks), 2)

    stack = stacks[0]
    self.assertEqual(len(stack.runs), 1)
    run = stack.runs[0]
    self.assertTrue(all('foo' in f for f in run.frames))
    self.assertEqual(len(run.frames), 10)
    self.assertEqual(stack.center, [100, 50])
    self.assertEqual(stack.width_pixels, 3000)
    self.assertEqual(stack.height_pixels, 2000)
    self.assertEqual(stack.image_scale, 1.0)

    stack = stacks[1]
    self.assertEqual(len(stack.runs), 1)
    run = stack.runs[0]
    self.assertTrue(all('bar' in f for f in run.frames))
    self.assertEqual(len(run.frames), 10)
    self.assertEqual(stack.center, [100, 50])
    self.assertEqual(stack.width_pixels, 6000)
    self.assertEqual(stack.height_pixels, 4000)
    self.assertEqual(stack.image_scale, 1.0)


if __name__ == '__main__':
  unittest.main()
