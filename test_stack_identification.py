import unittest
import tempfile

from make_test_frames import make_test_frames
import stack_identification
import spatial_association


class TestStackIdentification(unittest.TestCase):
  def test_one_perfect_stack(self):
    image_hashes, fits_headers, plate_solutions = make_test_frames(
        ra=100, dec=50, prefix='foo', number=10, dither=0)

    light_frames = list(plate_solutions.keys())
    spatial_components = spatial_association.find_components_by_cosine_similarity(
        light_frames, plate_solutions, fits_headers)

    stacks = stack_identification.identify_stacks(
        spatial_components, plate_solutions, fits_headers)
    self.assertEqual(len(stacks), 1)

    stack = stacks[0]
    self.assertEqual(len(stack.light_frames), 10)
    self.assertEqual(stack.center, [100, 50])
    self.assertEqual(stack.image_scale, 1.0)

  def test_two_stacks_different_image_scales(self):
    image_hashes, fits_headers, plate_solutions = make_test_frames(
        ra=100, dec=50, prefix='foo', number=10, dither=0)
    image_hashes, fits_headers, plate_solutions = make_test_frames(
        ra=100,
        dec=50,
        prefix='bar',
        number=10,
        dither=0,
        image_scale=2.0,
        image_hashes=image_hashes,
        fits_headers=fits_headers,
        plate_solutions=plate_solutions)

    light_frames = list(plate_solutions.keys())
    spatial_components = spatial_association.find_components_by_cosine_similarity(
        light_frames, plate_solutions, fits_headers)

    stacks = stack_identification.identify_stacks(
        spatial_components, plate_solutions, fits_headers)
    print('### stacks', stacks)
    self.assertEqual(len(stacks), 2)

    stack = stacks[0]
    self.assertTrue(all('foo' in f for f in stack.light_frames))
    self.assertEqual(len(stack.light_frames), 10)
    self.assertEqual(stack.center, [100, 50])
    self.assertEqual(stack.image_scale, 1.0)

    stack = stacks[1]
    self.assertTrue(all('bar' in f for f in stack.light_frames))
    self.assertEqual(len(stack.light_frames), 10)
    self.assertEqual(stack.center, [100, 50])
    self.assertEqual(stack.image_scale, 2.0)

  def test_two_stacks_different_image_sizes(self):
    image_hashes, fits_headers, plate_solutions = make_test_frames(
        ra=100,
        dec=50,
        prefix='foo',
        number=10,
        dither=0,
        width_pixels=3000,
        height_pixels=2000,
    )
    image_hashes, fits_headers, plate_solutions = make_test_frames(
        ra=100,
        dec=50,
        prefix='bar',
        number=10,
        dither=0,
        width_pixels=6000,
        height_pixels=4000,
        image_hashes=image_hashes,
        fits_headers=fits_headers,
        plate_solutions=plate_solutions)

    light_frames = list(plate_solutions.keys())
    spatial_components = spatial_association.find_components_by_cosine_similarity(
        light_frames, plate_solutions, fits_headers)

    stacks = stack_identification.identify_stacks(
        spatial_components, plate_solutions, fits_headers)
    self.assertEqual(len(stacks), 2)

    stack = stacks[0]
    self.assertTrue(all('foo' in f for f in stack.light_frames))
    self.assertEqual(len(stack.light_frames), 10)
    self.assertEqual(stack.center, [100, 50])
    self.assertEqual(stack.width_pixels, 3000)
    self.assertEqual(stack.height_pixels, 2000)
    self.assertEqual(stack.image_scale, 1.0)

    stack = stacks[1]
    self.assertTrue(all('bar' in f for f in stack.light_frames))
    self.assertEqual(len(stack.light_frames), 10)
    self.assertEqual(stack.center, [100, 50])
    self.assertEqual(stack.width_pixels, 6000)
    self.assertEqual(stack.height_pixels, 4000)
    self.assertEqual(stack.image_scale, 1.0)


if __name__ == '__main__':
  unittest.main()
