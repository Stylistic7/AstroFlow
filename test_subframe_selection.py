import ray_local

import datetime
import unittest
import tempfile
import ray

from context import context
from cache import cache
from make_test_frames import make_synthetic_frames
import file_io
from stack_identification import identify_stacks
from spatial_association import find_components_by_cosine_similarity
import subframe_selection
from make_test_context import make_test_context

# TODO: These tests don't actually test the subframe selection at all. The
# stack identification step no longer eliminates bad frames from the stack.
# Instead, we're going to do subframe selection right before stacking.


class TestSubframeSelection(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def test_star_count_fraction(self):
    image_hashes, fits_headers, plate_solutions = make_synthetic_frames(
        prefix='light_clear',
        start_datetime=datetime.datetime(2023, 12, 6, 16, 29, 43),
        number=5,
        image_type='Light',
        fill_value=100,
        instrument='ZWO ASI6200MM Pro',
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        num_stars=100,
        dest_dir=context.input_dirs[0])
    image_hashes, fits_headers, plate_solutions = make_synthetic_frames(
        prefix='light_cloudy',
        start_datetime=datetime.datetime(2023, 12, 6, 16, 59, 43),
        number=5,
        image_type='Light',
        fill_value=100,
        instrument='ZWO ASI6200MM Pro',
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        num_stars=10,
        dest_dir=context.input_dirs[0],
        image_hashes=image_hashes,
        fits_headers=fits_headers,
        plate_solutions=plate_solutions)

    filenames = file_io.find_image_files(context.input_dirs[0])
    self.assertEqual(len(filenames), 10)

    context.image_hashes = image_hashes
    context.fits_headers = fits_headers
    context.plate_solutions = plate_solutions

    ray.get(find_components_by_cosine_similarity.remote())
    ray.get(identify_stacks.remote())

    stacks = context.stacks
    self.assertEqual(len(stacks), 2)
    stack = next(iter(s for s in stacks.values() if s.geometric_hash))
    self.assertEqual(len(stack.all_frames(stacks)), 10)

  def test_split_run(self):
    image_hashes, fits_headers, plate_solutions = make_synthetic_frames(
        prefix='light',
        start_datetime=datetime.datetime(2023, 12, 6, 16, 29, 43),
        number=15,
        image_type='Light',
        fill_value=100,
        instrument='ZWO ASI6200MM Pro',
        exposure=600,
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        num_stars=100,
        dest_dir=context.input_dirs[0])

    # The middle frame has too few stars, and should split the run into two.
    for h, f in image_hashes.items():
      if f.endswith('_00007.fit'):
        middle_frame = h
    print(f'middle_frame {image_hashes[middle_frame]}')
    plate_solutions[middle_frame].num_stars = 10

    filenames = file_io.find_image_files(context.input_dirs[0])
    self.assertEqual(len(filenames), 15)

    context.image_hashes = image_hashes
    context.fits_headers = fits_headers
    context.plate_solutions = plate_solutions

    ray.get(find_components_by_cosine_similarity.remote())
    ray.get(identify_stacks.remote())

    stacks = context.stacks
    self.assertEqual(len(stacks), 2)
    stack = next(iter(s for s in stacks.values() if s.geometric_hash))
    self.assertEqual(len(stack.all_frames(stacks)), 15)


if __name__ == "__main__":
  unittest.main()
