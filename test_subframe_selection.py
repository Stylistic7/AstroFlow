import ray_local

import datetime
import unittest
import tempfile
import ray

import ray_diskcache
from context import Context
from cache import cache
from make_test_frames import make_synthetic_frames
import file_io
from stack_identification import identify_stacks
from spatial_association import find_components_by_cosine_similarity
import subframe_selection
from make_test_context import make_test_context


class TestSubframeSelection(unittest.TestCase):

  def setUp(self):
    self.context = make_test_context()

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
        dest_dir=self.context.input_dirs[0])
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
        dest_dir=self.context.input_dirs[0],
        image_hashes=image_hashes,
        fits_headers=fits_headers,
        plate_solutions=plate_solutions)

    filenames = file_io.find_image_files(self.context.input_dirs[0])
    self.assertEqual(len(filenames), 10)

    self.context.image_hashes = ray.put(image_hashes)
    self.context.fits_headers = ray.put(fits_headers)
    self.context.plate_solutions = ray.put(plate_solutions)

    self.context, _ = ray.get(
        find_components_by_cosine_similarity.remote(self.context))
    self.context, _ = ray.get(identify_stacks.remote(self.context))

    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 1)
    stack = next(iter(stacks.values()))

    self.assertEqual(len(stack.runs), 1)
    light_frames = [f for run in stack.runs.values() for f in run.frames]
    self.assertEqual(len(light_frames), 5)

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
        dest_dir=self.context.input_dirs[0])

    # The middle frame has too few stars, and should split the run into two.
    for h, f in image_hashes.items():
      if f.endswith('_00007.fit'):
        middle_frame = h
    print(f'middle_frame {image_hashes[middle_frame]}')
    plate_solutions[middle_frame].num_stars = 10

    filenames = file_io.find_image_files(self.context.input_dirs[0])
    self.assertEqual(len(filenames), 15)

    self.context.image_hashes = ray.put(image_hashes)
    self.context.fits_headers = ray.put(fits_headers)
    self.context.plate_solutions = ray.put(plate_solutions)

    self.context, _ = ray.get(
        find_components_by_cosine_similarity.remote(self.context))
    self.context, _ = ray.get(identify_stacks.remote(self.context))

    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 1)
    stack = next(iter(stacks.values()))

    self.assertEqual(len(stack.runs), 2)
    light_frames = [f for run in stack.runs.values() for f in run.frames]
    self.assertEqual(len(light_frames), 14)


if __name__ == "__main__":
  unittest.main()
