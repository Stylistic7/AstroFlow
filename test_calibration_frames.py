import ray_local

import datetime
import unittest
import tempfile
import ray
from cache import cache

from make_test_frames import make_synthetic_frames, copy_testdata_frames
import ray_diskcache
from context import Context
import file_io
from compute_graph import run_compute_graph
from image_metadata import scatter_compute_image_metadata
from spatial_association import find_components_by_cosine_similarity
from stack_identification import identify_stacks
from calibration_frames import associate_calibration_frames


@ray.remote
def _compute_graph_fragment(context, new_filenames):
  context = ray.get(
      scatter_compute_image_metadata.remote(context, new_filenames))
  context = ray.get(find_components_by_cosine_similarity.remote(context))
  context = ray.get(identify_stacks.remote(context))
  context = ray.get(associate_calibration_frames.remote(context))

  return context


# Select only the toplevel stacks, which contain only light frames.
def _get_light_stacks(stacks, fits_headers):
  return [
      s for s in stacks.values() if all(
          file_io.is_light_frame(fits_headers[f])
          for r in s.runs
          for f in r.frames)
  ]


class TestCalibrationFrames(unittest.TestCase):

  def setUp(self):
    ray_diskcache.clear()
    cache.clear()

    # Start with an empty context.
    self.context = Context()
    self.context.image_hashes = ray.put({})
    self.context.fits_headers = ray.put({})
    self.context.plate_solutions = ray.put({})

    self.input_dir = tempfile.TemporaryDirectory()
    self.context.input_dirs.append(self.input_dir.name)

    self.output_dir = tempfile.TemporaryDirectory()
    self.context.output_dir = self.output_dir.name

  def test_stack_dependency_graph(self):
    # These runs of dark and bias frames will be used by both runs of light
    # frames.
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

    # The first run of light frames and its associated flats.
    copy_testdata_frames(prefix='light_run_0',
                         target_name='NGC7635',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 16, 29, 43),
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

    # The second run of light frames and its associated flats.
    copy_testdata_frames(prefix='light_run_1',
                         target_name='NGC7635',
                         start_datetime=datetime.datetime(
                             2024, 2, 7, 16, 29, 43),
                         dest_dir=self.context.input_dirs[0])
    make_synthetic_frames(prefix='flat_run_1',
                          number=13,
                          image_type='Flat',
                          start_datetime=datetime.datetime(
                              2024, 2, 7, 18, 29, 43),
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=self.context.input_dirs[0])

    filenames = file_io.find_image_files(self.context.input_dirs[0])
    self.assertEqual(len(filenames), 42)

    self.context = ray.get(
        _compute_graph_fragment.remote(self.context, filenames))

    fits_headers = ray.get(self.context.fits_headers)
    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 5)

    light_stacks = _get_light_stacks(stacks, fits_headers)
    self.assertEqual(len(light_stacks), 1)

  def test_one_light_run_one_dark_run(self):
    copy_testdata_frames(target_name='NGC7635',
                         dest_dir=self.context.input_dirs[0])
    make_synthetic_frames(number=5,
                          image_type='Dark',
                          fill_value=100,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=self.context.input_dirs[0])

    filenames = file_io.find_image_files(self.context.input_dirs[0])
    self.assertEqual(len(filenames), 8)

    self.context = ray.get(
        _compute_graph_fragment.remote(self.context, filenames))

    fits_headers = ray.get(self.context.fits_headers)
    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 2)

    light_stacks = _get_light_stacks(stacks, fits_headers)
    self.assertEqual(len(light_stacks), 1)

  def test_flats_from_different_camera(self):
    # There are lights from two different cameras. This should result
    # in two stacks, with mutually-exclusive sets of flat frames.
    #
    # The NGC7635 images were taken in October 2023, while the M42 images
    # were taken in November 2023. The stacks should be grouped in
    # chronological order.
    copy_testdata_frames(target_name='NGC7635',
                         dest_dir=self.context.input_dirs[0])
    make_synthetic_frames(number=5,
                          image_type='Dark',
                          fill_value=100,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=self.context.input_dirs[0])
    make_synthetic_frames(number=7,
                          image_type='Bias',
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=self.context.input_dirs[0])
    make_synthetic_frames(number=11,
                          image_type='Flat',
                          fill_value=4096,
                          exposure=1,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=self.context.input_dirs[0])

    copy_testdata_frames(target_name='M42', dest_dir=self.context.input_dirs[0])
    make_synthetic_frames(number=13,
                          image_type='Dark',
                          fill_value=100,
                          instrument='ZWO ASI2600MM Pro',
                          width_pixels=6248,
                          height_pixels=4176,
                          temperature=-20,
                          dest_dir=self.context.input_dirs[0])
    make_synthetic_frames(number=17,
                          image_type='Bias',
                          fill_value=50,
                          instrument='ZWO ASI2600MM Pro',
                          width_pixels=6248,
                          height_pixels=4176,
                          temperature=-20,
                          dest_dir=self.context.input_dirs[0])
    make_synthetic_frames(number=19,
                          image_type='Flat',
                          fill_value=4096,
                          exposure=1,
                          instrument='ZWO ASI2600MM Pro',
                          width_pixels=6248,
                          height_pixels=4176,
                          temperature=-20,
                          dest_dir=self.context.input_dirs[0])

    filenames = file_io.find_image_files(self.context.input_dirs[0])
    self.assertEqual(len(filenames), 78)

    self.context = ray.get(
        _compute_graph_fragment.remote(self.context, filenames))

    fits_headers = ray.get(self.context.fits_headers)
    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 8)

    light_stacks = _get_light_stacks(stacks, fits_headers)
    self.assertEqual(len(light_stacks), 2)

  def test_dark_scaling(self):
    # The light frames are 180 seconds, but the dark frames are 60 or 300
    # seconds. AstroFlow should pick the closest darks that are equal to or
    # longer than the lights. One group of dark frames will never be stacked.
    copy_testdata_frames(target_name='NGC7635',
                         dest_dir=self.context.input_dirs[0])
    make_synthetic_frames(number=5,
                          image_type='Dark',
                          fill_value=100,
                          exposure=60,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=self.context.input_dirs[0])
    make_synthetic_frames(number=7,
                          image_type='Dark',
                          fill_value=100,
                          exposure=300,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=self.context.input_dirs[0])
    make_synthetic_frames(number=11,
                          image_type='Bias',
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=self.context.input_dirs[0])
    make_synthetic_frames(number=13,
                          image_type='Flat',
                          fill_value=4096,
                          exposure=1,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=self.context.input_dirs[0])

    filenames = file_io.find_image_files(self.context.input_dirs[0])
    self.assertEqual(len(filenames), 39)

    self.context = ray.get(
        _compute_graph_fragment.remote(self.context, filenames))

    fits_headers = ray.get(self.context.fits_headers)
    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 4)

    light_stacks = _get_light_stacks(stacks, fits_headers)
    self.assertEqual(len(light_stacks), 1)

  def test_temperature_mismatch(self):
    # The light frames are at temperature -13, but the darks are at -15. This
    # difference is small enough for the frames to still be associated.
    copy_testdata_frames(target_name='NGC7635',
                         dest_dir=self.context.input_dirs[0])
    make_synthetic_frames(number=5,
                          image_type='Dark',
                          fill_value=100,
                          exposure=300,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=self.context.input_dirs[0])

    filenames = file_io.find_image_files(self.context.input_dirs[0])
    self.assertEqual(len(filenames), 8)

    self.context = ray.get(
        _compute_graph_fragment.remote(self.context, filenames))

    fits_headers = ray.get(self.context.fits_headers)
    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 2)

    light_stacks = _get_light_stacks(stacks, fits_headers)
    self.assertEqual(len(light_stacks), 1)

  def test_stack_calibration_frames(self):
    copy_testdata_frames(target_name='NGC7635',
                         dest_dir=self.context.input_dirs[0])
    make_synthetic_frames(number=5,
                          image_type='Bias',
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-13,
                          dest_dir=self.context.input_dirs[0])
    make_synthetic_frames(number=7,
                          image_type='Dark',
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=self.context.input_dirs[0])
    make_synthetic_frames(number=11,
                          image_type='Flat',
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-17,
                          dest_dir=self.context.input_dirs[0])

    filenames = file_io.find_image_files(self.context.input_dirs[0])
    self.assertEqual(len(filenames), 26)

    self.context = ray.get(
        _compute_graph_fragment.remote(self.context, filenames))

    fits_headers = ray.get(self.context.fits_headers)
    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 4)

    light_stacks = _get_light_stacks(stacks, fits_headers)
    self.assertEqual(len(light_stacks), 1)

  def test_multiple_filters_for_flats(self):
    # Only one run of flat frames applies to the light frames, which use
    # the H filter.
    copy_testdata_frames(target_name='NGC7635',
                         dest_dir=self.context.input_dirs[0])
    make_synthetic_frames(number=5,
                          image_type='Flat',
                          filter='H',
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-10,
                          dest_dir=self.context.input_dirs[0])
    make_synthetic_frames(number=7,
                          image_type='Flat',
                          filter='S',
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-10,
                          dest_dir=self.context.input_dirs[0])

    filenames = file_io.find_image_files(self.context.input_dirs[0])
    self.assertEqual(len(filenames), 15)

    self.context = ray.get(
        _compute_graph_fragment.remote(self.context, filenames))

    fits_headers = ray.get(self.context.fits_headers)
    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 2)

    light_stacks = _get_light_stacks(stacks, fits_headers)
    self.assertEqual(len(light_stacks), 1)


if __name__ == "__main__":
  unittest.main()
