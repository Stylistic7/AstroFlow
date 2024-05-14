import ray_local

import datetime
import unittest
import tempfile
import ray
from cache import cache
import datetime
import pytz

from make_test_frames import make_synthetic_frames, copy_testdata_frames
import ray_diskcache
from context import Context
import file_io
from new_dependency_graph import run_graph, make_graph_from_list
from image_metadata import scatter_compute_image_metadata
from spatial_association import find_components_by_cosine_similarity
from stack_identification import identify_stacks
from calibration_frames import associate_calibration_frames
from make_test_context import make_test_context
from hashing import hash_list


def _build_compute_graph():
  return make_graph_from_list([(scatter_compute_image_metadata, 'all'),
                               find_components_by_cosine_similarity,
                               identify_stacks, associate_calibration_frames])


# Select only the toplevel stacks, which contain only light frames.
def _get_light_stacks(stacks, fits_headers):
  return [
      s for s in stacks.values() if all(
          file_io.is_light_frame(fits_headers[f])
          for r in s.runs.values()
          for f in r.frames)
  ]


# Check that Stacks and Runs have all the metadata they should have.
def _check_stacks(stacks):
  for stack in stacks.values():
    assert stack.type != None
    assert stack.num_frames != None
    assert stack.exposure_s != None

    for run in stack.runs.values():
      assert run.type != None
      assert stack.num_frames != None
      assert stack.exposure_s != None


class TestCalibrationFrames(unittest.TestCase):

  def setUp(self):
    self.context = make_test_context()

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

    graph = _build_compute_graph()
    self.context, _ = run_graph(graph, self.context)

    fits_headers = ray.get(self.context.fits_headers)
    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 5)
    _check_stacks(stacks)

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

    graph = _build_compute_graph()
    self.context, _ = run_graph(graph, self.context)

    fits_headers = ray.get(self.context.fits_headers)
    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 2)
    _check_stacks(stacks)

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

    graph = _build_compute_graph()
    self.context, _ = run_graph(graph, self.context)

    fits_headers = ray.get(self.context.fits_headers)
    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 8)
    _check_stacks(stacks)

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

    graph = _build_compute_graph()
    self.context, _ = run_graph(graph, self.context)

    fits_headers = ray.get(self.context.fits_headers)
    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 4)
    _check_stacks(stacks)

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

    graph = _build_compute_graph()
    self.context, _ = run_graph(graph, self.context)

    fits_headers = ray.get(self.context.fits_headers)
    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 2)
    _check_stacks(stacks)

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

    graph = _build_compute_graph()
    self.context, _ = run_graph(graph, self.context)

    fits_headers = ray.get(self.context.fits_headers)
    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 4)
    _check_stacks(stacks)

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

    graph = _build_compute_graph()
    self.context, _ = run_graph(graph, self.context)

    fits_headers = ray.get(self.context.fits_headers)
    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 2)
    _check_stacks(stacks)

    light_stacks = _get_light_stacks(stacks, fits_headers)
    self.assertEqual(len(light_stacks), 1)

  def test_flat_frame_selection(self):
    # There are two runs of flat frames, but the user has selected the second
    # one, which is much more distant chronologically.
    make_synthetic_frames(prefix='flat_run_0',
                          number=7,
                          start_datetime=datetime.datetime(
                              2023, 11, 6, 16, 29, 43),
                          image_type='Flat',
                          fill_value=4096,
                          exposure=1,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=self.context.input_dirs[0])
    copy_testdata_frames(prefix='light_run',
                         target_name='NGC7635',
                         start_datetime=datetime.datetime(
                             2023, 12, 6, 16, 29, 43),
                         dest_dir=self.context.input_dirs[0],
                         add_noise=False)
    make_synthetic_frames(prefix='flat_run_1',
                          number=5,
                          start_datetime=datetime.datetime(
                              2023, 12, 6, 17, 29, 43),
                          image_type='Flat',
                          fill_value=4096,
                          exposure=1,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=self.context.input_dirs[0])

    filenames = file_io.find_image_files(self.context.input_dirs[0])
    self.assertEqual(len(filenames), 15)

    # Run the graph the first time. It will use the default flat run, which
    # has prefix `flat_run_1` because it is closer chronologically.
    print('### Running compute graph for the first time')
    graph = _build_compute_graph()
    self.context, _ = run_graph(graph, self.context)
    image_hashes = ray.get(self.context.image_hashes)
    fits_headers = ray.get(self.context.fits_headers)
    stacks = ray.get(self.context.stacks)

    # Look through the stacks, and find the stacks for the two runs of flat
    # frames.
    def find_stack(filename_fragment):
      return next(stack_id for stack_id, stack in stacks.items()
                  if all(filename_fragment in image_hashes[f]
                         for run in stack.runs.values()
                         for f in run.frames))

    first_flat_stack = find_stack('flat_run_0')
    second_flat_stack = find_stack('flat_run_1')
    print(f'### first_flat_stack {first_flat_stack}')
    print(f'### second_flat_stack {second_flat_stack}')

    first_light_run_hash = hash_list(
        [k for k, v in image_hashes.items() if 'light_run' in v])
    print(f'### first_light_run_hash', first_light_run_hash)

    light_stacks = _get_light_stacks(stacks, fits_headers)
    self.assertEqual(len(light_stacks), 1)
    stack = next(iter(light_stacks))
    assert len(stack.runs) == 1
    run = next(iter(stack.runs.values()))
    assert run.content_hash == first_light_run_hash
    assert run.stack_dependencies == {
        'flat': second_flat_stack
    }, f'run.stack_dependencies {run.stack_dependencies}'

    # Configure the light run to use the other flat run, which has prefix
    # `flat_run_0`.
    ray_diskcache.set(('light_run_settings', first_light_run_hash),
                      {'flat': first_flat_stack})

    # Run the graph again, with the settings applied.
    print('### Running compute graph for the second time')
    graph = _build_compute_graph()
    self.context, _ = run_graph(graph, self.context)
    image_hashes = ray.get(self.context.image_hashes)
    fits_headers = ray.get(self.context.fits_headers)
    stacks = ray.get(self.context.stacks)

    self.assertEqual(len(stacks), 3)
    _check_stacks(stacks)

    light_stacks = _get_light_stacks(stacks, fits_headers)
    self.assertEqual(len(light_stacks), 1)
    stack = next(iter(light_stacks))
    assert len(stack.runs) == 1
    run = next(iter(stack.runs.values()))
    assert run.content_hash == first_light_run_hash
    assert run.stack_dependencies == {
        'flat': first_flat_stack
    }, f'run.stack_dependencies {run.stack_dependencies}'


if __name__ == "__main__":
  unittest.main()
