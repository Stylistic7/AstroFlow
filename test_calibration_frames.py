import ray_local

import datetime
import unittest
import tempfile
import ray
from cache import cache

from stack import Stack
from make_test_frames import make_synthetic_frames, copy_testdata_frames
import ray_diskcache
from context import Context
import file_io
from compute_graph import run_compute_graph
from image_metadata import scatter_compute_image_metadata
from spatial_association import find_components_by_cosine_similarity_ray
from stack_identification import identify_stacks_ray
from calibration_frames import associate_calibration_frames
from stacking import scatter_stack_subframes, construct_graph, group_parallel_tasks


@ray.remote
def _compute_graph_fragment(context, new_filenames):
  context = ray.get(
      scatter_compute_image_metadata.remote(context, new_filenames))
  context = ray.get(find_components_by_cosine_similarity_ray.remote(context))
  context = ray.get(identify_stacks_ray.remote(context))
  context = ray.get(associate_calibration_frames.remote(context))

  return context


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

    self.context = ray.get(run_compute_graph.remote(self.context, filenames))

    print('### END OF TEST')
    stacks = ray.get(self.context.stacks)
    print('### stacks', stacks)

    image_hashes = ray.get(self.context.image_hashes)
    for h, f in image_hashes.items():
      print(f'### image_hash {h} -> {f}')

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

    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 1)
    stack_iterator = iter(stacks)
    stack = stacks[next(stack_iterator)]

  def test_flats_from_different_camera(self):
    # There are lights flats from two different cameras. This should result
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

    filenames = file_io.find_image_files(self.context.output_dir)
    self.assertEqual(len(filenames), 78)

    self.context = ray.get(run_compute_graph.remote(self.context, filenames))

    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 2)
    stack_iterator = iter(stacks)

    # TODO: Don't depend on the order of insertion here.
    stack = stacks[next(stack_iterator)]
    print('### First stack is', stack.target)
    self.assertEqual(len(stack.light_frames), 3)
    self.assertEqual(len(stack.dark_frames), 5)
    self.assertEqual(len(stack.bias_frames), 7)
    self.assertEqual(len(stack.flat_frames), 11)
    self.assertTrue(('stack_thumbnail_image', stack.id,
                     stack.content_hash) in cache)
    self.assertTrue(('stack_thumbnail_video', stack.id,
                     stack.content_hash) in cache)

    stack = stacks[next(stack_iterator)]
    print('### Second stack is', stack.target)
    self.assertEqual(len(stack.light_frames), 3)
    self.assertEqual(len(stack.dark_frames), 13)
    self.assertEqual(len(stack.bias_frames), 17)
    self.assertEqual(len(stack.flat_frames), 19)
    self.assertTrue(('stack_thumbnail_image', stack.id,
                     stack.content_hash) in cache)
    self.assertTrue(('stack_thumbnail_video', stack.id,
                     stack.content_hash) in cache)

  def test_dark_scaling(self):
    # The light frames are 180 seconds, but the dark frames are 60 or 300
    # seconds. AstroFlow should pick the closest darks that are equal to or
    # longer than the lights.
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

    filenames = file_io.find_image_files(self.context.output_dir)
    self.assertEqual(len(filenames), 39)

    self.context = ray.get(run_compute_graph.remote(self.context, filenames))

    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 1)

    stack = stacks[next(iter(stacks))]
    self.assertEqual(len(stack.light_frames), 3)
    self.assertEqual(len(stack.dark_frames), 7)
    self.assertEqual(len(stack.bias_frames), 11)
    self.assertEqual(len(stack.flat_frames), 13)

    self.assertTrue(('stack_thumbnail_image', stack.id,
                     stack.content_hash) in cache)
    self.assertTrue(('stack_thumbnail_video', stack.id,
                     stack.content_hash) in cache)

  def test_temperature_mismatch(self):
    # The light frames are at temperature -15, but the darks are at -18. This
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
                          temperature=-18,
                          dest_dir=self.context.input_dirs[0])

    filenames = file_io.find_image_files(self.context.output_dir)
    self.assertEqual(len(filenames), 8)

    self.context = ray.get(run_compute_graph.remote(self.context, filenames))

    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 1)

    stack = stacks[next(iter(stacks))]
    self.assertEqual(len(stack.light_frames), 3)
    self.assertEqual(len(stack.dark_frames), 5)
    self.assertEqual(len(stack.bias_frames), 0)
    self.assertEqual(len(stack.flat_frames), 0)

    self.assertTrue(('stack_thumbnail_image', stack.id,
                     stack.content_hash) in cache)
    self.assertTrue(('stack_thumbnail_video', stack.id,
                     stack.content_hash) in cache)

  def test_stack_calibration_frames(self):
    make_synthetic_frames(number=5,
                          image_type='Bias',
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-18,
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
                          temperature=-10,
                          dest_dir=self.context.input_dirs[0])

    filenames = file_io.find_image_files(self.context.output_dir)
    self.assertEqual(len(filenames), 23)

    context = ray.get(run_compute_graph.remote(self.context, filenames))

    output_files = file_io.find_image_files(context.output_dir)
    self.assertEqual(len(output_files), 3)

    fits_headers = ray.get(context.fits_headers)
    master_biases = [
        f for f, h in fits_headers.items() if h['IMAGETYP'] == 'Master Bias'
    ]
    self.assertEqual(len(master_biases), 1)

    master_darks = [
        f for f, h in fits_headers.items() if h['IMAGETYP'] == 'Master Dark'
    ]
    self.assertEqual(len(master_darks), 1)

    master_flats = [
        f for f, h in fits_headers.items() if h['IMAGETYP'] == 'Master Flat'
    ]
    self.assertEqual(len(master_flats), 1)

  def test_multiple_filters_for_flats(self):
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

    filenames = file_io.find_image_files(self.context.output_dir)
    self.assertEqual(len(filenames), 12)

    context = ray.get(run_compute_graph.remote(self.context, filenames))

    output_files = file_io.find_image_files(context.output_dir)
    self.assertEqual(len(output_files), 2)

    fits_headers = ray.get(context.fits_headers)

    master_flats = [
        f for f, h in fits_headers.items() if h['IMAGETYP'] == 'Master Flat'
    ]

    # TODO: Calibration masters will only be made when there's a stack that
    # references them. Use just part of the compute graph here.
    # self.assertEqual(len(master_flats), 2)


if __name__ == "__main__":
  unittest.main()
