from ray_infra import ray_local

import datetime
import unittest

from compute_graph import build_compute_graph
from testing import make_test_context, make_synthetic_frames, copy_testdata_frames

from stacks.simple_stack import SimpleStack
from stacks.composite_stack import CompositeStack
from stacks.types import StackHash
from context import context
from file_io.find_image_files import find_image_files
from dependency_graph import run_graph
from stacks.get_stacks import get_toplevel_stacks
from metadata.actor import get_metadata
from customization.settings import Settings, StackSettings


class TestCalibrationFrames(unittest.TestCase):

  def setUp(self) -> None:
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def tearDown(self) -> None:
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_stack_dependency_graph(self) -> None:
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
                          dest_dir=context.input_dirs[0])
    make_synthetic_frames(prefix='bias',
                          number=7,
                          image_type='Bias',
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=context.input_dirs[0])

    # The first run of light frames and its associated flats.
    copy_testdata_frames(prefix='light_run_0',
                         target_name='NGC7635',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 16, 29, 43),
                         dest_dir=context.input_dirs[0])
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
                          dest_dir=context.input_dirs[0])

    # The second run of light frames and its associated flats.
    copy_testdata_frames(prefix='light_run_1',
                         target_name='NGC7635',
                         start_datetime=datetime.datetime(
                             2024, 2, 7, 16, 29, 43),
                         dest_dir=context.input_dirs[0])
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
                          dest_dir=context.input_dirs[0])

    filenames = find_image_files(context.input_dirs[0])
    self.assertEqual(len(filenames), 42)

    run_graph(
        build_compute_graph(regroup=True,
                            make_previews=False,
                            run_main_processing_graph=False))

    stacks = context.stacks
    self.assertEqual(len(stacks), 1)
    self.assertEqual(len(context.calibration_stacks), 4)

    light_stacks = [s for s in stacks.values() if s.image_type == 'light']
    self.assertEqual(len(light_stacks), 1)

  def test_one_light_run_one_dark_run(self) -> None:
    copy_testdata_frames(target_name='NGC7635', dest_dir=context.input_dirs[0])
    make_synthetic_frames(number=5,
                          image_type='Dark',
                          fill_value=100,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=context.input_dirs[0])

    filenames = find_image_files(context.input_dirs[0])
    self.assertEqual(len(filenames), 8)

    run_graph(
        build_compute_graph(regroup=True,
                            make_previews=False,
                            run_main_processing_graph=False))

    stacks = context.stacks
    self.assertEqual(len(stacks), 1)
    self.assertEqual(len(context.calibration_stacks), 1)

    light_stacks = [s for s in stacks.values() if s.image_type == 'light']
    self.assertEqual(len(light_stacks), 1)

  def test_flats_from_different_camera(self) -> None:
    # There are lights from two different cameras. This should result
    # in two stacks, with mutually-exclusive sets of flat frames.
    #
    # The NGC7635 images were taken in October 2023, while the M42 images
    # were taken in November 2023. The stacks should be grouped in
    # chronological order.
    copy_testdata_frames(target_name='NGC7635', dest_dir=context.input_dirs[0])
    make_synthetic_frames(number=5,
                          image_type='Dark',
                          fill_value=100,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=context.input_dirs[0])
    make_synthetic_frames(number=7,
                          image_type='Bias',
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=context.input_dirs[0])
    make_synthetic_frames(number=11,
                          image_type='Flat',
                          fill_value=4096,
                          exposure=1,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=context.input_dirs[0])

    copy_testdata_frames(target_name='M42', dest_dir=context.input_dirs[0])
    make_synthetic_frames(number=13,
                          image_type='Dark',
                          fill_value=100,
                          instrument='ZWO ASI2600MM Pro',
                          width_pixels=6248,
                          height_pixels=4176,
                          temperature=-20,
                          dest_dir=context.input_dirs[0])
    make_synthetic_frames(number=17,
                          image_type='Bias',
                          fill_value=50,
                          instrument='ZWO ASI2600MM Pro',
                          width_pixels=6248,
                          height_pixels=4176,
                          temperature=-20,
                          dest_dir=context.input_dirs[0])
    make_synthetic_frames(number=19,
                          image_type='Flat',
                          fill_value=4096,
                          exposure=1,
                          instrument='ZWO ASI2600MM Pro',
                          width_pixels=6248,
                          height_pixels=4176,
                          temperature=-20,
                          dest_dir=context.input_dirs[0])

    filenames = find_image_files(context.input_dirs[0])
    self.assertEqual(len(filenames), 78)

    run_graph(
        build_compute_graph(regroup=True,
                            make_previews=False,
                            run_main_processing_graph=False))

    stacks = context.stacks
    self.assertEqual(len(stacks), 2)
    self.assertEqual(len(context.calibration_stacks), 6)

    light_stacks = [
        s for s in stacks.values()
        if s.image_type == 'light' and isinstance(s, CompositeStack)
    ]
    self.assertEqual(len(light_stacks), 2)

  def test_dark_scaling(self) -> None:
    # The light frames are 180 seconds, but the dark frames are 60 or 300
    # seconds. AstroFlow should pick the closest darks that are equal to or
    # longer than the lights. One group of dark frames will never be stacked.
    copy_testdata_frames(target_name='NGC7635', dest_dir=context.input_dirs[0])
    make_synthetic_frames(number=5,
                          image_type='Dark',
                          fill_value=100,
                          exposure=60,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=context.input_dirs[0])
    make_synthetic_frames(number=7,
                          image_type='Dark',
                          fill_value=100,
                          exposure=300,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=context.input_dirs[0])
    make_synthetic_frames(number=11,
                          image_type='Bias',
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=context.input_dirs[0])
    make_synthetic_frames(number=13,
                          image_type='Flat',
                          fill_value=4096,
                          exposure=1,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=context.input_dirs[0])

    filenames = find_image_files(context.input_dirs[0])
    self.assertEqual(len(filenames), 39)

    run_graph(
        build_compute_graph(regroup=True,
                            make_previews=False,
                            run_main_processing_graph=False))

    stacks = context.stacks
    self.assertEqual(len(stacks), 1)
    self.assertEqual(len(context.calibration_stacks), 4)

    light_stacks = [s for s in stacks.values() if s.image_type == 'light']
    self.assertEqual(len(light_stacks), 1)

  # TODO: Test temperatures that do and do not match.
  def test_temperature_mismatch(self) -> None:
    # The light frames are at temperature -13, but the darks are at -15. This
    # difference is small enough for the frames to still be associated.
    copy_testdata_frames(target_name='NGC7635', dest_dir=context.input_dirs[0])
    make_synthetic_frames(number=5,
                          image_type='Dark',
                          fill_value=100,
                          exposure=300,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=context.input_dirs[0])

    filenames = find_image_files(context.input_dirs[0])
    self.assertEqual(len(filenames), 8)

    run_graph(
        build_compute_graph(regroup=True,
                            make_previews=False,
                            run_main_processing_graph=False))

    stacks = context.stacks
    self.assertEqual(len(stacks), 1)
    self.assertEqual(len(context.calibration_stacks), 1)

    light_stacks = [
        s for s in stacks.values()
        if isinstance(s, CompositeStack) and s.image_type == 'light'
    ]
    self.assertEqual(len(light_stacks), 1)
    stack = light_stacks[0]
    self.assertEqual(len(stack.contents), 1)
    substack = next(iter(stack.contents.values()))
    self.assertEqual(len(substack.frames), 3)
    self.assertEqual(len(substack.dependencies), 1)

  def test_stack_calibration_frames(self) -> None:
    copy_testdata_frames(target_name='NGC7635', dest_dir=context.input_dirs[0])
    make_synthetic_frames(number=5,
                          image_type='Bias',
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-13,
                          dest_dir=context.input_dirs[0])
    make_synthetic_frames(number=7,
                          image_type='Dark',
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=context.input_dirs[0])
    make_synthetic_frames(number=11,
                          image_type='Flat',
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-17,
                          dest_dir=context.input_dirs[0])

    filenames = find_image_files(context.input_dirs[0])
    self.assertEqual(len(filenames), 26)

    run_graph(
        build_compute_graph(regroup=True,
                            make_previews=False,
                            run_main_processing_graph=False))

    stacks = context.stacks
    self.assertEqual(len(stacks), 1)
    self.assertEqual(len(context.calibration_stacks), 3)

    light_stacks = [s for s in stacks.values() if s.image_type == 'light']
    self.assertEqual(len(light_stacks), 1)

  def test_multiple_filters_for_flats(self) -> None:
    # Only one run of flat frames applies to the light frames, which use
    # the H filter.
    copy_testdata_frames(target_name='NGC7635', dest_dir=context.input_dirs[0])
    make_synthetic_frames(number=5,
                          image_type='Flat',
                          filter='H',
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-10,
                          dest_dir=context.input_dirs[0])
    make_synthetic_frames(number=7,
                          image_type='Flat',
                          filter='S',
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-10,
                          dest_dir=context.input_dirs[0])

    filenames = find_image_files(context.input_dirs[0])
    self.assertEqual(len(filenames), 15)

    run_graph(
        build_compute_graph(regroup=True,
                            make_previews=False,
                            run_main_processing_graph=False))

    stacks = context.stacks
    self.assertEqual(len(stacks), 1)
    self.assertEqual(len(context.calibration_stacks), 2)

    light_stacks = [s for s in stacks.values() if s.image_type == 'light']
    self.assertEqual(len(light_stacks), 1)

  def test_flat_frame_selection(self) -> None:
    # There are two runs of flat frames, but the user has selected the first
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
                          dest_dir=context.input_dirs[0])
    copy_testdata_frames(prefix='light_run',
                         target_name='NGC7635',
                         start_datetime=datetime.datetime(
                             2023, 12, 6, 16, 29, 43),
                         dest_dir=context.input_dirs[0],
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
                          dest_dir=context.input_dirs[0])

    filenames = find_image_files(context.input_dirs[0])
    self.assertEqual(len(filenames), 15)

    # Run the graph the first time. It will use the default flat run, which
    # has prefix `flat_run_1` because it is closer chronologically.
    run_graph(
        build_compute_graph(regroup=True,
                            make_previews=False,
                            run_main_processing_graph=False))

    stacks = context.stacks

    # Look through the stacks, and find the stacks for the two runs of flat
    # frames.
    def find_cal_stack(filename_fragment: str) -> StackHash | None:
      for stack_id, stack in context.calibration_stacks.items():
        if not isinstance(stack, SimpleStack):
          continue

        filenames = get_metadata(stack.frames,
                                 transform_fn=lambda i: i.filename)
        print(f'### stack {stack} frames {stack.frames} filenames {filenames}')
        if all(filename_fragment in f for f in filenames):
          return stack_id

      return None

    first_flat_stack = find_cal_stack('flat_run_0')
    self.assertNotEqual(first_flat_stack, None)
    print(f'### first_flat_stack {first_flat_stack}')
    second_flat_stack = find_cal_stack('flat_run_1')
    self.assertNotEqual(second_flat_stack, None)
    print(f'### second_flat_stack {second_flat_stack}')

    light_stacks = [
        s for s in context.stacks.values()
        if s.image_type == 'light' and isinstance(s, CompositeStack)
    ]
    self.assertEqual(len(light_stacks), 1)
    stack = next(iter(light_stacks))
    self.assertEqual(len(stack.contents), 1)
    substack = next(iter(stack.contents.values()))
    self.assertEqual(substack.dependencies, {'flat': second_flat_stack})

    # Configure the light run to use the other flat run, which has prefix
    # `flat_run_0`.
    context.settings = Settings(
        stack_settings={
            substack.content_hash:
                StackSettings(dependencies={'flat': first_flat_stack})
        })

    # Run the graph again, with the settings applied.
    run_graph(
        build_compute_graph(regroup=True,
                            make_previews=False,
                            run_main_processing_graph=False))

    stacks = context.stacks
    self.assertEqual(len(stacks), 1)
    self.assertEqual(len(context.calibration_stacks), 2)

    light_stacks = [
        s for s in stacks.values()
        if s.image_type == 'light' and isinstance(s, CompositeStack)
    ]
    self.assertEqual(len(light_stacks), 1)
    stack = next(iter(light_stacks))
    self.assertEqual(len(stack.contents), 1)
    substack = next(iter(stack.contents.values()))
    self.assertEqual(substack.dependencies, {'flat': first_flat_stack})

  def test_generic_grouping(self) -> None:
    copy_testdata_frames(target_name='NGC7635', dest_dir=context.input_dirs[0])
    make_synthetic_frames(number=5,
                          image_type='Bias',
                          fill_value=100,
                          exposure=1,
                          filter=None,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=context.input_dirs[0])
    make_synthetic_frames(number=7,
                          image_type='Dark',
                          fill_value=100,
                          exposure=60,
                          filter=None,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=context.input_dirs[0])
    make_synthetic_frames(number=11,
                          image_type='Flat',
                          fill_value=100,
                          exposure=2,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15,
                          dest_dir=context.input_dirs[0])

    run_graph(
        build_compute_graph(regroup=True,
                            make_previews=False,
                            run_main_processing_graph=False))

    stacks = context.stacks
    self.assertEqual(len(stacks), 1)
    self.assertEqual(len(context.calibration_stacks), 3)

    toplevel_stacks = get_toplevel_stacks(stacks)
    self.assertEqual(len(toplevel_stacks), 1)
    stack = toplevel_stacks[0]
    self.assertEqual(len(stack.contents), 1)


if __name__ == "__main__":
  unittest.main()
