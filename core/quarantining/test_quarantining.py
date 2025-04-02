from ray_infra import ray_local

import datetime
import unittest

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from testing import make_test_context, make_synthetic_frames, copy_testdata_frames

from stacks.composite_stack import CompositeStack
from mosaics.orion_mosaic import add_orion_mosaic_images


class TestQuarantining(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def tearDown(self):
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_quarantine_run(self) -> None:
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

    run_graph(build_compute_graph())

    self.assertEqual(len(context.stacks), 1)
    self.assertEqual(len(context.targets), 1)
    self.assertEqual(len(context.mosaics), 0)

  def test_quarantine_toplevel_target(self):
    add_orion_mosaic_images()
    run_graph(build_compute_graph())

    self.assertEqual(len(context.targets), 4)
    self.assertEqual(len(context.mosaics), 1)

    stacks = [s for s in context.stacks if type(s) is CompositeStack]
    self.assertTrue(all(len(s.dependencies) == 3 for s in stacks))

    context.quarantined_targets = [next(iter(context.targets))]

    run_graph(build_compute_graph())

    self.assertEqual(len(context.targets), 3)
    self.assertEqual(len(context.mosaics), 1)

    stacks = [s for s in context.stacks if type(s) is CompositeStack]
    self.assertTrue(all(len(s.dependencies) == 3 for s in stacks))

  def test_quarantine_calibration_stack(self):
    add_orion_mosaic_images()
    run_graph(build_compute_graph())

    self.assertEqual(len(context.targets), 4)
    self.assertEqual(len(context.mosaics), 1)

    stacks = [s for s in context.stacks if type(s) is CompositeStack]
    self.assertTrue(all(len(s.dependencies) == 3 for s in stacks))

    context.quarantined_calibration_stacks = [
        next(
            iter(s for s in context.calibration_stacks
                 if context.calibration_stacks[s].image_type == 'dark'))
    ]
    run_graph(build_compute_graph())

    self.assertEqual(len(context.targets), 4)
    self.assertEqual(len(context.mosaics), 1)

    stacks = [s for s in context.stacks if type(s) is CompositeStack]
    self.assertTrue(all(len(s.dependencies) == 2 for s in stacks))


if __name__ == '__main__':
  unittest.main()
