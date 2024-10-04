from ray_infra import ray_local

import ray
import unittest

from context import context
from testing.test_context import make_test_context
from stacks.simple_stack import SimpleStack
from stacks.composite_stack import CompositeStack
# from quarantining import quarantine_stacks
from mosaics.orion_mosaic import add_orion_mosaic_images
from dependency_graph import Graph, make_graph_from_list, run_graph
from metadata.image_metadata import scatter_compute_image_metadata
from compute_graph import build_compute_graph


class TestQuarantining(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()
    context.regroup = True

  def tearDown(self):
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_quarantine_toplevel_target(self):
    add_orion_mosaic_images()
    run_graph(
        build_compute_graph(regroup=True,
                            make_previews=False,
                            run_main_processing_graph=False))

    self.assertEqual(len(context.targets), 4)
    self.assertEqual(len(context.mosaics), 1)

    stacks = [s for s in context.stacks if type(s) is CompositeStack]
    self.assertTrue(all(len(s.dependencies) == 3 for s in stacks))

    context.quarantined_targets = [next(iter(context.targets))]

    run_graph(
        build_compute_graph(regroup=True,
                            make_previews=False,
                            run_main_processing_graph=False))

    self.assertEqual(len(context.targets), 3)
    self.assertEqual(len(context.mosaics), 1)

    stacks = [s for s in context.stacks if type(s) is CompositeStack]
    self.assertTrue(all(len(s.dependencies) == 3 for s in stacks))

  def test_quarantine_calibration_stack(self):
    add_orion_mosaic_images()
    run_graph(
        build_compute_graph(regroup=True,
                            make_previews=False,
                            run_main_processing_graph=False))

    self.assertEqual(len(context.targets), 4)
    self.assertEqual(len(context.mosaics), 1)

    stacks = [s for s in context.stacks if type(s) is CompositeStack]
    self.assertTrue(all(len(s.dependencies) == 3 for s in stacks))

    context.quarantined_stacks = [
        next(
            iter(s for s in context.stacks
                 if context.stacks[s].image_type == 'dark'))
    ]
    run_graph(
        build_compute_graph(regroup=True,
                            make_previews=False,
                            run_main_processing_graph=False))

    self.assertEqual(len(context.targets), 4)
    self.assertEqual(len(context.mosaics), 1)

    stacks = [s for s in context.stacks if type(s) is CompositeStack]
    self.assertTrue(all(len(s.dependencies) == 2 for s in stacks))


if __name__ == '__main__':
  unittest.main()
