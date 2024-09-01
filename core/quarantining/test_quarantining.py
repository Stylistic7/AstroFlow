from ray_infra import ray_local

import ray
import unittest

from context import context
from testing.test_context import make_test_context
from stacks import SimpleStack, CompositeStack
from quarantining import quarantine_stacks
from mosaics.orion_mosaic import add_orion_mosaic_images
from dependency_graph import Graph, make_graph_from_list, run_graph
from metadata.image_metadata import scatter_compute_image_metadata
from compute_graph import maybe_build_regroup_graph


def compute_graph_fragment():
  return make_graph_from_list([
      (scatter_compute_image_metadata, 'input'),
      maybe_build_regroup_graph,
  ])


class TestQuarantining(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()
    context.regroup = True

  def tearDown(self):
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_simple_stack_quarantining(self):
    add_orion_mosaic_images()

    run_graph(compute_graph_fragment())

    self.assertEqual(len(context.targets), 4)
    self.assertEqual(len(context.mosaics), 1)

    context.quarantine_targets = [next(iter(context.targets))]

    run_graph(compute_graph_fragment())

    self.assertEqual(len(context.targets), 3)
    self.assertEqual(len(context.mosaics), 1)


if __name__ == '__main__':
  unittest.main()
