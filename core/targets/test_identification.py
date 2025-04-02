from ray_infra import ray_local

import unittest

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from hashing import hash_list
from testing import make_test_context, copy_testdata_frames

from stacks.composite_stack import CompositeStack


class TestTargetIdentification(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()
    context.regroup = True

  def tearDown(self):
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_orion_nebula(self):
    copy_testdata_frames(target_name='M42', dest_dir=context.input_dirs[0])

    graph = build_compute_graph()
    run_graph(graph)

    self.assertEqual(len(context.stacks), 1)
    stack = next(
        iter([
            s for s in context.stacks.values() if isinstance(s, CompositeStack)
        ]))
    self.assertNotEqual(stack.stable_hash, None)
    self.assertNotEqual(stack.content_hash, None)

    self.assertEqual(len(context.targets), 1)
    target = next(iter(context.targets.values()))
    self.assertEqual(target.stable_hash, stack.stable_hash)


if __name__ == '__main__':
  unittest.main()
