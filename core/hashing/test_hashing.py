from ray_infra import ray_local

import unittest

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from testing import make_test_context

from stacks.composite_stack import SimpleStacksDict
from mosaics.orion_mosaic import add_orion_mosaic_images


class TestHashing(unittest.TestCase):

  def setUp(self) -> None:
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def tearDown(self) -> None:
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_hashes(self) -> None:
    # This test uses a random seed because it checks the specific values of
    # hashes, so all of the frames will have predictable contents.
    add_orion_mosaic_images(panes=[1], random_seed=123)
    run_graph(build_compute_graph())

    self.assertEqual(len(context.stacks), 1)
    self.assertEqual(len(context.calibration_stacks), 3)
    self.assertEqual(len(context.targets), 1)
    self.assertEqual(len(context.mosaics), 0)

    stack = next(iter(context.stacks.values()))

    # The stable hash and content hashes are computed from the image data.
    # The image data should be predictable, because `random_seed` was provided
    # above.
    self.assertEqual(stack.stable_hash, '7825680a0bef1d88')
    self.assertEqual(stack.content_hash, 'f1043e9116d4f750')

    # The Python hash of a Stack is just an integer representation of its
    # content hash, which should be stable across Python sessions.
    self.assertEqual(hash(stack), 1247087795776248668)

    # The Python hash of a SimpleStacksDict goes through `hash_dict_int()`,
    # and should be stable across Python sessions.
    dict = SimpleStacksDict({stack.stable_hash: stack})
    self.assertEqual(hash(dict), 1247087795776248668)
