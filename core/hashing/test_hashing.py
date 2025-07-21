import unittest

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from stacks.composite_stack import SimpleStacksDict
from mosaics.orion_mosaic import add_orion_mosaic_images


def test_hashes(test_context) -> None:
  # This test uses a random seed because it checks the specific values of
  # hashes, so all of the frames will have predictable contents.
  add_orion_mosaic_images(panes=[1], random_seed=123)
  run_graph(build_compute_graph())

  assert len(context.stacks) == 1
  assert len(context.calibration_stacks) == 3
  assert len(context.targets) == 1
  assert len(context.mosaics) == 0

  stack = next(iter(context.stacks.values()))

  # The stable hash and content hashes are computed from the image data.
  # The image data should be predictable, because `random_seed` was provided
  # above.
  assert stack.stable_hash == '7825680a0bef1d88'
  assert stack.content_hash == 'f1043e9116d4f750'

  # The Python hash of a Stack is just an integer representation of its
  # content hash, which should be stable across Python sessions.
  assert hash(stack) == 1247087795776248668

  # The Python hash of a SimpleStacksDict goes through `hash_dict_int()`,
  # and should be stable across Python sessions.
  dict = SimpleStacksDict({stack.stable_hash: stack})
  assert hash(dict) == 1247087795776248668
