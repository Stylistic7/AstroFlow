from ray_infra import ray_local

import numpy as np
import os
import unittest
import ray
from typing import TypeVar, Any
import astropy.io.fits as pyfits  # type: ignore
from pathlib import Path
import structlog
from dataclasses import replace
import datetime

from hashing import hash_list
from testing.test_context import make_test_context
from context import context
from dependency_graph import run_graph
from stacks.composite_stack import CompositeStack, SimpleStacksDict
from compute_graph import build_compute_graph
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
    run_graph(
        build_compute_graph(regroup=True,
                            make_previews=False,
                            run_main_processing_graph=False))

    self.assertEqual(len(context.stacks), 4)
    self.assertEqual(len(context.calibration_stacks), 3)
    self.assertEqual(len(context.targets), 1)
    self.assertEqual(len(context.mosaics), 0)

    light_stack = next(
        iter(s for s in context.stacks.values()
             if isinstance(s, CompositeStack)))

    # The stable hash and content hashes are computed from the image data.
    # The image data should be predictable, because `random_seed` was provided
    # above.
    self.assertEqual(light_stack.stable_hash, 'e1f7d909ac125ebe')
    self.assertEqual(light_stack.content_hash, 'f1043e9116d4f750')

    # The Python hash of a Stack goes through `hash_dataclass()`, and should
    # be stable across Python sessions.
    self.assertEqual(hash(light_stack), 1890771923370192638)

    # The Python hash of a SimpleStacksDict goes through `hash_dict_int()`,
    # and should be stable across Python sessions.
    self.assertTrue(isinstance(context.calibration_stacks, SimpleStacksDict))
    self.assertEqual(hash(context.calibration_stacks), 760237515229308854)
