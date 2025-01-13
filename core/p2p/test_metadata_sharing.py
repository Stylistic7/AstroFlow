import unittest
import time
import hashlib
import json
import tempfile
import shutil
import uuid
from pathlib import Path

from context import context
from targets import Target
from stacks.composite_stack import CompositeStack
from p2p.test_ipfs_node import start_test_ipfs_node, stop_test_ipfs_node
from p2p.ipfs_api import get_node_id, add_file, provide_file, cat_file, publish_name, resolve_name, find_providers, get_file, check_download_status
from p2p.metadata import stacks_to_hdf5, hdf5_to_stacks
from mosaics.orion_mosaic import add_orion_mosaic_images
import compute_graph
from dependency_graph import run_graph
from testing.test_context import make_test_context


class TestMetadataSharing(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def tearDown(self):
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_stack_metadata_round_trip(self):
    add_orion_mosaic_images()
    graph = compute_graph.build_compute_graph(regroup=True,
                                              make_previews=True,
                                              run_main_processing_graph=True)
    run_graph(graph)

    metadata_file = tempfile.NamedTemporaryFile(delete=False)
    composite_stacks = {
        k: v for k, v in context.stacks.items() if type(v) is CompositeStack
    }
    stacks_to_hdf5(composite_stacks, metadata_file)
    metadata_file.close()

    stacks = hdf5_to_stacks(open(metadata_file.name, 'rb').read())

    for k in composite_stacks.keys():
      self.assertEqual(context.stacks[k], stacks[k])


if __name__ == "__main__":
  unittest.main()
