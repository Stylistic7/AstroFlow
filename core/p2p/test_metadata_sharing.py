import pytest

pytestmark = pytest.mark.testdata  # reads core/testdata

import unittest
import time
import hashlib
import json
import tempfile
import shutil
import uuid
from pathlib import Path

from context import context
from stacks.composite_stack import Stack
from p2p.metadata import stacks_to_hdf5, hdf5_to_stacks
from mosaics.orion_mosaic import add_orion_mosaic_images
import compute_graph
from dependency_graph import run_graph
from testing.test_context import make_test_context


def test_stack_metadata_round_trip(test_context):
    add_orion_mosaic_images()
    graph = compute_graph.build_compute_graph()
    run_graph(graph)

    metadata_file = tempfile.NamedTemporaryFile(delete=False)
    composite_stacks = {
        k: ref.get() for k, ref in context.stacks.items() if type(ref.get()) is Stack
    }
    stacks_to_hdf5(composite_stacks, metadata_file)
    metadata_file.close()

    stacks = hdf5_to_stacks(open(metadata_file.name, "rb").read())

    for k in composite_stacks.keys():
        assert context.stacks[k].get() == stacks[k]


def test_two_participant_mosaic(test_context):
    # Create an HDF5 file containing just panes 1 and 2.
    add_orion_mosaic_images(panes=[1, 2])
    graph = compute_graph.build_compute_graph()
    run_graph(graph)
    metadata_file_A = tempfile.NamedTemporaryFile(delete=False)
    composite_stacks = {
        k: ref.get() for k, ref in context.stacks.items() if type(ref.get()) is Stack
    }
    stacks_to_hdf5(composite_stacks, metadata_file_A)
    metadata_file_A.close()

    # Create an HDF5 file containing just panes 3 and 4.
    input_dirs = context.input_dirs
    output_dir = context.output_dir
    settings = context.settings
    context.clear()
    context.input_dirs = input_dirs
    context.output_dir = output_dir
    context.settings = settings
    context.stacks = {}
    context.targets = {}
    context.mosaics = {}
    add_orion_mosaic_images(panes=[3, 4])
    graph = compute_graph.build_compute_graph()
    run_graph(graph)
    metadata_file_B = tempfile.NamedTemporaryFile(delete=False)
    composite_stacks = {
        k: ref.get() for k, ref in context.stacks.items() if type(ref.get()) is Stack
    }
    stacks_to_hdf5(composite_stacks, metadata_file_B)
    metadata_file_B.close()

    # Load both HDF5 files at once.
    print("### Loading both HDF5 files...")
    input_dirs2 = context.input_dirs
    output_dir2 = context.output_dir
    settings2 = context.settings
    context.clear()
    context.input_dirs = input_dirs2
    context.output_dir = output_dir2
    context.settings = settings2
    context.stacks = {}
    context.targets = {}
    context.mosaics = {}
    hdf5_to_stacks(open(metadata_file_A.name, "rb").read())
    hdf5_to_stacks(open(metadata_file_B.name, "rb").read())
    graph = compute_graph.build_compute_graph()
    run_graph(graph)
    assert len(context.mosaics) == 1

    mosaic = next(iter(context.mosaics))
    print("### mosaic", mosaic)


if __name__ == "__main__":
    unittest.main()
