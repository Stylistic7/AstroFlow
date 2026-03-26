import pytest
import ray
import copy

from hashing.hashing import hash_list
from cache.context_ref import ContextRef
from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from stacks.composite_stack import RunsDict
from mosaics.orion_mosaic import add_orion_mosaic_images
from metadata.image_metadata import load_metadata


@pytest.mark.heavy
def test_hashes_and_metadata(test_context) -> None:
    """Hash stability for stacks and metadata after identification.

    Uses a fixed random seed so frame contents and hashes are deterministic.
    """
    add_orion_mosaic_images(panes=[1], random_seed=123)
    run_graph(build_compute_graph())

    assert len(context.stacks) == 1
    assert len(context.calibration_stacks) == 3
    assert len(context.targets) == 1
    assert len(context.mosaics) == 0

    # --- Stack hashes ---
    stack = next(iter(context.stacks.values())).get()
    assert stack.stable_hash == "7825680a0bef1d88"
    assert stack.content_hash == "3dd215b9d80b5616"
    assert hash(stack) == 1343723860084752222

    dict = RunsDict({stack.stable_hash: stack})
    assert hash(dict) == 1343723860084752222

    # --- Metadata hash ---
    metadata = ray.get(load_metadata.remote())
    assert metadata.content_hash == "977b6cfb2a98f2ba"

    # Adding more frames changes the metadata hash.
    add_orion_mosaic_images(
        panes=[2], add_calibration_frames=False, random_seed=456
    )
    run_graph(build_compute_graph())

    metadata = ray.get(load_metadata.remote())
    assert metadata.content_hash != "977b6cfb2a98f2ba"


def test_context_ref_large(test_context) -> None:
    d = {"A": 1, "B": 2}
    d_ref = ContextRef.store_large(d)
    assert d_ref.content_hash == "9ba9e7eeec8c514f"

    d_copy = copy.copy(d)
    d_copy_ref = ContextRef.store_large(d_copy)
    assert d_ref.content_hash == d_copy_ref.content_hash

    d["C"] = 3
    new_d_ref = ContextRef.store_large(d)
    assert new_d_ref.content_hash != d_ref.content_hash
