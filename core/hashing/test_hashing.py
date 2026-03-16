import ray
import copy

from hashing.hashing import hash_list
from hashing.types import LargeObject
from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from stacks.composite_stack import SimpleStacksDict
from mosaics.orion_mosaic import add_orion_mosaic_images
from metadata.image_metadata import load_metadata


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
    assert stack.stable_hash == "7825680a0bef1d88"
    assert stack.content_hash == "bb8f5281b0ab4846"

    # The Python hash of a Stack is just an integer representation of its
    # content hash, which should be stable across Python sessions.
    assert hash(stack) == 1605787510562869017

    # The Python hash of a SimpleStacksDict goes through `hash_dict_int()`,
    # and should be stable across Python sessions.
    dict = SimpleStacksDict({stack.stable_hash: stack})
    assert hash(dict) == 1605787510562869017


def test_metadata_dict(test_context) -> None:
    add_orion_mosaic_images(panes=[1], random_seed=123)
    run_graph(build_compute_graph())

    metadata = ray.get(load_metadata.remote())
    assert metadata.content_hash == "977b6cfb2a98f2ba"

    add_orion_mosaic_images(
        panes=[2], add_calibration_frames=False, random_seed=456
    )
    run_graph(build_compute_graph())

    metadata = ray.get(load_metadata.remote())
    assert metadata.content_hash != "977b6cfb2a98f2ba"


def test_large_object(test_context) -> None:
    d = {"A": 1, "B": 2}
    d_ref = LargeObject(d)
    assert d_ref.content_hash == "9ba9e7eeec8c514f"

    d_copy = copy.copy(d)
    d_copy_ref = LargeObject(d_copy)
    assert d_ref.content_hash == d_copy_ref.content_hash

    d["C"] = 3
    new_d_ref = LargeObject(d)
    assert new_d_ref.content_hash != d_ref.content_hash
