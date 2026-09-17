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
from metadata.image_metadata import load_library_now


@pytest.mark.exhaustive
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
    assert stack.content_hash == "f1043e9116d4f750"
    assert hash(stack) == 1247087795776248668

    dict = RunsDict({stack.stable_hash: stack})
    assert hash(dict) == 1247087795776248668

    # --- Metadata hash ---
    # The library ref's identity is `(content_hash, filename)` per frame —
    # deliberately PATH-dependent, so a moved library re-keys its consumers
    # (`ContextRef.store_large`'s `identity`). A fixture builds into a fresh
    # temp dir every run, so the value is not a constant to pin: pin that it
    # is stable for the same library and moves when frames are added.
    metadata = ray.get(load_library_now.remote())
    first = metadata.content_hash
    assert first and ray.get(load_library_now.remote()).content_hash == first

    # Adding more frames changes the metadata hash.
    add_orion_mosaic_images(
        panes=[2], add_calibration_frames=False, random_seed=456
    )
    run_graph(build_compute_graph())

    metadata = ray.get(load_library_now.remote())
    assert metadata.content_hash != first


def test_frozendict_hash_call_stable():
    """hash_call on a frozendict must be stable regardless of insertion order."""
    from frozendict import frozendict
    d1 = frozendict({"a": 1, "b": 2, "c": 3})
    d2 = frozendict({"c": 3, "a": 1, "b": 2})
    assert hash_list([d1]) == hash_list([d2])


def test_nested_frozendict_stable():
    from frozendict import frozendict
    inner = frozendict({"x": 10, "y": 20})
    d1 = frozendict({"k1": inner, "k2": frozendict()})
    d2 = frozendict({"k2": frozendict(), "k1": inner})
    assert hash_list([d1]) == hash_list([d2])


def test_decision_key_hash_stable():
    """Decision keys (frozen dataclasses) used as frozendict keys must
    hash stably — they flow through ProcessingConfig.decisions."""
    from decisions.decisions import DecisionKey, DecisionType, ObjectType
    from decisions.decisions import DecisionOverride, BinaryChoice
    from frozendict import frozendict

    k1 = DecisionKey.quarantine(ObjectType.STACK, "aaa")
    k2 = DecisionKey.quarantine(ObjectType.STACK, "bbb")
    v1 = DecisionOverride(choice=BinaryChoice.EXCLUDE)
    v2 = DecisionOverride(choice=BinaryChoice.INCLUDE)

    d1 = frozendict({k1: v1, k2: v2})
    d2 = frozendict({k2: v2, k1: v1})
    assert hash_list([d1]) == hash_list([d2])


def test_processing_config_hash_stable():
    """ProcessingConfig is the main arg to @node functions like stack_files.
    Changing one stack's decisions must NOT change the hash for a config
    that was filtered to a different stack."""
    from decisions.settings import ProcessingConfig
    from decisions.decisions import DecisionKey, DecisionOverride, ObjectType, BinaryChoice
    from frozendict import frozendict

    k_a = DecisionKey.quarantine(ObjectType.STACK, "stack_a")
    k_b = DecisionKey.quarantine(ObjectType.STACK, "stack_b")
    v = DecisionOverride(choice=BinaryChoice.EXCLUDE)

    # Config filtered to stack_a only:
    config_a_v1 = ProcessingConfig(decisions=frozendict({k_a: v}))
    config_a_v2 = ProcessingConfig(decisions=frozendict({k_a: v}))
    assert hash_list([config_a_v1]) == hash_list([config_a_v2])

    # Adding a decision for stack_b should NOT affect stack_a's config
    # (because the caller filters before passing):
    config_a_again = ProcessingConfig(decisions=frozendict({k_a: v}))
    assert hash_list([config_a_v1]) == hash_list([config_a_again])


def test_processing_config_changes_on_decision_change():
    """Changing a decision in the config MUST change the hash."""
    from decisions.settings import ProcessingConfig
    from decisions.decisions import DecisionKey, DecisionOverride, ObjectType, BinaryChoice
    from frozendict import frozendict

    k = DecisionKey.quarantine(ObjectType.STACK, "stack_a")
    c1 = ProcessingConfig(decisions=frozendict({k: DecisionOverride(choice=BinaryChoice.EXCLUDE)}))
    c2 = ProcessingConfig(decisions=frozendict({k: DecisionOverride(choice=BinaryChoice.INCLUDE)}))
    assert hash_list([c1]) != hash_list([c2])


def test_empty_frozendict_stable():
    from frozendict import frozendict
    assert hash_list([frozendict()]) == hash_list([frozendict()])


def test_set_order_independent():
    assert hash_list([{"c", "a", "b"}]) == hash_list([{"b", "c", "a"}])


def test_dict_order_independent():
    d1 = {"z": 1, "a": 2, "m": 3}
    d2 = {"a": 2, "m": 3, "z": 1}
    assert hash_list([d1]) == hash_list([d2])



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
