import pytest
import datetime
from dataclasses import replace

from frozendict import frozendict
from compute_graph import build_compute_graph
from context import context
from decisions.decisions import (
    DecisionType, DecisionOverride, BinaryChoice, ObjectType,
    decisions_for_stack,
)
from decisions.helpers import exclude, quarantine_key
from decisions.resolve import resolve_decisions
from decisions.compute_graph import ALL_EXCLUSION_TYPES, _is_quarantined
from dependency_graph import run_graph
from testing import make_synthetic_frames, copy_testdata_frames
from stacks.composite_stack import Stack


@pytest.mark.heavy
def test_quarantine_run(test_context) -> None:
    """Quarantining a run via quarantine_key(parent_id=stack_hash) must
    produce a key that survives config_for's per-stack filtering and
    actually excludes the run during resolve_decisions."""
    copy_testdata_frames(
        prefix="light_run_0",
        target_name="NGC7635",
        start_datetime=datetime.datetime(2024, 2, 5, 16, 29, 43),
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        prefix="flat_run_0",
        number=11,
        image_type="Flat",
        start_datetime=datetime.datetime(2024, 2, 5, 18, 29, 43),
        fill_value=50,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )
    copy_testdata_frames(
        prefix="light_run_1",
        target_name="NGC7635",
        start_datetime=datetime.datetime(2024, 2, 7, 16, 29, 43),
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        prefix="flat_run_1",
        number=13,
        image_type="Flat",
        start_datetime=datetime.datetime(2024, 2, 7, 18, 29, 43),
        fill_value=50,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    assert len(context.stacks) == 1
    assert len(context.targets) == 1

    stack = next(iter(context.stacks.values())).get()
    assert len(stack.contents) == 2
    runs = list(stack.contents.keys())

    # Build quarantine key the same way the web endpoint does:
    # parent_id = stack hash, object_id = run hash.
    key = quarantine_key(ObjectType.STACK, runs[0], parent_id=stack.stable_hash)
    assert key[2] == stack.stable_hash, "key[2] must be the stack hash for config_for filtering"
    assert key[3] == runs[0], "key[3] must be the run hash for _apply_run_exclusions"

    overrides = frozendict(exclude(key))

    # Simulate config_for's per-stack filtering via decisions_for_stack.
    filtered = decisions_for_stack(overrides, stack.stable_hash)
    assert len(filtered) == 1, "quarantine key must survive per-stack filtering"

    # resolve_decisions should filter to only the second run.
    resolved = resolve_decisions(stack, ALL_EXCLUSION_TYPES, filtered, min_frames_in_stack=1)
    assert resolved is not None
    assert list(resolved.contents.keys()) == [runs[1]]


@pytest.mark.heavy
def test_quarantine_target(with_orion_mosaic):
    """Quarantining a target via decision override marks it as quarantined."""
    assert len(context.targets) == 4
    assert len(context.mosaics) == 1

    target_hash = next(iter(context.targets))
    overrides = frozendict(exclude(quarantine_key(ObjectType.TARGET, target_hash)))

    assert _is_quarantined(overrides, ObjectType.TARGET, target_hash)

    # Other targets are not affected.
    other_hashes = [h for h in context.targets if h != target_hash]
    for h in other_hashes:
        assert not _is_quarantined(overrides, ObjectType.TARGET, h)


@pytest.mark.heavy
def test_quarantine_stack(with_orion_mosaic):
    """Quarantining a stack via decision override marks it as quarantined."""
    stack_hash = next(iter(context.stacks))
    overrides = frozendict(exclude(quarantine_key(ObjectType.STACK, stack_hash)))

    assert _is_quarantined(overrides, ObjectType.STACK, stack_hash)

    # Other stacks are not affected.
    other_hashes = [h for h in context.stacks if h != stack_hash]
    for h in other_hashes:
        assert not _is_quarantined(overrides, ObjectType.STACK, h)
