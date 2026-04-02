import pytest
import datetime
from dataclasses import replace

from frozendict import frozendict
from compute_graph import build_compute_graph
from context import context
from decisions.decisions import (
    DecisionType, DecisionOverride, BinaryChoice, ObjectType,
)
from decisions.helpers import exclude, quarantine_key
from decisions.resolve import resolve_decisions
from decisions.compute_graph import ALL_EXCLUSION_TYPES, _is_quarantined
from dependency_graph import run_graph
from testing import make_synthetic_frames, copy_testdata_frames
from stacks.composite_stack import Stack


@pytest.mark.heavy
def test_quarantine_run(test_context) -> None:
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

    # Quarantine the first run.
    overrides = frozendict(exclude(
        (DecisionType.QUARANTINE, ObjectType.STACK, stack.stable_hash, runs[0])
    ))

    # resolve_decisions should filter to only the second run.
    resolved = resolve_decisions(stack, ALL_EXCLUSION_TYPES, overrides, min_frames_in_stack=1)
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
