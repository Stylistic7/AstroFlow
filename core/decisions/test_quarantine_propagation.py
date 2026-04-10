"""Tests for quarantine helpers and propagation logic."""
import datetime
import pytest
from frozendict import frozendict

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from testing import make_synthetic_frames
from decisions.decisions import DecisionKey, ObjectType
from decisions.helpers import exclude
from decisions.compute_graph import _is_quarantined


def test_is_quarantined(test_context):
    """_is_quarantined returns True only when an explicit EXCLUDE override exists."""
    overrides = frozendict(exclude(DecisionKey.quarantine(ObjectType.TARGET, "abc")))
    assert _is_quarantined(overrides, ObjectType.TARGET, "abc")
    assert not _is_quarantined(overrides, ObjectType.TARGET, "xyz")
    assert not _is_quarantined(frozendict(), ObjectType.TARGET, "abc")


def test_all_stacks_quarantined_implies_target_quarantined(test_context):
    """When every stack in a target is quarantined, the target is effectively quarantined."""
    make_synthetic_frames(
        ra=100, dec=50, prefix="a", number=5, dither=0,
        start_datetime=datetime.datetime(2023, 12, 1, 20, 0, 0),
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    assert len(context.targets) == 1
    target_hash = next(iter(context.targets))
    target = context.targets[target_hash].get()

    # Quarantine every stack.
    overrides = {}
    for stack_id in target.stack_ids:
        overrides |= exclude(DecisionKey.quarantine(ObjectType.STACK, stack_id))
    overrides = frozendict(overrides)

    # Target itself is not explicitly quarantined...
    assert not _is_quarantined(overrides, ObjectType.TARGET, target_hash)

    # ...but all its stacks are.
    assert all(
        _is_quarantined(overrides, ObjectType.STACK, sid)
        for sid in target.stack_ids
    )


def test_partial_quarantine_does_not_affect_target(test_context):
    """When only some stacks are quarantined, the target is not effectively quarantined."""
    make_synthetic_frames(
        ra=100, dec=50, prefix="a", number=5, dither=0, filter="H",
        start_datetime=datetime.datetime(2023, 12, 1, 20, 0, 0),
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        ra=100, dec=50, prefix="b", number=5, dither=0, filter="O",
        start_datetime=datetime.datetime(2023, 12, 1, 21, 0, 0),
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    assert len(context.targets) == 1
    target_hash = next(iter(context.targets))
    target = context.targets[target_hash].get()
    assert len(target.stack_ids) == 2

    # Quarantine only one stack.
    first_stack_id = target.stack_ids[0]
    overrides = frozendict(exclude(DecisionKey.quarantine(ObjectType.STACK, first_stack_id)))

    assert _is_quarantined(overrides, ObjectType.STACK, first_stack_id)
    assert not _is_quarantined(overrides, ObjectType.STACK, target.stack_ids[1])
