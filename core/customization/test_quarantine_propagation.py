"""Tests for quarantine propagation: stack quarantine → target visibility."""
import datetime
import ray
import pytest
from dataclasses import replace
from frozendict import frozendict

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from testing import make_synthetic_frames
from customization.decisions import (
    DecisionType,
    ObjectType,
    DecisionOverride,
    BinaryChoice,
)
from customization.compute_graph import scatter_customize_targets


def _customize_targets(overrides, targets):
    return ray.get(scatter_customize_targets.remote(overrides, targets))


@pytest.mark.heavy
def test_quarantine_target_directly(test_context):
    """Explicitly quarantining a target removes it from the customized output."""
    make_synthetic_frames(
        ra=100, dec=50, prefix="a", number=5, dither=0,
        start_datetime=datetime.datetime(2023, 12, 1, 20, 0, 0),
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    assert len(context.targets) == 1
    target_hash = next(iter(context.targets))

    # Without quarantine, target survives.
    result = _customize_targets(frozendict(), context.targets)
    assert target_hash in result

    # With quarantine, target is removed.
    key = (DecisionType.QUARANTINE, ObjectType.TARGET, target_hash, target_hash)
    overrides = frozendict({key: DecisionOverride(choice=BinaryChoice.EXCLUDE)})
    result = _customize_targets(overrides, context.targets)
    assert target_hash not in result


@pytest.mark.heavy
def test_quarantine_all_stacks_hides_target(test_context):
    """When every stack in a target is quarantined, the target is excluded."""
    make_synthetic_frames(
        ra=100, dec=50, prefix="a", number=5, dither=0,
        start_datetime=datetime.datetime(2023, 12, 1, 20, 0, 0),
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    assert len(context.targets) == 1
    target_hash = next(iter(context.targets))
    target = context.targets[target_hash].get()

    # Quarantine every stack in the target.
    overrides = {}
    for stack_id in target.stack_ids:
        key = (DecisionType.QUARANTINE, ObjectType.STACK, stack_id, stack_id)
        overrides[key] = DecisionOverride(choice=BinaryChoice.EXCLUDE)

    result = _customize_targets(frozendict(overrides), context.targets)
    assert target_hash not in result, (
        "Target with all stacks quarantined should be excluded"
    )


@pytest.mark.heavy
def test_quarantine_some_stacks_keeps_target(test_context):
    """When only some stacks in a target are quarantined, the target survives."""
    # Create two runs with different filters so they become separate stacks
    # in the same target.
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
    assert len(target.stack_ids) == 2, "Expected two stacks (H and O filters)"

    # Quarantine only the first stack.
    first_stack_id = target.stack_ids[0]
    key = (DecisionType.QUARANTINE, ObjectType.STACK, first_stack_id, first_stack_id)
    overrides = frozendict({key: DecisionOverride(choice=BinaryChoice.EXCLUDE)})

    result = _customize_targets(overrides, context.targets)
    assert target_hash in result, (
        "Target with some stacks still active should not be excluded"
    )
