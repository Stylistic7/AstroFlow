import datetime
import math
import pytest
from frozendict import frozendict

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from testing import make_synthetic_frames
from customization.decisions import (
    DecisionPoint,
    DecisionType,
    BinaryChoice,
    ObjectType,
    DecisionOverride,
    resolve_decision,
)


def test_resolve_decision_default():
    dp = DecisionPoint(
        key=(DecisionType.ROTATION_OUTLIER, ObjectType.STACK, "s1", "r1"),
        decision_type=DecisionType.ROTATION_OUTLIER,
        default_choice=BinaryChoice.INCLUDE,
        title="test",
        description="test",
        object_type=ObjectType.STACK,
        object_id="s1",
        subject_id="r1",
    )
    assert resolve_decision(dp, frozendict()) == BinaryChoice.INCLUDE


def test_resolve_decision_override():
    dp = DecisionPoint(
        key=(DecisionType.ROTATION_OUTLIER, ObjectType.STACK, "s1", "r1"),
        decision_type=DecisionType.ROTATION_OUTLIER,
        default_choice=BinaryChoice.INCLUDE,
        title="test",
        description="test",
        object_type=ObjectType.STACK,
        object_id="s1",
        subject_id="r1",
    )
    overrides = frozendict({
        dp.key: DecisionOverride(choice=BinaryChoice.EXCLUDE),
    })
    assert resolve_decision(dp, overrides) == BinaryChoice.EXCLUDE


@pytest.mark.heavy
def test_rotation_outlier_detection(test_context):
    """Frames with notably different rotation angles should generate a decision point."""
    # First group: 10 frames at rotation 90 degrees (session 1)
    make_synthetic_frames(
        ra=100,
        dec=50,
        prefix="normal",
        number=10,
        dither=0,
        rotation_angle_deg=90.0,
        start_datetime=datetime.datetime(2023, 12, 1, 20, 0, 0),
        dest_dir=context.input_dirs[0],
    )
    # Second group: 5 frames at rotation 97 degrees (session 2, >8h later
    # so they form a separate run within the same stack)
    make_synthetic_frames(
        ra=100,
        dec=50,
        prefix="rotated",
        number=5,
        dither=0,
        rotation_angle_deg=97.0,
        start_datetime=datetime.datetime(2023, 12, 2, 20, 0, 0),
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    decisions = context.get("decisions", {})
    assert len(decisions) > 0, "Expected at least one decision point to be detected"

    # All detected decisions should be rotation outliers
    for key, dp in decisions.items():
        assert dp.decision_type == DecisionType.ROTATION_OUTLIER
        assert dp.default_choice == BinaryChoice.INCLUDE
        assert dp.object_type == ObjectType.STACK
        assert "diff_deg" in dp.metadata


@pytest.mark.heavy
def test_exclude_rotated_run_splits_stack(test_context):
    """Excluding a rotated run via decision override should remove it from the stack."""
    # Create 10 frames at 90° and 5 frames at 97° — same position, different rotation.
    # These form one stack with two runs. The 7° difference should trigger a decision.
    make_synthetic_frames(
        ra=100, dec=50, prefix="normal", number=10, dither=0,
        rotation_angle_deg=90.0,
        start_datetime=datetime.datetime(2023, 12, 1, 20, 0, 0),
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        ra=100, dec=50, prefix="rotated", number=5, dither=0,
        rotation_angle_deg=97.0,
        start_datetime=datetime.datetime(2023, 12, 2, 20, 0, 0),
        dest_dir=context.input_dirs[0],
    )

    # First run: default behavior includes all frames.
    run_graph(build_compute_graph())

    assert len(context.stacks) == 1
    stack = next(iter(context.stacks.values())).get()
    total_frames_before = len(stack.all_frames())
    assert total_frames_before == 15

    # A rotation outlier decision should have been detected.
    decisions = context.get("decisions", {})
    rotation_decisions = {
        k: v for k, v in decisions.items()
        if v.decision_type == DecisionType.ROTATION_OUTLIER
    }
    assert len(rotation_decisions) > 0

    # Find the decision for the rotated run (5 frames) and exclude it.
    rotated_key = next(
        key for key, dp in rotation_decisions.items()
        if dp.metadata["run_frame_count"] == 5
    )
    overrides = frozendict({
        rotated_key: DecisionOverride(choice=BinaryChoice.EXCLUDE),
    })

    # Update settings with the override and re-run.
    from customization.settings import Settings
    new_settings = Settings(
        clustering_settings=context.settings.clustering_settings,
        development_settings=context.settings.development_settings,
        preview_settings=context.settings.preview_settings,
        decision_overrides=overrides,
    )
    context.settings = new_settings

    run_graph(build_compute_graph())

    assert len(context.stacks) == 1
    stack_after = next(iter(context.stacks.values())).get()
    total_frames_after = len(stack_after.all_frames())

    # The excluded run's 5 frames should be removed, leaving 10.
    assert total_frames_after == 10


@pytest.mark.heavy
def test_no_decisions_for_uniform_rotation(test_context):
    """Frames with identical rotation should not generate decisions."""
    make_synthetic_frames(
        ra=100,
        dec=50,
        prefix="uniform",
        number=10,
        dither=0,
        rotation_angle_deg=90.0,
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    decisions = context.get("decisions", {})
    assert len(decisions) == 0, "No decisions expected for uniform rotation"
