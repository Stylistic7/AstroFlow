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
    get_decisions_for_object,
)
from customization.resolve import resolve_decisions
from customization.compute_graph import ALL_EXCLUSION_TYPES


def _make_decision(key):
    return DecisionPoint(
        key=key, decision_type=key[0], default_choice=BinaryChoice.INCLUDE,
        title="test", description="test",
        object_type=key[1], object_id=key[2], subject_id=key[3],
    )


def test_resolve_decision_default(test_context):
    key = (DecisionType.ROTATION_OUTLIER, ObjectType.STACK, "s1", "r1")
    dp = _make_decision(key)
    # No override → default value is computed, cached, and returned.
    assert resolve_decision(dp, frozendict(), default=BinaryChoice.INCLUDE) == BinaryChoice.INCLUDE
    # Second call returns cached default, not the new default argument.
    assert resolve_decision(dp, frozendict(), default=BinaryChoice.EXCLUDE) == BinaryChoice.INCLUDE


def test_resolve_decision_callable_default(test_context):
    key = (DecisionType.ROTATION_OUTLIER, ObjectType.STACK, "s2", "r2")
    dp = _make_decision(key)
    call_count = [0]
    def expensive_default():
        call_count[0] += 1
        return BinaryChoice.INCLUDE
    # First call: lambda runs, result cached.
    assert resolve_decision(dp, frozendict(), default=expensive_default) == BinaryChoice.INCLUDE
    assert call_count[0] == 1
    # Second call: cached, lambda not called again.
    assert resolve_decision(dp, frozendict(), default=expensive_default) == BinaryChoice.INCLUDE
    assert call_count[0] == 1


def test_resolve_decision_override(test_context):
    key = (DecisionType.ROTATION_OUTLIER, ObjectType.STACK, "s1", "r1")
    dp = _make_decision(key)
    overrides = frozendict({
        key: DecisionOverride(choice=BinaryChoice.EXCLUDE),
    })
    # User override wins regardless of default.
    assert resolve_decision(dp, overrides, default=BinaryChoice.INCLUDE) == BinaryChoice.EXCLUDE


def test_resolve_decision_override_caches_default_for_ui(test_context):
    """When there's an override, the default is still computed and cached for the UI."""
    key = (DecisionType.ROTATION_OUTLIER, ObjectType.STACK, "s3", "r3")
    dp = _make_decision(key)
    overrides = frozendict({
        key: DecisionOverride(choice=BinaryChoice.EXCLUDE),
    })
    # Override wins.
    assert resolve_decision(dp, overrides, default=BinaryChoice.INCLUDE) == BinaryChoice.EXCLUDE
    # But the cached DecisionPoint has the computed default for UI display.
    cached = get_decisions_for_object(ObjectType.STACK, "s3")
    assert key in cached
    assert cached[key].default_choice == BinaryChoice.INCLUDE


@pytest.mark.heavy
def test_rotation_outlier_detection_and_exclusion(test_context):
    """Rotation outlier: detection, resolve_decisions filtering, and cache key invalidation."""
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

    run_graph(build_compute_graph())

    assert len(context.stacks) == 1
    stack = next(iter(context.stacks.values())).get()
    stack_hash = next(iter(context.stacks))

    # Enriched stack preserves all frames.
    assert len(stack.all_frames()) == 15

    # --- Detection ---
    all_decisions = get_decisions_for_object(ObjectType.STACK, stack_hash)
    rotation_decisions = {
        k: dp for k, dp in all_decisions.items()
        if dp.decision_type == DecisionType.ROTATION_OUTLIER
    }
    assert len(rotation_decisions) > 0, "Expected at least one rotation outlier decision"
    for key, dp in rotation_decisions.items():
        assert dp.default_choice == BinaryChoice.INCLUDE
        assert dp.object_type == ObjectType.STACK
        assert "diff_deg" in dp.metadata

    # Find the rotated run's decision (5 frames).
    rotated_key = next(
        k for k, dp in rotation_decisions.items()
        if dp.metadata["run_frame_count"] == 5
    )

    # --- Filtering ---
    # Without override: all 15 frames survive.
    resolved = resolve_decisions(stack, ALL_EXCLUSION_TYPES, frozendict(), min_frames_in_stack=1)
    assert len(resolved.all_frames()) == 15

    # With EXCLUDE override: only the 10 normal frames survive.
    overrides = frozendict({rotated_key: DecisionOverride(choice=BinaryChoice.EXCLUDE)})
    resolved = resolve_decisions(stack, ALL_EXCLUSION_TYPES, overrides, min_frames_in_stack=1)
    assert len(resolved.all_frames()) == 10

    # --- Cache key invalidation ---
    from hashing import hash_call, hash_source_code
    from stacks.stacking import stack_files

    fn_hash = hash_source_code(stack_files)
    cal_stacks = context.calibration_stacks
    dev_settings = context.settings.development_settings
    min_frames = context.settings.clustering_settings.min_frames_in_stack

    key_no_override = hash_call(fn_hash, stack, cal_stacks, dev_settings, min_frames, frozendict())
    key_with_override = hash_call(fn_hash, stack, cal_stacks, dev_settings, min_frames, overrides)
    assert key_no_override != key_with_override, (
        "Changing decisions must produce a different cache key"
    )


@pytest.mark.heavy
def test_subframe_quality_decision_and_filtering(test_context):
    """Subframe quality: detection, metadata, resolve_decisions filtering, and override."""
    make_synthetic_frames(
        ra=100, dec=50, prefix="good", number=10, dither=0,
        rotation_angle_deg=90.0, num_stars=1000,
        start_datetime=datetime.datetime(2023, 12, 1, 20, 0, 0),
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        ra=100, dec=50, prefix="bad", number=5, dither=0,
        rotation_angle_deg=90.0, num_stars=100,
        start_datetime=datetime.datetime(2023, 12, 1, 21, 0, 0),
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    assert len(context.stacks) == 1
    stack = next(iter(context.stacks.values())).get()
    stack_hash = next(iter(context.stacks))

    # Enriched stack preserves all frames.
    assert len(stack.all_frames()) == 15

    # --- Detection ---
    all_decisions = get_decisions_for_object(ObjectType.STACK, stack_hash)
    quality_decisions = {
        k: v for k, v in all_decisions.items()
        if v.decision_type == DecisionType.SUBFRAME_QUALITY
    }
    assert len(quality_decisions) == 1
    dp = next(iter(quality_decisions.values()))
    assert dp.metadata["excluded_count"] == 5
    assert dp.metadata["total_frames"] == 15
    # The cached default is the quality threshold (a float).
    assert dp.default_choice == 0.75

    # --- Filtering with default threshold ---
    # Default 75% threshold: 100 stars < 750 (75% of 1000), so bad frames excluded.
    resolved = resolve_decisions(stack, ALL_EXCLUSION_TYPES, frozendict(), min_frames_in_stack=1)
    assert len(resolved.all_frames()) == 10

    # --- Override with threshold=0.0 (keep all frames) ---
    quality_key = next(iter(quality_decisions))
    override_keep_all = frozendict({quality_key: DecisionOverride(choice=0.0)})
    resolved = resolve_decisions(stack, ALL_EXCLUSION_TYPES, override_keep_all, min_frames_in_stack=1)
    assert len(resolved.all_frames()) == 15

    # --- Override with stricter threshold (0.9 = 900 stars minimum) ---
    override_strict = frozendict({quality_key: DecisionOverride(choice=0.9)})
    resolved = resolve_decisions(stack, ALL_EXCLUSION_TYPES, override_strict, min_frames_in_stack=1)
    # Only frames with >= 900 stars survive (the 10 "good" frames have ~1000 stars each,
    # but with noise they might vary — just verify fewer than 15 survive).
    assert len(resolved.all_frames()) <= 10


@pytest.mark.heavy
def test_no_subframe_decision_for_minor_exclusion(test_context):
    """When only a small fraction of frames are excluded, no decision is generated."""
    make_synthetic_frames(
        ra=100, dec=50, prefix="good", number=14, dither=0,
        rotation_angle_deg=90.0, num_stars=1000,
        start_datetime=datetime.datetime(2023, 12, 1, 20, 0, 0),
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        ra=100, dec=50, prefix="bad", number=1, dither=0,
        rotation_angle_deg=90.0, num_stars=100,
        start_datetime=datetime.datetime(2023, 12, 1, 21, 0, 0),
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    assert len(context.stacks) == 1
    stack = next(iter(context.stacks.values())).get()
    # Enriched stack preserves all frames.
    assert len(stack.all_frames()) == 15

    # No subframe quality decision should be generated (1/15 = 6.7% < 25%).
    stack_hash = next(iter(context.stacks))
    decisions = get_decisions_for_object(ObjectType.STACK, stack_hash)
    quality_decisions = {
        k: v for k, v in decisions.items()
        if v.decision_type == DecisionType.SUBFRAME_QUALITY
    }
    assert len(quality_decisions) == 0


@pytest.mark.heavy
def test_no_decisions_for_uniform_rotation(test_context):
    """Frames with identical rotation should not generate decisions."""
    make_synthetic_frames(
        ra=100, dec=50, prefix="uniform", number=10, dither=0,
        rotation_angle_deg=90.0,
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    stack_hash = next(iter(context.stacks))
    decisions = get_decisions_for_object(ObjectType.STACK, stack_hash)
    rotation_decisions = {
        k: v for k, v in decisions.items()
        if v.decision_type == DecisionType.ROTATION_OUTLIER
    }
    assert len(rotation_decisions) == 0, "No rotation decisions expected for uniform rotation"
