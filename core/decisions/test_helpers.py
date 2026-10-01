"""Unit tests for decision framework helper functions."""
import pytest
from frozendict import frozendict

from decisions.decisions import (
    DecisionKey, DecisionType, ObjectType, DecisionOverride, BinaryChoice,
)
from decisions.helpers import (
    exclude, include, calibration_override,
)


def test_quarantine_key_target():
    key = DecisionKey.quarantine(ObjectType.TARGET, "abc")
    assert key == DecisionKey(DecisionType.QUARANTINE, ObjectType.TARGET, "abc", "abc")


def test_quarantine_key_stack():
    key = DecisionKey.quarantine(ObjectType.STACK, "s1")
    assert key == DecisionKey(DecisionType.QUARANTINE, ObjectType.STACK, "s1", "s1")


def test_quarantine_key_run_with_parent():
    # run-level keys are RUN-typed, so the run that carries its stack's id
    # never collides with the stack-level key
    key = DecisionKey.quarantine(ObjectType.STACK, "run1", parent_stack_id="stack1")
    assert key == DecisionKey(DecisionType.QUARANTINE, ObjectType.RUN, "stack1", "run1")
    assert key.matches_stack("stack1")
    assert not key.matches_stack("other")


def test_calibration_key_flat():
    key = DecisionKey.calibration("flat", "stack1", "run1")
    assert key == DecisionKey(DecisionType.CALIBRATION_FLAT, ObjectType.STACK, "stack1", "run1")


def test_calibration_key_dark():
    key = DecisionKey.calibration("dark", "s", "r")
    assert key == DecisionKey(DecisionType.CALIBRATION_DARK, ObjectType.STACK, "s", "r")


def test_calibration_key_bias():
    key = DecisionKey.calibration("bias", "s", "r")
    assert key == DecisionKey(DecisionType.CALIBRATION_BIAS, ObjectType.STACK, "s", "r")


def test_calibration_key_invalid():
    with pytest.raises(KeyError):
        DecisionKey.calibration("invalid", "s", "r")


def test_exclude():
    key = DecisionKey.quarantine(ObjectType.STACK, "s1")
    result = exclude(key)
    assert result == {key: DecisionOverride(choice=BinaryChoice.EXCLUDE)}


def test_include():
    key = DecisionKey.quarantine(ObjectType.STACK, "s1")
    result = include(key)
    assert result == {key: DecisionOverride(choice=BinaryChoice.INCLUDE)}


def test_exclude_composable_with_pipe():
    k1 = DecisionKey.quarantine(ObjectType.STACK, "s1")
    k2 = DecisionKey.quarantine(ObjectType.STACK, "s2")
    combined = exclude(k1) | exclude(k2)
    assert len(combined) == 2
    assert combined[k1].choice == BinaryChoice.EXCLUDE
    assert combined[k2].choice == BinaryChoice.EXCLUDE


def test_calibration_override():
    result = calibration_override("flat", "stack1", "run1", "flat_stack_42")
    key = DecisionKey.calibration("flat", "stack1", "run1")
    assert result == {key: DecisionOverride(choice="flat_stack_42")}


def test_calibration_override_composable():
    o1 = calibration_override("flat", "s1", "r1", "f1")
    o2 = calibration_override("dark", "s1", "r1", "d1")
    combined = o1 | o2
    assert len(combined) == 2


def test_overrides_work_with_frozendict():
    overrides = frozendict(
        exclude(DecisionKey.quarantine(ObjectType.STACK, "s1"))
        | calibration_override("flat", "s1", "r1", "f1")
    )
    assert len(overrides) == 2
