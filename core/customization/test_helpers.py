"""Unit tests for decision framework helper functions."""
import pytest
from frozendict import frozendict

from customization.decisions import (
    DecisionType, ObjectType, DecisionOverride, BinaryChoice,
)
from customization.helpers import (
    quarantine_key, calibration_key,
    exclude, include, calibration_override,
)


def test_quarantine_key_target():
    key = quarantine_key(ObjectType.TARGET, "abc")
    assert key == (DecisionType.QUARANTINE, ObjectType.TARGET, "abc", "abc")


def test_quarantine_key_stack():
    key = quarantine_key(ObjectType.STACK, "s1")
    assert key == (DecisionType.QUARANTINE, ObjectType.STACK, "s1", "s1")


def test_calibration_key_flat():
    key = calibration_key("flat", "stack1", "run1")
    assert key == (DecisionType.CALIBRATION_FLAT, ObjectType.STACK, "stack1", "run1")


def test_calibration_key_dark():
    key = calibration_key("dark", "s", "r")
    assert key == (DecisionType.CALIBRATION_DARK, ObjectType.STACK, "s", "r")


def test_calibration_key_bias():
    key = calibration_key("bias", "s", "r")
    assert key == (DecisionType.CALIBRATION_BIAS, ObjectType.STACK, "s", "r")


def test_calibration_key_invalid():
    with pytest.raises(KeyError):
        calibration_key("invalid", "s", "r")


def test_exclude():
    key = quarantine_key(ObjectType.STACK, "s1")
    result = exclude(key)
    assert result == {key: DecisionOverride(choice=BinaryChoice.EXCLUDE)}


def test_include():
    key = quarantine_key(ObjectType.STACK, "s1")
    result = include(key)
    assert result == {key: DecisionOverride(choice=BinaryChoice.INCLUDE)}


def test_exclude_composable_with_pipe():
    k1 = quarantine_key(ObjectType.STACK, "s1")
    k2 = quarantine_key(ObjectType.STACK, "s2")
    combined = exclude(k1) | exclude(k2)
    assert len(combined) == 2
    assert combined[k1].choice == BinaryChoice.EXCLUDE
    assert combined[k2].choice == BinaryChoice.EXCLUDE


def test_calibration_override():
    result = calibration_override("flat", "stack1", "run1", "flat_stack_42")
    key = calibration_key("flat", "stack1", "run1")
    assert result == {key: DecisionOverride(choice="flat_stack_42")}


def test_calibration_override_composable():
    o1 = calibration_override("flat", "s1", "r1", "f1")
    o2 = calibration_override("dark", "s1", "r1", "d1")
    combined = o1 | o2
    assert len(combined) == 2


def test_overrides_work_with_frozendict():
    overrides = frozendict(
        exclude(quarantine_key(ObjectType.STACK, "s1"))
        | calibration_override("flat", "s1", "r1", "f1")
    )
    assert len(overrides) == 2
