"""Unit tests for decision resolve helpers."""
from frozendict import frozendict

from decisions.decisions import (
    DecisionType, ObjectType, DecisionOverride, BinaryChoice, DECISION_PRECEDENCE,
)
from decisions.helpers import exclude, quarantine_key
from decisions.resolve import _apply_run_exclusions


def test_decision_precedence_order():
    """Quarantine should be applied before rotation outlier, which is before subframe quality."""
    assert DECISION_PRECEDENCE[DecisionType.QUARANTINE] < DECISION_PRECEDENCE[DecisionType.ROTATION_OUTLIER]
    assert DECISION_PRECEDENCE[DecisionType.ROTATION_OUTLIER] < DECISION_PRECEDENCE[DecisionType.SUBFRAME_QUALITY]


def test_calibrations_for_image_type():
    """Verify the calibration-for-image-type mapping."""
    from decisions.resolve import _CALIBRATIONS_FOR_IMAGE_TYPE
    assert _CALIBRATIONS_FOR_IMAGE_TYPE["light"] == ["bias", "dark", "flat"]
    assert _CALIBRATIONS_FOR_IMAGE_TYPE["dark"] == ["bias"]
    assert _CALIBRATIONS_FOR_IMAGE_TYPE["flat"] == ["bias"]
    assert _CALIBRATIONS_FOR_IMAGE_TYPE["bias"] == []


def test_cal_decision_types():
    """Verify the calibration decision type mapping."""
    from decisions.resolve import _CAL_DECISION_TYPES
    assert _CAL_DECISION_TYPES["bias"] == DecisionType.CALIBRATION_BIAS
    assert _CAL_DECISION_TYPES["dark"] == DecisionType.CALIBRATION_DARK
    assert _CAL_DECISION_TYPES["flat"] == DecisionType.CALIBRATION_FLAT
