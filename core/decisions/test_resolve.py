"""Unit tests for decision resolve helpers."""
from decisions.decisions import (
    DecisionKey, DecisionType, ObjectType, DECISION_PRECEDENCE,
    _CAL_TYPE_TO_DECISION_TYPE,
)


def test_decision_precedence_order():
    """Quarantine should be applied before rotation outlier, which is before subframe quality."""
    assert DECISION_PRECEDENCE[DecisionType.QUARANTINE] < DECISION_PRECEDENCE[DecisionType.ROTATION_OUTLIER]
    assert DECISION_PRECEDENCE[DecisionType.ROTATION_OUTLIER] < DECISION_PRECEDENCE[DecisionType.SUBFRAME_QUALITY]


def test_calibrations_for_image_type():
    """Verify the Landmann §7.1 Case 2 calibration topology (matched-exposure
    darks; no master bias for lights; no pre-calibrated master dark). This is
    the single source of truth shared by light resolution AND cal-to-cal
    association."""
    from calibration.topology import CALIBRATION_TOPOLOGY
    assert CALIBRATION_TOPOLOGY["light"] == ("dark", "flat")
    assert CALIBRATION_TOPOLOGY["dark"] == ()
    assert CALIBRATION_TOPOLOGY["flat"] == ("bias",)
    assert CALIBRATION_TOPOLOGY["bias"] == ()


def test_cal_decision_types():
    """Verify the calibration decision type mapping."""
    assert _CAL_TYPE_TO_DECISION_TYPE["bias"] == DecisionType.CALIBRATION_BIAS
    assert _CAL_TYPE_TO_DECISION_TYPE["dark"] == DecisionType.CALIBRATION_DARK
    assert _CAL_TYPE_TO_DECISION_TYPE["flat"] == DecisionType.CALIBRATION_FLAT


def test_decision_key_via_calibration_constructor():
    key = DecisionKey.calibration("bias", "stack_a", "run_b")
    assert key.decision_type == DecisionType.CALIBRATION_BIAS
    assert key.object_type == ObjectType.STACK
    assert key.parent_stack_id == "stack_a"
    assert key.subject_id == "run_b"
