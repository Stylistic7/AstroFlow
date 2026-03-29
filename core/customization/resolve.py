"""Apply decisions to a Stack to produce filtered or annotated views."""
from __future__ import annotations

from dataclasses import replace
from frozendict import frozendict

from stacks.composite_stack import Stack, CalibrationStacksDict
from stacks.run import Run, RunsDict
from stacks.types import StackIdDependency
from customization.decisions import (
    BinaryChoice, DecisionKey, DecisionOverride, DecisionPoint, DecisionType,
    ObjectType, DECISION_PRECEDENCE, resolve_decision, get_decisions_for_object,
)
from customization.subframe_selection import select_subframes
from quarantining.quarantining import apply_quarantine


def _apply_run_exclusions(
    stack: Stack,
    decision_overrides: frozendict[DecisionKey, DecisionOverride],
    decision_type: DecisionType,
) -> Stack | None:
    """Remove runs excluded by a specific decision type."""
    excluded = {
        key[3]  # subject_id = run hash
        for key, override in decision_overrides.items()
        if (
            key[0] == decision_type
            and key[1] == ObjectType.STACK
            and override.choice == BinaryChoice.EXCLUDE
        )
    }
    if not excluded:
        return stack
    return apply_quarantine(stack, excluded)


def resolve_decisions(
    stack: Stack,
    decision_types: tuple[DecisionType, ...],
    decision_overrides: frozendict[DecisionKey, DecisionOverride],
    min_frames_in_stack: int,
) -> Stack | None:
    """Apply the specified decision types to produce a filtered view.

    Decision types are automatically sorted by DECISION_PRECEDENCE (lower
    first): run-level exclusions before frame-level exclusions.
    """
    sorted_types = sorted(
        decision_types,
        key=lambda dt: DECISION_PRECEDENCE.get(dt, 999),
    )

    for dt in sorted_types:
        if stack is None:
            return None
        if dt in (DecisionType.QUARANTINE, DecisionType.ROTATION_OUTLIER):
            stack = _apply_run_exclusions(stack, decision_overrides, dt)
        elif dt == DecisionType.SUBFRAME_QUALITY:
            stack = select_subframes(
                stack, min_frames_in_stack,
                decision_overrides=decision_overrides,
            )

    return stack


def is_frame_included(
    frame_hash: str,
    stack: Stack,
    decision_overrides: frozendict[DecisionKey, DecisionOverride],
) -> bool:
    """Check if a frame survives the quality filter. For per-frame annotation."""
    metadata = stack.all_metadata()
    if frame_hash not in metadata:
        return False
    m = metadata[frame_hash]
    if m.plate_solution is None:
        return False

    # Check the resolved quality threshold from the decision cache.
    quality_key = (
        DecisionType.SUBFRAME_QUALITY, ObjectType.STACK,
        stack.stable_hash, stack.stable_hash,
    )

    # Check override first.
    override = decision_overrides.get(quality_key)
    if override is not None:
        quality_threshold = override.choice  # float, e.g. 0.0 to keep all
    else:
        # Check cached decision.
        decisions = get_decisions_for_object(ObjectType.STACK, stack.stable_hash)
        dp = decisions.get(quality_key)
        if dp is not None and dp.default_choice is not None:
            quality_threshold = dp.default_choice
        else:
            from customization.subframe_selection import _DEFAULT_QUALITY_THRESHOLD
            quality_threshold = _DEFAULT_QUALITY_THRESHOLD

    from customization.subframe_selection import _compute_threshold_stars
    threshold_stars = _compute_threshold_stars(metadata, stack.all_frames(), quality_threshold)
    return m.plate_solution.num_stars >= threshold_stars


def is_run_included(
    run_hash: str,
    stack: Stack,
    decision_overrides: frozendict[DecisionKey, DecisionOverride],
) -> bool:
    """Check if a run survives rotation/quarantine decisions. For per-run annotation."""
    for dt in (DecisionType.QUARANTINE, DecisionType.ROTATION_OUTLIER):
        key = (dt, ObjectType.STACK, stack.stable_hash, run_hash)
        override = decision_overrides.get(key)
        if override is not None and override.choice == BinaryChoice.EXCLUDE:
            return False

    # Check published decisions — if the default is EXCLUDE (unlikely but possible).
    decisions = get_decisions_for_object(ObjectType.STACK, stack.stable_hash)
    for dt in (DecisionType.QUARANTINE, DecisionType.ROTATION_OUTLIER):
        key = (dt, ObjectType.STACK, stack.stable_hash, run_hash)
        dp = decisions.get(key)
        if dp is not None:
            choice = resolve_decision(dp, decision_overrides, default=dp.default_choice)
            if choice == BinaryChoice.EXCLUDE:
                return False

    return True


# ---------------------------------------------------------------------------
# Calibration resolution via the decision framework
# ---------------------------------------------------------------------------

_CAL_DECISION_TYPES = {
    "bias": DecisionType.CALIBRATION_BIAS,
    "dark": DecisionType.CALIBRATION_DARK,
    "flat": DecisionType.CALIBRATION_FLAT,
}

_CALIBRATIONS_FOR_IMAGE_TYPE = {
    "bias": [],
    "dark": ["bias"],
    "flat": ["bias"],
    "light": ["bias", "dark", "flat"],
}


def resolve_calibration(
    stack: Stack,
    cal_stacks: CalibrationStacksDict,
    decision_overrides: frozendict[DecisionKey, DecisionOverride],
) -> Stack:
    """Resolve calibration dependencies for each run via the decision framework.

    For each run, for each calibration type (bias, dark, flat), the matching
    logic is wrapped in a lazy default. The decision framework returns the
    cached default or user override.
    """
    from calibration.calibration_frames import _build_cal_index, _midpoint_s

    bias_by_cam, dark_by_cam, flat_by_cam_filter = _build_cal_index(cal_stacks)

    def _choose_bias(run: Run):
        cand = bias_by_cam.get(run.camera)
        if not cand:
            return None
        mid = _midpoint_s(run.oldest_date, run.newest_date)
        T = run.mean_temperature
        filtered = [(m, t, sid) for (m, t, sid) in cand if abs(t - T) < 10.0]
        if not filtered:
            return None
        return min(filtered, key=lambda x: abs(x[0] - mid))[2]

    def _choose_dark(run: Run):
        cand = dark_by_cam.get(run.camera)
        if not cand:
            return None
        mid = _midpoint_s(run.oldest_date, run.newest_date)
        T = run.mean_temperature
        exp = run.exposure
        filtered = [(m, t, e, sid) for (m, t, e, sid) in cand if abs(t - T) < 10.0 and e <= exp]
        if not filtered:
            return None
        return min(filtered, key=lambda x: abs(x[0] - mid))[3]

    def _choose_flat(run: Run):
        # No automatic flat selection — the user must choose manually.
        return None

    pickers = {"bias": _choose_bias, "dark": _choose_dark, "flat": _choose_flat}

    new_contents = RunsDict()
    for run_hash, run in stack.contents.items():
        deps = dict(run.dependencies)
        for cal_type in _CALIBRATIONS_FOR_IMAGE_TYPE.get(run.image_type, []):
            if cal_type in deps:
                continue  # Already set (e.g., from cal-to-cal association)

            dt = _CAL_DECISION_TYPES[cal_type]
            decision = DecisionPoint.make(
                decision_type=dt,
                object_id=stack.stable_hash,
                subject_id=run_hash,
                title=f"Calibration: {cal_type}",
                description=f"Which {cal_type} stack to use for this run",
            )

            sid = resolve_decision(
                decision,
                decision_overrides,
                default=lambda r=run, p=pickers[cal_type]: p(r),
            )

            if sid is not None:
                deps[cal_type] = StackIdDependency(stack_id=sid)

        new_contents[run_hash] = replace(run, dependencies=frozendict(deps))

    return replace(stack, contents=new_contents)
