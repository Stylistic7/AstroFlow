"""Apply decisions to a Stack to produce filtered or annotated views."""
from __future__ import annotations

from dataclasses import replace
from frozendict import frozendict

from stacks.composite_stack import Stack, CalibrationStacksDict
from stacks.run import Run, RunsDict
from stacks.types import FilenameDependency, StackIdDependency
from decisions.decisions import (
    BinaryChoice, DecisionKey, DecisionOverride, DecisionPoint, DecisionType,
    ObjectType, DECISION_PRECEDENCE, resolve_decision, get_decisions_for_object,
)
from decisions.subframe_selection import select_subframes
from quarantining.quarantining import apply_quarantine


def _apply_run_exclusions(
    stack: Stack,
    decision_overrides: frozendict[DecisionKey, DecisionOverride],
    decision_type: DecisionType,
) -> Stack | None:
    """Remove runs excluded by a specific decision type."""
    excluded = {
        key.subject_id
        for key, override in decision_overrides.items()
        if (
            key.decision_type == decision_type
            and key.object_type == ObjectType.STACK
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
    quality_key = DecisionKey.subframe_quality(stack.stable_hash)

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
            from decisions.subframe_selection import _DEFAULT_QUALITY_THRESHOLD
            quality_threshold = _DEFAULT_QUALITY_THRESHOLD

    from decisions.subframe_selection import _compute_threshold_stars
    threshold_stars = _compute_threshold_stars(metadata, stack.all_frames(), quality_threshold)
    return m.plate_solution.num_stars >= threshold_stars


def is_run_included(
    run_hash: str,
    stack: Stack,
    decision_overrides: frozendict[DecisionKey, DecisionOverride],
) -> bool:
    """Check if a run survives rotation/quarantine decisions. For per-run annotation."""
    keys = (
        DecisionKey.quarantine(
            ObjectType.STACK, run_hash, parent_stack_id=stack.stable_hash,
        ),
        DecisionKey.rotation_outlier(stack.stable_hash, run_hash),
    )

    for key in keys:
        override = decision_overrides.get(key)
        if override is not None and override.choice == BinaryChoice.EXCLUDE:
            return False

    # Check published decisions — if the default is EXCLUDE (unlikely but possible).
    decisions = get_decisions_for_object(ObjectType.STACK, stack.stable_hash)
    for key in keys:
        dp = decisions.get(key)
        if dp is not None:
            choice = resolve_decision(dp, decision_overrides, default=dp.default_choice)
            if choice == BinaryChoice.EXCLUDE:
                return False

    return True


# ---------------------------------------------------------------------------
# Calibration resolution via the decision framework
# ---------------------------------------------------------------------------

# Calibration topology — Landmann §7.1 Case 2 (matched-exposure darks).
#
# `dark: []`    — integrate raw darks WITHOUT pre-calibrating them
#                 (Landmann §6.2: "NEVER pre-calibrate the MasterDark"
#                 — pre-calibration trips WBPP's §8.2 undercorrection
#                 path when paired with the master-bias-for-lights
#                 anti-pattern).
# `flat: ["bias"]` — bias-calibrate CMOS short flats (Landmann §6.3.1).
# `light: ["dark", "flat"]` — NO master bias for lights. Case 2's
#                 closed form is `(L − MD) · s0 / MF` with no MB term;
#                 staging MB into the lights' WBPP working dir would
#                 cause WBPP to subtract it again on a master dark
#                 that already contains the bias (§8.2). With this
#                 topology WBPP's auto-detect lands on the documented
#                 Case 2 settings.
_CALIBRATIONS_FOR_IMAGE_TYPE = {
    "bias": [],
    "dark": [],
    "flat": ["bias"],
    "light": ["dark", "flat"],
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
    from calibration.directory_affinity import find_external_master

    bias_by_cam, dark_by_cam, flat_by_cam_filter = _build_cal_index(cal_stacks)

    def _chronological_bias(run: Run):
        cand = bias_by_cam.get(run.camera)
        if not cand:
            return None
        mid = _midpoint_s(run.oldest_date, run.newest_date)
        T = run.mean_temperature
        filtered = [(m, t, sid) for (m, t, sid) in cand if abs(t - T) < 10.0]
        if not filtered:
            return None
        return min(filtered, key=lambda x: abs(x[0] - mid))[2]

    def _chronological_dark(run: Run):
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

    # Build set of quarantined flat stack IDs so auto-selection skips them.
    quarantined_flat_ids = {
        key.subject_id
        for key, override in decision_overrides.items()
        if key.decision_type == DecisionType.QUARANTINE
        and override.choice == BinaryChoice.EXCLUDE
        and key.subject_id in cal_stacks
        and cal_stacks[key.subject_id].image_type == "flat"
    }

    def _chronological_flat(run: Run):
        cand = flat_by_cam_filter.get((run.camera, run.filter))
        if not cand:
            return None
        cand = [(m, sid) for m, sid in cand if sid not in quarantined_flat_ids]
        if not cand:
            return None
        mid = _midpoint_s(run.oldest_date, run.newest_date)
        return min(cand, key=lambda x: abs(x[0] - mid))[1]

    chronological = {
        "bias": _chronological_bias,
        "dark": _chronological_dark,
        "flat": _chronological_flat,
    }

    def _default_choice(cal_type: str, run: Run) -> str | None:
        """Try directory-affinity first; fall back to chronological.

        Directory-affinity returns an absolute path to a pre-stacked external
        master sitting in an ancestor directory of the run's frames (typically
        a `Calibration Masters/MasterDark*.fit` file under the target's tree).
        Chronological returns a StackHash of an AstroFlow-stacked calibration
        stack. Disambiguated downstream by whether the choice starts with "/".
        """
        path = find_external_master(
            (m.filename for m in run.metadata.values()),
            kind=cal_type,
            camera=run.camera,
            temperature=run.mean_temperature,
            filter=(run.filter if cal_type == "flat" else None),
            exposure=(run.exposure if cal_type == "dark" else None),
        )
        if path is not None:
            return path
        return chronological[cal_type](run)

    new_contents = RunsDict()
    for run_hash, run in stack.contents.items():
        deps = dict(run.dependencies)
        for cal_type in _CALIBRATIONS_FOR_IMAGE_TYPE.get(run.image_type, []):
            if cal_type in deps:
                continue  # Already set (e.g., from cal-to-cal association)

            decision = DecisionPoint.make(
                key=DecisionKey.calibration(cal_type, stack.stable_hash, run_hash),
                title=f"Calibration: {cal_type}",
                description=f"Which {cal_type} source to use for this run",
            )

            choice = resolve_decision(
                decision,
                decision_overrides,
                default=lambda c=cal_type, r=run: _default_choice(c, r),
            )

            if choice is None:
                continue
            if isinstance(choice, str) and choice.startswith("/"):
                deps[cal_type] = FilenameDependency(filename=choice)
            else:
                # resolve_decision now always returns the fresh-computed
                # default (or the user override). A choice not in
                # cal_stacks at this point means the user explicitly
                # overrode to a stack that no longer exists — fail loud
                # rather than silently dropping the dependency.
                assert choice in cal_stacks, (
                    f"calibration override for {cal_type} on stack "
                    f"{stack.stable_hash} run {run_hash} points to "
                    f"{choice!r}, which is not in the current cal_stacks"
                )
                deps[cal_type] = StackIdDependency(stack_id=choice)

        new_contents[run_hash] = replace(run, dependencies=frozendict(deps))

    return replace(stack, contents=new_contents)
