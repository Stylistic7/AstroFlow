"""Apply decisions to a Stack to produce filtered or annotated views."""
from __future__ import annotations

from collections import Counter
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

# Calibration topology (which masters each image type depends on) lives in
# `calibration.topology.CALIBRATION_TOPOLOGY` — the single source of truth
# shared with the cal-to-cal associator, so the two can't drift.


# What the user actually SEES when each kind of master is missing. A generic
# message here said "vignetting or dust shadows" for a missing bias, which is
# the flat's symptom and sends someone looking for the wrong problem.
_UNCALIBRATED_SYMPTOM = {
    "bias": "leaves the sensor's read offset and fixed-pattern noise in",
    "dark": "leaves hot pixels, amp glow and dark current in",
    "flat": "leaves vignetting and dust shadows uncorrected",
}

# The user's explicit "no flat" — a pinned choice, distinct from "no
# override" (which now means the star flat). Deliberate, so no concern.
NO_FLAT = "none"


def _star_flat_for(run, cal_stacks, cam_id) -> str | None:
    """The group's star flat for this run, or None: same sensor geometry,
    same filter, same optical train (2% FOCALLEN), preferring the same
    camera fingerprint; among several, the one solved from the most subs."""
    from calibration.calibration_frames import focal_length_of, same_optical_train, sensor_of
    run_focal = focal_length_of(run)
    cands = [
        (sid, s) for sid, s in cal_stacks.items()
        if getattr(s, "kind", None) == "star"
        and s.filter == run.filter
        and sensor_of(s) == sensor_of(run)
        and same_optical_train(run_focal, s.focal_length)
    ]
    if not cands:
        return None
    cands = _prefer_same_camera(cands, cam_id, lambda c: c[1].camera_id_hash)
    return max(cands, key=lambda c: (c[1].n_subs, c[0]))[0]


def _prefer_same_camera(cands: list, cam_id: str | None, cam_of) -> list:
    """Narrow candidates to the SAME physical camera, where that is knowable.

    `camera_id_hash` is a defect fingerprint, so it exists only for frames
    that carried enough hot pixels to cluster — biases and clean flats often
    have none, and external pre-stacked masters have none at all (they are
    integrations, and `extract_defects` wants a raw mono frame).

    So this PREFERS rather than REQUIRES: when both sides have an identity
    and some candidate matches, that is the strongest evidence available and
    it wins outright; otherwise the whole sensor-geometry bucket stands.
    Requiring it would recreate the bug this replaced — refusing to
    calibrate because a master could not be fingerprinted is how stacks
    silently lost their masters in the first place.
    """
    if cam_id is None:
        return cands
    same = [c for c in cands if cam_of(c) == cam_id]
    return same or cands


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
    from calibration.calibration_frames import (
        _build_cal_index, _midpoint_s, sensor_of,
        focal_length_of, same_optical_train,
    )
    from calibration.directory_affinity import find_external_master
    from calibration.topology import CALIBRATION_TOPOLOGY

    bias_by_sensor, dark_by_sensor, flat_by_sensor_filter = _build_cal_index(
        cal_stacks
    )
    # Camera identity for THIS stack: a defect fingerprint, or None when its
    # frames carried no usable hot pixels. Used to prefer, never to require.
    cam_id = stack.camera_id_hash

    def _chronological_bias(run: Run):
        cand = bias_by_sensor.get(sensor_of(run))
        if not cand:
            return None
        cand = _prefer_same_camera(cand, cam_id, lambda x: x[2])
        # NO temperature window. A bias is offset + read noise, which on
        # CMOS (IMX455/571) does not vary with sensor temperature — unlike a
        # dark, where it is the whole point. The old `abs(c[1] - T) < 10.0`
        # filter starved every warm flat of a bias master and raised
        # thousands of "Stacking with no bias master" concerns (2026-09-07).
        # Candidates are already bucketed by sensor geometry and narrowed by
        # camera fingerprint, so the nearest in TIME is the right pick.
        # (Gain/offset matching would be better still, but `_build_cal_index`
        # does not carry them — a follow-up, not a blocker.)
        mid = _midpoint_s(run.oldest_date, run.newest_date)
        return min(cand, key=lambda x: abs(x[0] - mid))[3]

    def _chronological_dark(run: Run):
        cand = dark_by_sensor.get(sensor_of(run))
        if not cand:
            return None
        cand = _prefer_same_camera(cand, cam_id, lambda x: x[3])
        mid = _midpoint_s(run.oldest_date, run.newest_date)
        T = run.mean_temperature
        exp = run.exposure
        filtered = [
            c for c in cand if abs(c[1] - T) < 10.0 and c[2] <= exp
        ]
        if not filtered:
            return None
        return min(filtered, key=lambda x: abs(x[0] - mid))[4]

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
        cand = flat_by_sensor_filter.get((sensor_of(run), run.filter))
        if not cand:
            return None
        cand = [c for c in cand if c[3] not in quarantined_flat_ids]
        # A flat from a different optical train (FOCALLEN mismatch beyond
        # tolerance) is REFUSED outright: its vignetting profile is wrong
        # under any interpretation, and with two same-sensor rigs shooting
        # same-night flats, chronology alone would pick between them at
        # random. No candidate at all (concern raised downstream) beats a
        # wrong-train flat. Unknown FOCALLEN on either side still matches.
        run_focal = focal_length_of(run)
        cand = [c for c in cand if same_optical_train(run_focal, c[2])]
        if not cand:
            return None
        cand = _prefer_same_camera(cand, cam_id, lambda x: x[1])
        mid = _midpoint_s(run.oldest_date, run.newest_date)
        return min(cand, key=lambda x: abs(x[0] - mid))[3]

    def _sky_flat_for(run: Run):
        """The nearest NIGHT-SKY flat in time for this run's optical train.

        Matched exactly like any other flat — same sensor geometry, same
        filter, same train, preferring the same camera — because that is
        what it is. What sets it apart is its ILLUMINANT: lit by the night
        sky's emission lines, it captures the sensor's etalon at the same
        spectral weighting the lights see it, which a panel or twilight flat
        (continuum) cannot. Hence its place at the TOP of the flat ladder.

        Nothing selects one that a user did not create: `cal_stacks` carries
        a night-sky flat only where a Stack was explicitly promoted."""
        run_focal = focal_length_of(run)
        cand = [
            (s.oldest_date, sid, s) for sid, s in cal_stacks.items()
            if getattr(s, "kind", None) == "sky"
            and s.filter == run.filter
            and sensor_of(s) == sensor_of(run)
            and same_optical_train(run_focal, getattr(s, "focal_length", None))
        ]
        if not cand:
            return None
        cand = _prefer_same_camera(cand, cam_id, lambda c: c[2].camera_id_hash)
        mid = _midpoint_s(run.oldest_date, run.newest_date)
        return min(cand, key=lambda c: abs(
            _midpoint_s(c[2].oldest_date, c[2].newest_date) - mid))[1]

    chronological = {
        "bias": _chronological_bias,
        "dark": _chronological_dark,
        "flat": _chronological_flat,
    }

    def _default_choice(cal_type: str, run: Run) -> str | None:
        """Flats: the group's star flat first. Then directory-affinity; then
        chronological.

        A star flat (`stacks/star_flat.py`) is solved from the photometry of
        every light shot through the run's (camera, focal length, filter)
        and corrects vignetting AND the narrowband etalon a panel cannot;
        it exists only where the group passed the quality gate, so a sparse
        train falls through to the older defaults below.
        Directory-affinity returns an absolute path to a pre-stacked external
        master sitting in an ancestor directory of the run's frames (typically
        a `Calibration Masters/MasterDark*.fit` file under the target's tree).
        Chronological returns a StackHash of an AstroFlow-stacked calibration
        stack. Disambiguated downstream by whether the choice starts with "/".
        """
        if cal_type == "flat":
            # Three flavours of flat, most spectrally-matched first: a
            # night-sky flat sees the etalon exactly as the lights do; a star
            # flat measures it from continuum and needs k to scale it; a
            # panel/twilight flat barely sees it at all.
            sky = _sky_flat_for(run)
            if sky is not None:
                return sky
            star = _star_flat_for(run, cal_stacks, cam_id)
            if star is not None:
                return star
        path = find_external_master(
            (m.filename for m in run.metadata.values()),
            kind=cal_type,
            shape=sensor_of(run),
            temperature=run.mean_temperature,
            filter=(run.filter if cal_type == "flat" else None),
            exposure=(run.exposure if cal_type == "dark" else None),
        )
        if path is not None:
            return path
        return chronological[cal_type](run)

    new_contents = RunsDict()
    missing: Counter[str] = Counter()
    for run_hash, run in stack.contents.items():
        deps = dict(run.dependencies)
        for cal_type in CALIBRATION_TOPOLOGY.get(run.image_type, ()):
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

            if choice == NO_FLAT:
                continue   # the user pinned "no flat": deliberate, no concern
            if choice is None:
                # Nothing matched. Proceeding is deliberate — plenty of
                # people never shoot flats and failing their data outright
                # would be worse — but it must not be silent. An
                # uncalibrated integration looks perfectly fine and is
                # wrong, so it is surfaced on the stack instead of being
                # dropped here (which is how four Starbase stacks were
                # integrated flat-less without anyone noticing).
                missing[cal_type] += 1
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

    if missing:
        from concerns import warn as _concern_warn
        kinds = "/".join(sorted(missing))
        _concern_warn(
            f"Stacking with no {kinds} master",
            detail=(
                "; ".join(
                    f"{n} run{'s' if n != 1 else ''} had no {k} master to "
                    f"match ({_UNCALIBRATED_SYMPTOM[k]})"
                    for k, n in sorted(missing.items())
                )
                + ". The integration proceeds uncalibrated — that is "
                "deliberate, since not everyone shoots every kind of "
                "calibration frame, and it is reported rather than silent "
                "because the result looks perfectly plausible either way. "
                "Either shoot the missing frames, or choose a master "
                "explicitly from this stack's calibration decisions."
            ),
            stack=stack.stable_hash,
            missing=dict(sorted(missing.items())),
        )

    return replace(stack, contents=new_contents)
