import datetime
import math
import pytest
from frozendict import frozendict

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from testing import make_synthetic_frames
from decisions.decisions import (
    DecisionKey,
    DecisionPoint,
    DecisionType,
    BinaryChoice,
    ObjectType,
    DecisionOverride,
    resolve_decision,
    get_decisions_for_object,
)
from decisions.resolve import resolve_decisions
from decisions.compute_graph import ALL_EXCLUSION_TYPES


def test_decisions_live_in_the_durable_store():
    """Decisions are user curation (quarantines, calibration overrides,
    subframe-quality choices) — NOT regenerable @node results — so they must
    route to the durable cache and survive LRU eviction / regular-cache wipes.
    The `("decisions", ...)` namespace being evictable was a real data-loss
    hazard."""
    from context.cache import is_durable_key
    from decisions.decisions import _decisions_key
    key = _decisions_key(ObjectType.STACK, "abc123")
    assert is_durable_key(key) is True


def _make_decision(key: DecisionKey) -> DecisionPoint:
    return DecisionPoint(
        key=key,
        decision_type=key.decision_type,
        default_choice=BinaryChoice.INCLUDE,
        title="test",
        description="test",
        object_type=key.object_type,
        object_id=key.parent_stack_id,
        subject_id=key.subject_id,
    )


def test_resolve_decision_default(test_context):
    key = DecisionKey.rotation_outlier("s1", "r1")
    dp = _make_decision(key)
    # No override → default is always whatever the current call passes.
    # The cache in context is output-only (refreshed for the UI),
    # never read back as a source of truth — so a different `default`
    # on the next call wins.
    assert resolve_decision(dp, frozendict(), default=BinaryChoice.INCLUDE) == BinaryChoice.INCLUDE
    assert resolve_decision(dp, frozendict(), default=BinaryChoice.EXCLUDE) == BinaryChoice.EXCLUDE


def test_resolve_decision_callable_default(test_context):
    key = DecisionKey.rotation_outlier("s2", "r2")
    dp = _make_decision(key)
    call_count = [0]
    def expensive_default():
        call_count[0] += 1
        return BinaryChoice.INCLUDE
    # The callable runs every time — the cache isn't a memoization
    # layer, it's a UI-display side effect. Callers that need to
    # amortize an expensive computation should do so at their own
    # call site (e.g., resolve_calibration's _build_cal_index).
    assert resolve_decision(dp, frozendict(), default=expensive_default) == BinaryChoice.INCLUDE
    assert call_count[0] == 1
    assert resolve_decision(dp, frozendict(), default=expensive_default) == BinaryChoice.INCLUDE
    assert call_count[0] == 2


def test_resolve_decision_override(test_context):
    key = DecisionKey.rotation_outlier("s1", "r1")
    dp = _make_decision(key)
    overrides = frozendict({
        key: DecisionOverride(choice=BinaryChoice.EXCLUDE),
    })
    # User override wins regardless of default.
    assert resolve_decision(dp, overrides, default=BinaryChoice.INCLUDE) == BinaryChoice.EXCLUDE


def test_resolve_decision_override_caches_default_for_ui(test_context):
    """When there's an override, the default is still computed and cached for the UI."""
    key = DecisionKey.rotation_outlier("s3", "r3")
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


def test_decisions_for_objects_scopes_to_closure():
    """The cache key of an image may depend only on decisions pertaining to
    that object or a contained object. `decisions_for_objects` enforces the
    mosaic/target side of that rule: keys owned by closure ids stay, keys
    owned by anything else are dropped — so a decision on an unrelated
    object can never move this pipeline's config hash."""
    from types import SimpleNamespace
    from decisions.decisions import (
        BinaryChoice, DecisionKey, DecisionOverride, ObjectType,
        decisions_for_objects,
    )

    mos_key = DecisionKey.quarantine(ObjectType.MOSAIC, "mosaic_aaaa")
    run_key = DecisionKey.rotation_outlier("stack_bbbb", "run_1")
    unrelated = DecisionKey.quarantine(ObjectType.STACK, "stack_zzzz")
    o = DecisionOverride(choice=BinaryChoice.EXCLUDE)
    overrides = frozendict({mos_key: o, run_key: o, unrelated: o})

    scoped = decisions_for_objects(overrides, {"mosaic_aaaa", "stack_bbbb"})
    assert mos_key in scoped          # object-level key, owned by closure
    assert run_key in scoped          # run-level key, owning stack in closure
    assert unrelated not in scoped    # unrelated ownership NEVER enters

    # Empty overrides: scoped == global — the invariant that made deploying
    # this fix hash-neutral while no decisions exist.
    assert decisions_for_objects(frozendict(), {"x"}) == frozendict()

    # Shared-calibration inclusion: a decision parented on a cal-stack id is
    # consumed by calibration_plan/resolve_calibration inside ANY pipeline
    # that might select that master — config_for unions cal-stack ids into
    # every closure, so the key must survive the filter when they are.
    cal_key = DecisionKey.quarantine(ObjectType.STACK, "cal_dddd")
    overrides2 = frozendict({cal_key: o, unrelated: o})
    scoped2 = decisions_for_objects(
        overrides2, {"mosaic_aaaa"} | {"cal_dddd"},  # closure | cal_stack_ids
    )
    assert cal_key in scoped2
    assert unrelated not in scoped2


def test_decision_closure_ids_walks_mosaic_containment():
    """The closure of a mosaic = itself + its targets + their stacks,
    resolved from the build's own objects dicts via plain id tuples."""
    from types import SimpleNamespace
    from compute_graph.compute_graph import _decision_closure_ids

    t1 = SimpleNamespace(stable_hash="t1", stack_ids=("s1", "s2"))
    mosaic = SimpleNamespace(stable_hash="m1", target_ids=("t1",))
    ids = _decision_closure_ids(mosaic, "m1", {"target": {"t1": t1}})
    assert ids == frozenset({"m1", "t1", "s1", "s2"})

    # A target entity contributes its own stacks.
    tgt = SimpleNamespace(stable_hash="t9", stack_ids=("s9",))
    ids = _decision_closure_ids(tgt, "t9", {"target": {}})
    assert ids == frozenset({"t9", "s9"})


def test_subframe_quality_decision_and_filtering(test_context):
    """Default keeps ALL subframes (no decision spam); a per-Stack
    override dials a specific Stack down even though the default would
    exclude nothing. Also covers the RunsTab selected/total helpers."""
    from decisions.decisions import DecisionKey
    from decisions.subframe_selection import (
        resolved_quality_threshold, selected_counts_by_run,
    )

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
    assert len(stack.all_frames()) == 15

    # Default keeps everything → NO subframe-quality decision published
    # (we don't nag the user when nothing is excluded).
    quality_decisions = {
        k: v for k, v in
        get_decisions_for_object(ObjectType.STACK, stack_hash).items()
        if v.decision_type == DecisionType.SUBFRAME_QUALITY
    }
    assert quality_decisions == {}

    # Default: keep all 15; per-run helper reports everything selected.
    assert resolved_quality_threshold(stack, frozendict()) == 0.0
    resolved = resolve_decisions(
        stack, ALL_EXCLUSION_TYPES, frozendict(), min_frames_in_stack=1
    )
    assert len(resolved.all_frames()) == 15
    counts = selected_counts_by_run(stack, frozendict())
    assert sum(s for s, _ in counts.values()) == 15
    assert all(s == t for s, t in counts.values())

    # Per-Stack override dials THIS stack down even though the default
    # (keep-all) excluded nothing — the key requirement.
    quality_key = DecisionKey.subframe_quality(stack_hash)
    strict = frozendict({quality_key: DecisionOverride(choice=0.9)})
    assert resolved_quality_threshold(stack, strict) == 0.9
    resolved = resolve_decisions(
        stack, ALL_EXCLUSION_TYPES, strict, min_frames_in_stack=1
    )
    assert len(resolved.all_frames()) <= 10
    s_counts = selected_counts_by_run(stack, strict)
    assert sum(s for s, _ in s_counts.values()) <= 10
    # Totals (full run sizes) are unaffected by the threshold.
    assert {h: t for h, (_, t) in counts.items()} == {
        h: t for h, (_, t) in s_counts.items()
    }

    # Explicit keep-all override behaves like the default.
    keep_all = frozendict({quality_key: DecisionOverride(choice=0.0)})
    resolved = resolve_decisions(
        stack, ALL_EXCLUSION_TYPES, keep_all, min_frames_in_stack=1
    )
    assert len(resolved.all_frames()) == 15


def test_no_subframe_decision_for_minor_exclusion(test_context):
    """The default keeps all frames, so no subframe-quality decision is
    generated (nothing is excluded to nag about)."""
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


def test_star_flat_is_the_default_flat_and_none_is_a_pinned_choice(test_context, monkeypatch):
    """`resolve_calibration` on a stack from a real graph run (which also
    exercises the star-flat tasks wired above the spatial split — with no
    plate solutions there are no candidates, and the graph must still run):
    the group's star flat is the default flat; an override of NO_FLAT drops
    the flat WITHOUT the missing-master concern; an explicit flat override
    beats the star flat."""
    import concerns
    from decisions.resolve import NO_FLAT, resolve_calibration
    from stacks.composite_stack import CalibrationStack, CalibrationStacksDict
    from stacks.run import RunsDict
    from stacks.star_flat import StarFlatStack
    from stacks.types import StackHash, StackIdDependency
    from units import Celsius, Seconds

    make_synthetic_frames(number=10, dest_dir=context.input_dirs[0],
                          start_datetime=datetime.datetime(2023, 12, 1, 20, 0, 0))
    run_graph(build_compute_graph())
    stack = next(iter(context.stacks.values())).get()
    (run_hash, run), = stack.contents.items()

    def _flat(cls, sid, **kw):
        fields = dict(content_hash=sid, image_type="flat", camera="cam", filter=run.filter, mean_temperature=Celsius(-10.0),
                      exposure=Seconds(1.0), integration=Seconds(0.0), width_pixels=run.width_pixels,
                      height_pixels=run.height_pixels, stable_hash=sid, oldest_date=datetime.datetime(2026, 1, 1),
                      newest_date=datetime.datetime(2026, 1, 2), state=frozendict(), contents=RunsDict(), camera_id_hash=None)
        fields.update(kw)
        return cls(**fields)

    cal = CalibrationStacksDict({
        StackHash("star"): _flat(StarFlatStack, "star", focal_length=2066.0, member_files=("/a", "/b")),
        StackHash("explicit"): _flat(CalibrationStack, "explicit"),
    })
    warned = []
    monkeypatch.setattr(concerns, "warn", lambda title, detail="", **f: warned.append(f.get("missing", {})))
    key = DecisionKey.calibration("flat", stack.stable_hash, run_hash)

    resolved = resolve_calibration(stack, cal, frozendict())
    assert resolved.contents[run_hash].dependencies["flat"] == StackIdDependency(stack_id="star")
    resolved = resolve_calibration(stack, cal, frozendict({key: DecisionOverride(choice=NO_FLAT)}))
    assert "flat" not in resolved.contents[run_hash].dependencies
    resolved = resolve_calibration(stack, cal, frozendict({key: DecisionOverride(choice="explicit")}))
    assert resolved.contents[run_hash].dependencies["flat"] == StackIdDependency(stack_id="explicit")
    assert warned and all("flat" not in m for m in warned)      # darks are missing (a concern); "none" never is
