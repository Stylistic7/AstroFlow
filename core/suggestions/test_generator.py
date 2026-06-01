from types import SimpleNamespace
from unittest.mock import patch

from suggestions.generator import generate_suggestions
from suggestions.types import SuggestionKind


def _ref(value):
    return SimpleNamespace(get=lambda: value)


def _stack(stable_hash, filter, num_stars_per_frame, exposure_per_frame_s):
    """Build a minimal fake stack that looks enough like Stack for generator."""
    frames = tuple(f"frame_{stable_hash}_{i}" for i in range(len(num_stars_per_frame)))
    metadata = {
        f: SimpleNamespace(
            plate_solution=SimpleNamespace(num_stars=n),
            header={"EXPOSURE": e},
        )
        for f, n, e in zip(frames, num_stars_per_frame, exposure_per_frame_s)
    }
    return SimpleNamespace(
        stable_hash=stable_hash,
        filter=filter,
        all_frames=lambda: list(frames),
        all_metadata=lambda: metadata,
    )


def _target(stable_hash, name, stack_ids):
    return SimpleNamespace(
        stable_hash=stable_hash,
        name=name,
        center=(100.0, 0.0),
        target_ids=None,
        stack_ids=tuple(stack_ids),
    )


def _mosaic(stable_hash, target_ids):
    return SimpleNamespace(
        stable_hash=stable_hash,
        target_ids=tuple(target_ids),
    )


def _ctx(shopping_cart, mosaics, targets, stacks):
    return SimpleNamespace(
        shopping_cart=shopping_cart,
        mosaics={m.stable_hash: _ref(m) for m in mosaics},
        targets={t.stable_hash: _ref(t) for t in targets},
        stacks={s.stable_hash: _ref(s) for s in stacks},
    )


ALL_FILTERS = frozenset({"H", "S", "O", "L", "R", "G", "B"})


def test_completion_emitted_for_exposure_imbalance():
    # Two panels, both in H. Panel A has 5h (18000s), panel B has 2h (7200s).
    stack_a = _stack("stack_a", "H",
                     num_stars_per_frame=[100] * 60,
                     exposure_per_frame_s=[300] * 60)  # 60 * 300 = 18000s
    stack_b = _stack("stack_b", "H",
                     num_stars_per_frame=[100] * 24,
                     exposure_per_frame_s=[300] * 24)  # 24 * 300 = 7200s
    target_a = _target("t_a", "PanelA", [stack_a.stable_hash])
    target_b = _target("t_b", "PanelB", [stack_b.stable_hash])
    mosaic = _mosaic("m1", [target_a.stable_hash, target_b.stable_hash])

    ctx = _ctx(
        shopping_cart=[("mosaic", mosaic.stable_hash, "DownscaledMaster", "PixInsight")],
        mosaics=[mosaic],
        targets=[target_a, target_b],
        stacks=[stack_a, stack_b],
    )

    with patch("suggestions.generator.context", ctx):
        sugs = generate_suggestions(ALL_FILTERS)

    assert len(sugs) == 1
    s = sugs[0]
    assert s.kind == SuggestionKind.COMPLETION
    assert s.target_id == "t_b"
    assert s.filter == "H"
    assert s.current_exposure_s == 7200.0
    assert s.target_exposure_s == 18000.0
    assert s.default_exposure_per_frame_s == 300.0  # median of existing frames
    # needed = 10800s; 10800/300 = 36
    assert s.default_frame_count == 36


def test_missing_channel_emitted_when_panel_lacks_filter():
    # Panel A has H + O. Panel B has H only. Expect MISSING_CHANNEL for B in O.
    stack_a_h = _stack("a_h", "H", [100] * 30, [300] * 30)
    stack_a_o = _stack("a_o", "O", [100] * 30, [300] * 30)
    stack_b_h = _stack("b_h", "H", [100] * 30, [300] * 30)
    target_a = _target("t_a", "A", [stack_a_h.stable_hash, stack_a_o.stable_hash])
    target_b = _target("t_b", "B", [stack_b_h.stable_hash])
    mosaic = _mosaic("m1", [target_a.stable_hash, target_b.stable_hash])

    ctx = _ctx(
        shopping_cart=[("mosaic", mosaic.stable_hash, "DownscaledMaster", "PixInsight")],
        mosaics=[mosaic],
        targets=[target_a, target_b],
        stacks=[stack_a_h, stack_a_o, stack_b_h],
    )

    with patch("suggestions.generator.context", ctx):
        sugs = generate_suggestions(ALL_FILTERS)

    missing = [s for s in sugs if s.kind == SuggestionKind.MISSING_CHANNEL]
    assert len(missing) == 1
    assert missing[0].target_id == "t_b"
    assert missing[0].filter == "O"
    assert missing[0].current_exposure_s == 0.0
    assert missing[0].target_exposure_s == 9000.0  # 30 * 300


def test_completion_uses_stacker_reported_integration_when_master_available():
    """When a stack's master carries `master_integration_s` (i.e. the
    stacker has run and reported its own post-rejection total), the
    suggestion's current_exposure_s reflects that number — NOT
    AstroFlow's predicted-effective sum across raw frames. This is the
    whole point of routing through `effective_integration`.
    """
    # Panel A: predicted-effective would be 60 * 300 = 18000s.
    # Panel B: predicted-effective would be 24 * 300 = 7200s.
    # But the stacker dropped some frames on each, reporting:
    #   master A → 15000s  (kept 50/60 frames)
    #   master B →  6000s  (kept 20/24 frames)
    # Suggestion math should use the stacker numbers as ground truth:
    # peer max = 15000s, B's shortfall = 9000s, default_frame_count =
    # ceil(9000/300) = 30.
    stack_a = _stack("stack_a", "H", [100] * 60, [300] * 60)
    stack_b = _stack("stack_b", "H", [100] * 24, [300] * 24)
    target_a = _target("t_a", "PanelA", [stack_a.stable_hash])
    target_b = _target("t_b", "PanelB", [stack_b.stable_hash])
    mosaic = _mosaic("m1", [target_a.stable_hash, target_b.stable_hash])

    ctx = _ctx(
        shopping_cart=[("mosaic", mosaic.stable_hash, "PixInsight")],
        mosaics=[mosaic],
        targets=[target_a, target_b],
        stacks=[stack_a, stack_b],
    )

    fake_master_meta_by_stack = {
        "stack_a": SimpleNamespace(master_integration_s=15000.0),
        "stack_b": SimpleNamespace(master_integration_s=6000.0),
    }

    with patch("suggestions.generator.context", ctx), \
         patch("suggestions.generator._master_meta_by_stack",
               return_value=fake_master_meta_by_stack):
        sugs = generate_suggestions(ALL_FILTERS)

    assert len(sugs) == 1
    s = sugs[0]
    assert s.kind == SuggestionKind.COMPLETION
    assert s.target_id == "t_b"
    # Stacker's number, not 7200s (predicted-effective on raw frames).
    assert s.current_exposure_s == 6000.0
    assert s.target_exposure_s == 15000.0
    assert s.default_frame_count == 30  # ceil((15000 - 6000) / 300)


def test_per_frame_default_uses_all_captured_frames_not_just_keepers():
    """Per-frame cadence answers "what's the user's capture habit?" —
    it should reflect every captured EXPOSURE, including frames whose
    star count would otherwise be filtered out. Previously the median
    was over star-count keepers only, which biased the suggested
    cadence on capture sessions with mixed conditions."""
    # Panel A: all 60 frames at 300s (high stars).
    # Panel B: 10 frames at 300s (high stars), 10 frames at 60s (low stars).
    # Predicted-effective on B with _DEFAULT_QUALITY_THRESHOLD=0.0 keeps
    # everything (sum = 3600s). Peer max = 18000. New per-frame default
    # for the COMPLETION on B should be the median over ALL captured
    # EXPOSUREs in B's stack, not over keepers — that's max(60, 300) by
    # count-1 in the existing data ordering. With 10×300 + 10×60 the
    # median is 180 (average of the two middle values).
    stack_a = _stack("stack_a", "H", [100] * 60, [300] * 60)
    stack_b = _stack(
        "stack_b", "H",
        num_stars_per_frame=[100] * 10 + [5] * 10,        # low-star frames
        exposure_per_frame_s=[300.0] * 10 + [60.0] * 10,  # mixed cadence
    )
    target_a = _target("t_a", "PanelA", [stack_a.stable_hash])
    target_b = _target("t_b", "PanelB", [stack_b.stable_hash])
    mosaic = _mosaic("m1", [target_a.stable_hash, target_b.stable_hash])
    ctx = _ctx(
        shopping_cart=[("mosaic", mosaic.stable_hash, "PixInsight")],
        mosaics=[mosaic],
        targets=[target_a, target_b],
        stacks=[stack_a, stack_b],
    )

    with patch("suggestions.generator.context", ctx):
        sugs = generate_suggestions(ALL_FILTERS)

    b_completion = next(
        s for s in sugs
        if s.target_id == "t_b" and s.kind == SuggestionKind.COMPLETION
    )
    # statistics.median of [300]*10 + [60]*10 = (60 + 300) / 2 = 180.0,
    # rounded → 180. Validates the cadence is over ALL captured frames.
    assert b_completion.default_exposure_per_frame_s == 180.0


def test_filter_not_in_allowed_set_is_suppressed():
    # Panel A has H + G (moon's out, G is not allowed).
    stack_a_h = _stack("a_h", "H", [100] * 30, [300] * 30)
    stack_a_g = _stack("a_g", "G", [100] * 30, [60] * 30)
    stack_b_h = _stack("b_h", "H", [100] * 10, [300] * 10)
    target_a = _target("t_a", "A", [stack_a_h.stable_hash, stack_a_g.stable_hash])
    target_b = _target("t_b", "B", [stack_b_h.stable_hash])
    mosaic = _mosaic("m1", [target_a.stable_hash, target_b.stable_hash])

    ctx = _ctx(
        shopping_cart=[("mosaic", mosaic.stable_hash, "DownscaledMaster", "PixInsight")],
        mosaics=[mosaic],
        targets=[target_a, target_b],
        stacks=[stack_a_h, stack_a_g, stack_b_h],
    )

    narrowband = frozenset({"H", "S", "O"})
    with patch("suggestions.generator.context", ctx):
        sugs = generate_suggestions(narrowband)

    # Should have no G suggestions (filter suppressed).
    assert all(s.filter != "G" for s in sugs)
    # Panel B should get a COMPLETION in H.
    assert any(s.kind == SuggestionKind.COMPLETION and s.target_id == "t_b" and s.filter == "H" for s in sugs)


def test_mosaic_not_in_shopping_cart_is_ignored():
    stack_a = _stack("a", "H", [100] * 10, [300] * 10)
    stack_b = _stack("b", "H", [100] * 30, [300] * 30)
    target_a = _target("t_a", "A", [stack_a.stable_hash])
    target_b = _target("t_b", "B", [stack_b.stable_hash])
    mosaic = _mosaic("m1", [target_a.stable_hash, target_b.stable_hash])

    ctx = _ctx(
        shopping_cart=[],  # empty cart
        mosaics=[mosaic],
        targets=[target_a, target_b],
        stacks=[stack_a, stack_b],
    )

    with patch("suggestions.generator.context", ctx):
        sugs = generate_suggestions(ALL_FILTERS)

    assert sugs == []
