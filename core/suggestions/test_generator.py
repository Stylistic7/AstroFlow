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
