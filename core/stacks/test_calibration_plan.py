"""Unit tests for `calibration_plan` — the consumer-invariant calibration
dep-list builder (the 2026-08-06 re-stack-loop fix).

The property under test: a shared calibration stack's `stack_files` arg list
is a function of THAT stack alone. The old accumulator passed "every master
built so far", so the same dark got a different arg list depending on which
consumer's trace built it — one call_hash per consumer for one physical
stack, and a perpetual re-stack loop as each pass demanded its own variant.

`resolve_calibration` and `stack_files` are monkeypatched — pure tests, no
context, no Ray.
"""
from types import SimpleNamespace

import stacks.calibration_selection as sel
from stacks.types import StackIdDependency


def _stack(sid, deps=()):
    """A fake BaseStack: one run whose dependencies name `deps` stacks."""
    run = SimpleNamespace(dependencies={
        f"d{i}": StackIdDependency(stack_id=d) for i, d in enumerate(deps)
    })
    return SimpleNamespace(stable_hash=sid, contents={"run0": run})


def _wire(monkeypatch, entities):
    """resolve_calibration returns the entity itself (already 'enriched');
    stack_files records (stack_id, arg ids) and returns a master token."""
    calls = []

    def fake_resolve(s, cal_stacks, decisions):
        return s

    def fake_stack_files(s, config, *masters):
        token = f"master:{s.stable_hash}"
        calls.append((s.stable_hash, tuple(masters)))
        return token

    monkeypatch.setattr(sel, "resolve_calibration", fake_resolve)
    monkeypatch.setattr(sel, "stack_files", fake_stack_files)
    config = SimpleNamespace(
        get_dust_epochs=lambda: None,
        get_calibration_stacks=lambda: entities, decisions={})
    return calls, config


def test_cal_stack_args_are_its_own_deps_only(monkeypatch):
    """bias: no masters; flat: exactly its bias; light: dark+flat (sorted),
    never the bias (Landmann §8.2)."""
    bias = _stack("bias1")
    dark = _stack("dark1")
    flat = _stack("flat1", deps=("bias1",))
    light = _stack("light1", deps=("dark1", "flat1"))
    entities = {"bias1": bias, "dark1": dark, "flat1": flat}
    calls, config = _wire(monkeypatch, entities)

    cal_masters = sel.calibrate(light, config)

    by_id = dict(calls)
    assert by_id["bias1"] == ()
    assert by_id["dark1"] == ()
    assert by_id["flat1"] == ("master:bias1",)
    # light's own masters: dark + flat, sorted id order, no bias
    assert cal_masters == ["master:dark1", "master:flat1"]


def test_shared_dark_gets_identical_args_from_every_consumer(monkeypatch):
    """THE regression: two consumers with different flats share a dark; the
    dark's stack_files args must be identical (and empty) in both plans.
    Under the old accumulator, consumer L handed it [flat_L-era masters]
    while consumer B handed it something else — the flap."""
    bias = _stack("bias1")
    dark = _stack("dark_shared")
    flat_l = _stack("flat_L", deps=("bias1",))
    flat_b = _stack("flat_B", deps=("bias1",))
    light_l = _stack("light_L", deps=("flat_L", "dark_shared"))
    light_b = _stack("light_B", deps=("dark_shared", "flat_B"))
    entities = {"bias1": bias, "dark_shared": dark,
                "flat_L": flat_l, "flat_B": flat_b}

    calls_l, config = _wire(monkeypatch, entities)
    sel.calibrate(light_l, config)
    dark_args_from_l = dict(calls_l)["dark_shared"]

    calls_b, config = _wire(monkeypatch, entities)
    sel.calibrate(light_b, config)
    dark_args_from_b = dict(calls_b)["dark_shared"]

    assert dark_args_from_l == dark_args_from_b == ()
    # And the flats each get exactly their own bias — not each other.
    assert dict(calls_l)["flat_L"] == ("master:bias1",)
    assert dict(calls_b)["flat_B"] == ("master:bias1",)


def test_dep_order_is_sorted_not_visit_order(monkeypatch):
    """Arg order must be canonical (sorted ids) regardless of the order the
    consumer's runs iterate their dependencies."""
    dark_a = _stack("dark_a")
    dark_z = _stack("dark_z")
    light1 = _stack("light1", deps=("dark_z", "dark_a"))  # reversed
    light2 = _stack("light2", deps=("dark_a", "dark_z"))
    entities = {"dark_a": dark_a, "dark_z": dark_z}

    calls1, config = _wire(monkeypatch, entities)
    m1 = sel.calibrate(light1, config)
    calls2, config = _wire(monkeypatch, entities)
    m2 = sel.calibrate(light2, config)

    assert m1 == m2 == ["master:dark_a", "master:dark_z"]


def test_plan_steps_are_dependency_ordered(monkeypatch):
    """A step's dep_ids only ever name EARLIER steps (bias before flat)."""
    bias = _stack("bias1")
    flat = _stack("flat1", deps=("bias1",))
    light = _stack("light1", deps=("flat1",))
    entities = {"bias1": bias, "flat1": flat}
    _, config = _wire(monkeypatch, entities)

    steps, top_ids = sel.calibration_plan(light, config)

    seen = set()
    for cs, dep_ids in steps:
        assert all(d in seen for d in dep_ids), (cs.stable_hash, dep_ids)
        seen.add(str(cs.stable_hash))
    assert top_ids == ("flat1",)


def test_a_pipeline_integrator_builds_every_master(monkeypatch):
    """`integrate` replaces WBPP for the whole chain: the Siril pipeline's
    bias, dark and flat masters never touch `stack_files`."""
    entities = {"bias1": _stack("bias1"), "dark1": _stack("dark1"),
                "flat1": _stack("flat1", deps=("bias1",))}
    wbpp_calls, config = _wire(monkeypatch, entities)
    own = []

    def integrate(s, config, *masters):
        own.append((s.stable_hash, masters))
        return f"own:{s.stable_hash}"

    cal = sel.calibrate(_stack("light1", deps=("dark1", "flat1")), config, integrate=integrate)
    assert wbpp_calls == []
    assert dict(own)["flat1"] == ("own:bias1",)
    assert cal == ["own:dark1", "own:flat1"]


def test_only_a_night_sky_flat_carries_dust_models():
    """A star flat already holds the lights' dust; a real flat holds its own.
    Only the mote-free night-sky flat takes the runs' mote models."""
    from types import SimpleNamespace
    from stacks.calibration_selection import carries_motes
    from stacks.sky_flat import SkyFlatStack
    from stacks.star_flat import StarFlatStack
    sky = SkyFlatStack.__new__(SkyFlatStack)
    star = StarFlatStack.__new__(StarFlatStack)
    dark = SimpleNamespace(image_type="dark")
    assert carries_motes([dark, sky])
    assert not carries_motes([dark, star])
    assert not carries_motes([dark, sky, star])
    assert not carries_motes([dark, SimpleNamespace(image_type="flat")])
