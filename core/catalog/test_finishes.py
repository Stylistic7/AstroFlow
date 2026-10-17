"""Binding finishes to an object's taps — pure composition over STUBBED
primitives (the per-entity tap filing, the object's filters, disk). These
pin the rules that make `/<pipeline>/<finish>` resolvable: name binding for
composites, ROLE binding for per-filter taps (raw header spellings like
"SII"/"Ha" normalise), joins carrying every band, registration order as
preference, status as the worst required input, and never a trace.
"""
from pathlib import Path

import pytest

from catalog import finishes as cf
from catalog.finishes import ABSENT, LIVE, OUT_OF_DATE
from finishes.framework import Finish


def _finish(name, inputs, optional=(), role=None):
    return Finish(name=name, inputs=tuple(inputs), params=[], wgsl="",
                  fn=None, finish_hash=name + "#", optional=tuple(optional),
                  role=role)


HUBBLE = _finish("Hubble", ("Linear SHO", "Linear RGB Stars"), ("Linear RGB Stars",))
HUBBLE_NB = _finish("Hubble", ("Linear Narrowband Starless", "Linear RGB Stars"), ("Linear RGB Stars",))
RGB = _finish("RGB", ("Linear RGB Starless", "Linear RGB Stars"), ("Linear RGB Stars",))
H_SHO = _finish("H", ("Linear SHO", "Linear RGB Stars"), ("Linear RGB Stars",), role="H")
H_PP = _finish("H", ("Post-Processed", "Linear RGB Stars"), ("Linear RGB Stars",), role="H")
S_PP = _finish("S", ("Post-Processed", "Linear RGB Stars"), ("Linear RGB Stars",), role="S")
MASTER = _finish("Master", ("Master",), role="none")
REGISTRY = [HUBBLE, HUBBLE_NB, RGB, H_SHO, H_PP, S_PP, MASTER]


def _e(display, filter, path="/o/x"):
    return {"display": display, "filter": filter, "path": path}


@pytest.fixture
def world(monkeypatch):
    """`taps[(type, id, pipeline)]`, `filters[(type, id)]`, and per-product
    disk status `status[(display, filter)]` (default: live)."""
    taps, filters, status = {}, {}, {}
    monkeypatch.setattr(cf, "all_finishes", lambda: list(REGISTRY))
    monkeypatch.setattr(cf.queries, "product_taps",
                        lambda t, i, p: taps.get((t, i, p), []))
    monkeypatch.setattr(cf.queries, "filters_for",
                        lambda t, i: filters.get((t, i), []))
    monkeypatch.setattr(cf.queries, "pipelines_for",
                        lambda t, i: sorted({p for (tt, ii, p) in taps if (tt, ii) == (t, i)}))

    def resolve(product):
        st = status.get((product.display, product.filter), LIVE)
        return (None, ABSENT) if st == ABSENT else (Path(f"/served/{product.display}/{product.filter}"), st)
    monkeypatch.setattr(cf.queries, "resolve_product", resolve)
    return taps, filters, status


def _mosaic(world, pipeline="P", narrowband=True):
    taps, filters, _ = world
    entries = [_e("Mosaic Starless", f) for f in "HSORGB"]
    entries += [_e("Linear RGB Starless", "RGB"), _e("Linear RGB Stars", "RGB")]
    if narrowband:
        entries.append(_e("Linear SHO", "RGB"))
    taps[("mosaic", "m", pipeline)] = entries
    filters[("mosaic", "m")] = list("HSORGB")


def test_composite_binds_its_taps_by_name(world):
    _mosaic(world)
    b = cf.resolve_finish("mosaic", "m", "P", "Hubble")
    assert [(i.tap, i.filter) for i in b.inputs] == [("Linear SHO", "RGB"), ("Linear RGB Stars", "RGB")]
    assert b.status == LIVE and b.role is None


def test_channel_finish_reads_the_join_when_its_role_is_present(world):
    _mosaic(world)
    b = cf.resolve_finish("mosaic", "m", "P", "H")
    assert b.finish is H_SHO and b.inputs[0].filter == "RGB"


def test_role_gate_uses_the_objects_real_filters(world):
    """"S" is not offered on an H+O object, even though the narrowband join
    (a sentinel-filter tap) would bind any role."""
    taps, filters, _ = world
    taps[("mosaic", "m", "P")] = [_e("Linear Narrowband Starless", "RGB")]
    filters[("mosaic", "m")] = ["H", "O"]
    assert cf.resolve_finish("mosaic", "m", "P", "S") is None
    assert [b.name for b in cf.finishes_for("mosaic", "m", "P")] == ["Hubble"]


def test_per_filter_taps_bind_by_role_across_spellings(world):
    taps, filters, _ = world
    taps[("stack", "s", "P")] = [_e("Master", "SII"), _e("Post-Processed", "SII")]
    filters[("stack", "s")] = ["SII"]
    b = cf.resolve_finish("stack", "s", "P", "S")
    assert b.finish is S_PP and b.inputs[0].filter == "SII"
    assert cf.resolve_finish("stack", "s", "P", "H") is None
    taps[("stack", "h", "P")] = [_e("Post-Processed", "Ha")]
    filters[("stack", "h")] = ["Ha"]
    assert cf.resolve_finish("stack", "h", "P", "H").finish is H_PP


def test_filterless_master_binds_only_the_master_finish(world):
    taps, filters, _ = world
    taps[("calibration_stack", "d", "P")] = [_e("Master", "none")]
    filters[("calibration_stack", "d")] = ["none"]
    assert [b.name for b in cf.finishes_for("calibration_stack", "d", "P")] == ["Master"]
    taps[("stack", "s", "P")] = [_e("Master", "H"), _e("Post-Processed", "H")]
    filters[("stack", "s")] = ["H"]
    assert [b.name for b in cf.finishes_for("stack", "s", "P")] == ["H"]


def test_registration_order_is_preference_and_disk_breaks_ties(world):
    """The join is preferred; when it is not built yet the next registration
    that IS servable shows; when nothing is, the first traced one is returned
    with status absent."""
    taps, filters, status = world
    taps[("mosaic", "m", "P")] = [_e("Linear SHO", "RGB"), _e("Post-Processed", "H")]
    filters[("mosaic", "m")] = ["H"]
    assert cf.resolve_finish("mosaic", "m", "P", "H").finish is H_SHO
    status[("Linear SHO", "RGB")] = ABSENT
    assert cf.resolve_finish("mosaic", "m", "P", "H").finish is H_PP
    status[("Post-Processed", "H")] = ABSENT
    b = cf.resolve_finish("mosaic", "m", "P", "H")
    assert b.finish is H_SHO and b.status == ABSENT


def test_status_is_the_worst_required_input_and_optional_never_downgrades(world):
    taps, filters, status = world
    _mosaic(world)
    status[("Linear RGB Stars", "RGB")] = ABSENT
    b = cf.resolve_finish("mosaic", "m", "P", "Hubble")
    assert b.status == LIVE and b.inputs[1].path is None and b.inputs[1].optional
    status[("Linear SHO", "RGB")] = OUT_OF_DATE
    assert cf.resolve_finish("mosaic", "m", "P", "Hubble").status == OUT_OF_DATE


def test_ambiguous_binding_is_no_binding(world):
    """A composite declaring a per-filter tap would match several entries —
    it does not bind rather than picking one silently."""
    taps, filters, _ = world
    taps[("target", "t", "P")] = [_e("Channel", f) for f in "HSO"]
    filters[("target", "t")] = list("HSO")
    composite = _finish("X", ("Channel",))
    assert cf.bind_finish("target", "t", "P", composite) is None
    assert cf.bind_finish("target", "t", "P", H_PP) is None   # no Post-Processed traced


def test_finishes_for_dedupes_by_name_in_registry_order(world):
    _mosaic(world)
    assert [b.name for b in cf.finishes_for("mosaic", "m", "P")] == ["Hubble", "RGB", "H"]
    assert cf.finishes_for("mosaic", "m", "Q") == []


def test_default_prefers_registry_order_then_pipeline_rank(world, monkeypatch):
    taps, filters, status = world
    _mosaic(world, pipeline="PixInsight Master", narrowband=False)
    _mosaic(world, pipeline="Hybrid: PI Master + OSS Mosaic")
    b = cf.default_finish_for("mosaic", "m")
    assert (b.pipeline, b.name) == ("Hybrid: PI Master + OSS Mosaic", "Hubble")
    status[("Linear SHO", "RGB")] = ABSENT
    b = cf.default_finish_for("mosaic", "m")
    assert (b.pipeline, b.name) == ("PixInsight Master", "RGB")
    for f in ("Linear RGB Starless", "Linear SHO"):
        status[(f, "RGB")] = ABSENT
    for f in "HSORGB":
        status[("Mosaic Starless", f)] = ABSENT
    assert cf.default_finish_for("mosaic", "m") is None


def test_band_roles():
    assert cf.band_roles(1, {}, "Ha") == ["H"]
    assert cf.band_roles(1, {}, "none") == ["none"]
    assert cf.band_roles(3, {"COLORSPC": "linear-sho"}, "RGB") == ["S", "H", "O"]
    assert cf.band_roles(3, {"BANDS": ["H", "O"]}, "RGB") == ["S", "H", "O"]
    assert cf.band_roles(3, {}, "RGB") == ["R", "G", "B"]


def test_never_traces(world, monkeypatch):
    import dependency_graph.trace_paths as tp

    def _boom(*a, **k):
        raise AssertionError("the catalog must not trace")
    monkeypatch.setattr(tp, "_resolve_pipeline_call", _boom)
    assert cf.finishes_for("mosaic", "nope", "P") == []
    assert cf.default_finish_for("mosaic", "nope") is None


def test_a_target_lists_each_pipelines_masters_even_a_pipeline_that_only_stacked(world, monkeypatch):
    """The target page is one section per pipeline: the masters it made (its
    `Master` taps on the target's stacks) and the finishes applied to them. A
    pipeline that only stacked the frames still has a section."""
    from types import SimpleNamespace
    taps, filters, _ = world
    monkeypatch.setattr(cf.queries, "resolve_entity",
                        lambda t, i: SimpleNamespace(stack_ids=["s_S", "s_H"]) if (t, i) == ("target", "t") else object())
    filters[("target", "t")] = ["H", "S"]
    filters[("stack", "s_H")], filters[("stack", "s_S")] = ["H"], ["S"]
    taps[("target", "t", "P")] = [_e("Linear SHO", "RGB")]
    for sid, f in (("s_H", "H"), ("s_S", "S")):
        taps[("stack", sid, "P")] = [_e("Master", f), _e("Post-Processed", f)]
    taps[("stack", "s_H", "Q")] = [_e("Master", "H")]          # Q only stacked

    m = cf.finishes_matrix("target", "t")
    by_name = {p["name"]: p for p in m["pipelines"]}
    assert sorted(by_name) == ["P", "Q"]
    assert [(x["stack_id"], x["filter"], x["finish"]) for x in by_name["P"]["masters"]] == \
        [("s_H", "H", "H"), ("s_S", "S", "S")]                   # filter order, not stack order
    assert [x["stack_id"] for x in by_name["Q"]["masters"]] == ["s_H"]
    assert by_name["Q"]["finishes"] == []
