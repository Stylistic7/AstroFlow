"""Unit tests for the catalog queries — pure composition over STUBBED primitives
(registry, the per-entity tap filing, disk). No Ray, no real context. These pin
the contract for the edge cases that motivated the module: the one "viewable"
existence definition, the per-entity product lookup, filter union + ordering +
the missing-stack None-guard. (Finishes over these taps: `test_finishes.py`.)
"""
import pytest

from catalog import queries as q
from products.product import Product


# ── fakes ─────────────────────────────────────────────────────────────────────
class FakeStack: pass
class FakeTarget: pass
class FakeMosaic: pass


class _Ref:
    """ContextRef-like: `.get()` returns the entity."""
    def __init__(self, entity): self._e = entity
    def get(self): return self._e


class _Entity:
    def __init__(self, **kw): self.__dict__.update(kw)


class _FakePath:
    def __init__(self, s, exists=False): self._s, self._exists = s, exists
    def __str__(self): return self._s
    def exists(self): return self._exists


class _Store(dict):
    """The durable store as `product_taps` sees it: `context.get(key)`."""
    def get(self, key, default=None):
        return dict.get(self, key, default)


@pytest.fixture
def fake_context(monkeypatch):
    ctx = _Entity(stacks={}, targets={}, mosaics={}, calibration_stacks={})
    monkeypatch.setattr(q, "context", ctx)
    monkeypatch.setattr(q, "_MOSAIC_FILTERS_MEMO", {})
    monkeypatch.setattr(q, "resolve_entity",
                        lambda t, i: _resolve(ctx, t, i))
    return ctx


def _resolve(ctx, object_type, object_id):
    if object_type == "calibration_stack":
        return ctx.calibration_stacks.get(object_id)
    table = {"stack": ctx.stacks, "target": ctx.targets, "mosaic": ctx.mosaics}
    ref = table.get(object_type, {}).get(object_id)
    return ref.get() if ref is not None else None


@pytest.fixture
def taps(monkeypatch, fake_context):
    """Register taps for `(object_type, object_id, pipeline)` the way the graph
    build files them, and make `registered_pipelines` know the names."""
    store = _Store()
    names = set()
    fake_context.get = store.get
    monkeypatch.setattr(q, "product_taps_key",
                        lambda t, i, c, p: ("taps", t, i, c, p))
    monkeypatch.setattr(q, "registered_pipelines",
                        lambda: {(n, FakeStack): 1 for n in names})

    def add(object_type, object_id, pipeline, entries, content_hash="c"):
        names.add(pipeline)
        store[("taps", object_type, object_id, content_hash, pipeline)] = entries
    return add


def _prod(**kw):
    base = dict(object_type="stack", object_id="s1",
               pipeline="PixInsight Master", display="Master", filter="H")
    base.update(kw)
    return Product(**base)


def test_product_is_frozen_and_hashable():
    p = _prod()
    assert p in {p}
    with pytest.raises(Exception):
        p.pipeline = "x"


# ── product_taps: ONE read, keyed by the entity's own identity ───────────────
def test_product_taps_reads_the_entity_key(fake_context, taps):
    fake_context.stacks = {"s1": _Ref(_Entity(content_hash="c"))}
    taps("stack", "s1", "PixInsight Master",
         [{"display": "Master", "filter": "H", "path": "/o/a"}])
    assert q.product_taps("stack", "s1", "PixInsight Master") == [
        {"display": "Master", "filter": "H", "path": "/o/a"}]


def test_product_taps_misses_on_a_stale_content_hash(fake_context, taps):
    """New frames move the entity's content hash; the filing under the old
    one is an orphan, never a stale answer."""
    fake_context.stacks = {"s1": _Ref(_Entity(content_hash="NEW"))}
    taps("stack", "s1", "P", [{"display": "Master", "filter": "H", "path": "/o/a"}],
         content_hash="OLD")
    assert q.product_taps("stack", "s1", "P") == []


def test_product_taps_unknown_entity_is_empty(fake_context, taps):
    assert q.product_taps("stack", "nope", "P") == []


# ── pipelines_for = the pipelines that traced a tap to THIS object ───────────
def test_pipelines_for_is_whoever_routed_a_tap_here(fake_context, taps):
    """A calibration stack is never a cart item and no pipeline `processes=`
    its class — yet the mosaic pipeline that built its master filed a tap
    under it, so that pipeline is the one on its page."""
    fake_context.calibration_stacks = {"f1": _Entity(content_hash="c")}
    fake_context.stacks = {"s1": _Ref(_Entity(content_hash="c"))}
    taps("calibration_stack", "f1", "WBPP + GraXpert + OSS Mosaic",
         [{"display": "Star Flat", "filter": "S", "path": "/o/flat"}])
    taps("stack", "s1", "PixInsight Master",
         [{"display": "Master", "filter": "H", "path": "/o/m"}])
    assert q.pipelines_for("calibration_stack", "f1") == [
        "WBPP + GraXpert + OSS Mosaic"]
    assert q.pipelines_for("stack", "s1") == ["PixInsight Master"]


def test_pipelines_for_unknown_entity_is_empty(fake_context, taps):
    taps("stack", "s1", "P", [{"display": "Master", "filter": "H", "path": "/o"}])
    assert q.pipelines_for("bogus", "x") == []


# ── predict_path picks the (display, filter) tap ─────────────────────────────
def test_predict_path_matches_display_and_filter(fake_context, taps):
    fake_context.stacks = {"s1": _Ref(_Entity(content_hash="c"))}
    taps("stack", "s1", "PixInsight Master", [
        {"display": "Master", "filter": "H", "path": "/out/m.xisf"},
        {"display": "Post-Processed", "filter": "H", "path": "/out/pp.xisf"},
    ])
    assert str(q.predict_path(_prod())) == "/out/m.xisf"
    assert q.predict_path(_prod(display="Nope")) is None
    assert q.predict_path(_prod(filter="S")) is None


def test_predict_path_compares_filters_as_strings(fake_context, taps):
    """A filterless tap (`None`) is addressed by the URL as "None"."""
    fake_context.calibration_stacks = {"d1": _Entity(content_hash="c")}
    taps("calibration_stack", "d1", "P",
         [{"display": "Master", "filter": None, "path": "/out/dark"}])
    assert str(q.predict_path(Product("calibration_stack", "d1", "P",
                                      "Master", "None"))) == "/out/dark"


def test_master_path_is_the_primary_only_when_on_disk(monkeypatch):
    monkeypatch.setattr(q, "predict_path", lambda p: _FakePath("/o/m.xisf"))
    monkeypatch.setattr(q.os.path, "exists", lambda p: True)
    assert str(q.master_path(_prod())) == "/o/m.xisf"
    monkeypatch.setattr(q.os.path, "exists", lambda p: False)
    assert q.master_path(_prod()) is None
    monkeypatch.setattr(q, "predict_path", lambda p: None)
    assert q.master_path(_prod()) is None


# ── the ONE "exists = viewable" definition (pyramid-or-master) ───────────────
def test_viewable_prefers_pyramid_even_when_master_reclaimed(monkeypatch):
    monkeypatch.setattr(q, "predict_path", lambda p: _FakePath("/o/m.xisf"))
    monkeypatch.setattr(q, "pyramid_path_for",
                        lambda m: _FakePath("/o/m_tiles.tiff", exists=True))
    monkeypatch.setattr(q.os.path, "exists", lambda p: False)  # primary gone
    assert str(q.resolve_product(_prod())[0]) == "/o/m_tiles.tiff"


def test_viewable_falls_back_to_display_ready_master_tiff(monkeypatch):
    """A uint8 Subframe preview IS its own display artifact — no pyramid
    needed, serves directly."""
    monkeypatch.setattr(q, "predict_path", lambda p: _FakePath("/o/m.tiff"))
    monkeypatch.setattr(q, "pyramid_path_for",
                        lambda m: _FakePath("/o/m_tiles.tiff", exists=False))
    monkeypatch.setattr(q, "needs_display_pyramid", lambda m: False)
    monkeypatch.setattr(q.os.path, "exists", lambda p: True)
    assert str(q.resolve_product(_prod())[0]) == "/o/m.tiff"


def test_viewable_refuses_a_float_master_with_no_pyramid(monkeypatch):
    """The 2026-08-14 regression: a float master is ONE page and strip-based,
    so serving it as tiles yields full-res at every zoom. Not viewable until
    its pyramid exists."""
    monkeypatch.setattr(q, "predict_path", lambda p: _FakePath("/o/m.tiff"))
    monkeypatch.setattr(q, "pyramid_path_for",
                        lambda m: _FakePath("/o/m_tiles.tiff", exists=False))
    monkeypatch.setattr(q, "needs_display_pyramid", lambda m: True)
    monkeypatch.setattr(q.os.path, "exists", lambda p: True)
    assert q.resolve_product(_prod())[0] is None


def test_viewable_none_when_xisf_master_and_no_pyramid(monkeypatch):
    monkeypatch.setattr(q, "predict_path", lambda p: _FakePath("/o/m.xisf"))
    monkeypatch.setattr(q, "pyramid_path_for",
                        lambda m: _FakePath("/o/m_tiles.tiff", exists=False))
    monkeypatch.setattr(q.os.path, "exists", lambda p: False)
    assert q.resolve_product(_prod()) == (None, "absent")


def test_viewable_none_when_no_predicted_path(monkeypatch):
    monkeypatch.setattr(q, "predict_path", lambda p: None)
    assert q.resolve_product(_prod())[0] is None


# ── filters_for: union / ordering / None-guard / calibration ─────────────────
def test_filters_stack_is_its_own_filter(fake_context):
    fake_context.stacks = {"s1": _Ref(_Entity(filter="H"))}
    assert q.filters_for("stack", "s1") == ["H"]


def test_filters_calibration_none_is_empty(fake_context):
    fake_context.calibration_stacks = {"c1": _Entity(filter=None)}
    assert q.filters_for("calibration_stack", "c1") == []


def test_filters_target_union_ordered(fake_context):
    fake_context.stacks = {"s_b": _Ref(_Entity(filter="B")),
                           "s_h": _Ref(_Entity(filter="H"))}
    fake_context.targets = {"t1": _Ref(_Entity(stack_ids=["s_b", "s_h"]))}
    assert q.filters_for("target", "t1") == ["H", "B"]  # L,H,S,O,R,G,B order


def test_filters_mosaic_skips_missing_stack_ref(fake_context):
    # The None-guard the backend has and the frontend copies dropped: a missing
    # stack ref is SKIPPED, not a crash.
    fake_context.stacks = {"s_h": _Ref(_Entity(filter="H"))}  # s_o absent
    fake_context.targets = {"t1": _Ref(_Entity(stack_ids=["s_h", "s_o"]))}
    fake_context.mosaics = {"m1": _Ref(_Entity(stable_hash="m1",
                                               target_ids=["t1"]))}
    assert q.filters_for("mosaic", "m1") == ["H"]


def test_filters_unknown_entity_is_empty(fake_context):
    assert q.filters_for("stack", "nope") == []


# ── viewable_from_master + snapshot-only reads (the perf-critical additions) ──
def test_viewable_from_master_no_retrace(monkeypatch):
    monkeypatch.setattr(q, "pyramid_path_for",
                        lambda m: _FakePath(str(m) + "_tiles.tiff", exists=True))
    assert str(q.viewable_from_master("/o/m.xisf")).endswith("_tiles.tiff")
    monkeypatch.setattr(q, "pyramid_path_for",
                        lambda m: _FakePath("none", exists=False))
    monkeypatch.setattr(q, "needs_display_pyramid", lambda m: False)
    monkeypatch.setattr(q.os.path, "exists", lambda p: True)
    assert str(q.viewable_from_master("/o/m.tiff")) == "/o/m.tiff"
    assert q.viewable_from_master(None) is None


def test_pipelines_for_never_traces(fake_context, taps, monkeypatch):
    """A web request must never trigger a live trace: the catalog reads the
    per-entity filing only, and a miss is an empty answer, not a trace."""
    import dependency_graph.trace_paths as tp

    def _boom(*a, **k):
        raise AssertionError("the catalog must not trace")
    monkeypatch.setattr(tp, "_resolve_pipeline_call", _boom)
    fake_context.stacks = {"s1": _Ref(_Entity(content_hash="c"))}
    taps("stack", "s1", "P", [])
    assert q.pipelines_for("stack", "s1") == []


def test_object_class_is_the_url_word_to_class_map():
    from products.product import object_class
    from mosaics.types import Mosaic
    from stacks.composite_stack import Stack
    from targets.types import Target

    assert object_class("mosaic") is Mosaic
    assert object_class("stack") is Stack
    assert object_class("target") is Target
    assert object_class("calibration_stack") is None   # nothing processes one
    assert object_class("nebula") is None
