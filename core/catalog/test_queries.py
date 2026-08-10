"""Unit tests for the catalog queries — pure composition over STUBBED primitives
(registry, trace, disk). No Ray, no real context. These pin the contract for the
edge cases that motivated the module: the one "viewable" existence definition,
the class-slice, filter union + ordering + the missing-stack None-guard, and the
preference-ordered default.
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


@pytest.fixture
def classes(monkeypatch):
    monkeypatch.setattr(q, "object_class", lambda t: {
        "stack": FakeStack, "target": FakeTarget, "mosaic": FakeMosaic}.get(t))


@pytest.fixture
def fake_context(monkeypatch):
    ctx = _Entity(stacks={}, targets={}, mosaics={}, calibration_stacks={})
    monkeypatch.setattr(q, "context", ctx)
    monkeypatch.setattr(q, "_MOSAIC_FILTERS_MEMO", {})
    return ctx


def _prod(**kw):
    base = dict(object_type="stack", object_id="s1",
               pipeline="PixInsight Master", display="Master", filter="H")
    base.update(kw)
    return Product(**base)


# ── Product identity ──────────────────────────────────────────────────────────
def test_product_is_frozen_hashable_and_as_view():
    p = _prod()
    assert p.as_view() == {"pipeline": "PixInsight Master",
                           "display": "Master", "filter": "H"}
    assert p in {p}
    with pytest.raises(Exception):
        p.pipeline = "x"


# ── pipelines_for = class-slice, not engagement ──────────────────────────────
def test_pipelines_for_is_the_class_slice(classes, monkeypatch):
    monkeypatch.setattr(q, "registered_pipelines", lambda: {
        ("PixInsight Master", FakeStack): 1,
        ("PixInsight Downscaled", FakeStack): 1,
        ("Hybrid: OSS Stack + OSS Mosaic", FakeMosaic): 1,  # mosaic-only, excluded
    })
    assert q.pipelines_for("stack", "s1") == [
        "PixInsight Downscaled", "PixInsight Master"]


def test_pipelines_for_unknown_type_is_empty(classes):
    assert q.pipelines_for("bogus", "x") == []


# ── predict_path delegates to the trace with the right item ──────────────────
def test_predict_path_delegates_to_trace(monkeypatch):
    seen = {}
    def fake(item, display, filter, ancestors=()):
        seen["args"] = (item, display, filter); return "/out/m.xisf"
    monkeypatch.setattr(q, "tap_path_for", fake)
    assert q.predict_path(_prod()) == "/out/m.xisf"
    assert seen["args"] == (("stack", "s1", "PixInsight Master"), "Master", "H")


# ── the ONE "exists = viewable" definition (pyramid-or-master) ───────────────
def test_viewable_prefers_pyramid_even_when_master_reclaimed(monkeypatch):
    monkeypatch.setattr(q, "predict_path", lambda p, ancestors=(): _FakePath("/o/m.xisf"))
    monkeypatch.setattr(q, "pyramid_path_for",
                        lambda m: _FakePath("/o/m_tiles.tiff", exists=True))
    monkeypatch.setattr(q.os.path, "exists", lambda p: False)  # primary gone
    assert str(q.viewable_path(_prod())) == "/o/m_tiles.tiff"
    assert q.exists(_prod()) is True


def test_viewable_falls_back_to_master_tiff(monkeypatch):
    monkeypatch.setattr(q, "predict_path", lambda p, ancestors=(): _FakePath("/o/m.tiff"))
    monkeypatch.setattr(q, "pyramid_path_for",
                        lambda m: _FakePath("/o/m_tiles.tiff", exists=False))
    monkeypatch.setattr(q.os.path, "exists", lambda p: True)
    assert str(q.viewable_path(_prod())) == "/o/m.tiff"


def test_viewable_none_when_xisf_master_and_no_pyramid(monkeypatch):
    monkeypatch.setattr(q, "predict_path", lambda p, ancestors=(): _FakePath("/o/m.xisf"))
    monkeypatch.setattr(q, "pyramid_path_for",
                        lambda m: _FakePath("/o/m_tiles.tiff", exists=False))
    monkeypatch.setattr(q.os.path, "exists", lambda p: False)
    assert q.viewable_path(_prod()) is None
    assert q.exists(_prod()) is False


def test_viewable_none_when_no_predicted_path(monkeypatch):
    monkeypatch.setattr(q, "predict_path", lambda p, ancestors=(): None)
    assert q.viewable_path(_prod()) is None


# ── products_for crosses pipelines × taps ────────────────────────────────────
def test_products_for_crosses_pipelines_and_taps(classes, monkeypatch):
    monkeypatch.setattr(q, "registered_pipelines",
                        lambda: {("PixInsight Master", FakeStack): 1})
    monkeypatch.setattr(q, "tap_outputs_for", lambda item, recompute=True, ancestors=(): [
        {"display": "Master", "filter": "H", "path": "/o/a"},
        {"display": "Starless", "filter": "H", "path": "/o/b"},
    ])
    assert list(q.products_for("stack", "s1")) == [
        _prod(display="Master"), _prod(display="Starless")]


def test_products_for_passes_sentinel_filter_through(classes, monkeypatch):
    # A colour composite's "RGB" isn't a real object filter — the catalog treats
    # it as just another filter value.
    monkeypatch.setattr(q, "registered_pipelines",
                        lambda: {("PixInsight Master", FakeStack): 1})
    monkeypatch.setattr(q, "tap_outputs_for", lambda item, recompute=True, ancestors=(): [
        {"display": "Stars", "filter": "RGB", "path": "/o/rgb"}])
    got = list(q.products_for("stack", "s1"))
    assert got[0].filter == "RGB"


# ── existing_for keeps only what's viewable ──────────────────────────────────
def test_existing_for_keeps_only_viewable(classes, monkeypatch):
    monkeypatch.setattr(q, "registered_pipelines", lambda: {("P", FakeStack): 1})
    monkeypatch.setattr(q, "tap_outputs_for", lambda item, recompute=True, ancestors=(): [
        {"display": "Master", "filter": "H", "path": "/o/a"},
        {"display": "Gone", "filter": "H", "path": "/o/b"},
    ])
    monkeypatch.setattr(q, "exists", lambda p, ancestors=(): p.display == "Master")
    assert [p.display for p in q.existing_for("stack", "s1")] == ["Master"]


# ── default_product_for: preference + default display + filter pin ───────────
def test_default_prefers_canonical_pipeline_and_display(monkeypatch):
    prods = [
        _prod(pipeline="Hybrid: OSS Stack + OSS Mosaic", display="Master"),
        _prod(pipeline="PixInsight Master", display="Starless"),
        _prod(pipeline="PixInsight Master", display="Master"),
    ]
    monkeypatch.setattr(q, "existing_for", lambda t, i, ancestors=(): iter(prods))
    assert q.default_product_for("stack", "s1") == _prod(
        pipeline="PixInsight Master", display="Master")


def test_default_filter_pin(monkeypatch):
    prods = [
        _prod(object_type="target", object_id="t1", display="Channel", filter="H"),
        _prod(object_type="target", object_id="t1", display="Channel", filter="S"),
    ]
    monkeypatch.setattr(q, "existing_for", lambda t, i, ancestors=(): iter(prods))
    assert q.default_product_for("target", "t1", filter="S").filter == "S"


def test_default_none_when_nothing_exists(monkeypatch):
    monkeypatch.setattr(q, "existing_for", lambda t, i, ancestors=(): iter([]))
    assert q.default_product_for("stack", "s1") is None


def test_default_pipeline_prefers_existing(monkeypatch):
    prods = [
        _prod(pipeline="Hybrid: OSS Stack + OSS Mosaic", display="Master"),
        _prod(pipeline="PixInsight Downscaled", display="Master"),
    ]
    monkeypatch.setattr(q, "existing_for", lambda t, i, ancestors=(): iter(prods))
    # Preference order (not disk order) decides: Downscaled outranks Hybrid-OSS.
    assert q.default_pipeline_for("stack", "s1") == "PixInsight Downscaled"


def test_default_pipeline_falls_back_to_top_preference(monkeypatch):
    # Nothing on disk → the top preference (the caller then finds no file and
    # reports empty); never a scattered "PixInsight Master" literal.
    monkeypatch.setattr(q, "existing_for", lambda t, i, ancestors=(): iter([]))
    assert q.default_pipeline_for("stack", "s1") == q._PIPELINE_PREFERENCE[0]


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
    monkeypatch.setattr(q.os.path, "exists", lambda p: True)
    assert str(q.viewable_from_master("/o/m.tiff")) == "/o/m.tiff"
    assert q.viewable_from_master(None) is None


def test_products_for_reads_snapshot_only(classes, monkeypatch):
    # A web request must never trigger a live trace: products_for reads
    # recompute=False, so a miss skips instead of falling into the @node.
    seen = {}
    monkeypatch.setattr(q, "registered_pipelines", lambda: {("P", FakeStack): 1})

    def fake_tap(item, recompute=True, ancestors=()):
        seen["recompute"] = recompute
        return [{"display": "Master", "filter": "H", "path": "/o/a"}]
    monkeypatch.setattr(q, "tap_outputs_for", fake_tap)
    list(q.products_for("stack", "s1"))
    assert seen["recompute"] is False


def test_object_class_accepts_both_url_grammar_cases():
    """The tap URL grammar carries PascalCase CLASS names ("Mosaic" —
    extract_pipeline_taps does no lowercase translation by design); cart
    items use lowercase. object_class is the single string→class home and
    must accept both: a PascalCase miss here made every /tap/* resolve
    "absent" for mosaics, disabling the graceful N/N-1 fallback exactly
    when it mattered (mid-regeneration "not yet built" pages, 2026-08-10)."""
    from products.product import object_class
    from mosaics.types import Mosaic
    from stacks.composite_stack import Stack
    from targets.types import Target

    for spelling in ("Mosaic", "mosaic"):
        assert object_class(spelling) is Mosaic, spelling
    assert object_class("Stack") is Stack
    assert object_class("Target") is Target
    assert object_class("nebula") is None
