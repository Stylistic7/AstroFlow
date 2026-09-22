"""Entity tools over real identified testdata: the identity hierarchy, run
and frame detail, quarantine state, listing, and the product matrix —
without ever tracing."""
import pytest

from decisions.decisions import BinaryChoice, DecisionKey, ObjectType
from mcp_server.server import NotAvailableYet


def _ids(with_real_testdata):
    from context import local_context as lc
    stacks = lc.stacks
    targets = lc.targets
    mosaics = lc.mosaics
    return stacks, targets, mosaics


def test_get_object_walks_mosaic_to_frames(with_real_testdata):
    from mcp_server import entities
    stacks, targets, mosaics = _ids(with_real_testdata)
    mid = next(iter(mosaics))
    m = entities.get_object("mosaic", mid, depth=3)   # mosaic -> targets -> stacks -> runs
    assert m["stable_hash"] == mid and m["n_panels"] == len(m["panels"])
    assert [p["panel"] for p in m["panels"]] == list(range(1, m["n_panels"] + 1))
    panel = m["panels"][0]
    assert panel["target"]["object_type"] == "target"
    stack = panel["target"]["stacks"][0]
    assert stack["object_type"] == "stack" and "runs" in stack
    run = stack["runs"][0]
    assert {"run_id", "n_frames", "quarantined", "oldest_date"} <= run.keys()
    assert "frames" not in run   # depth 2 stops at runs for a mosaic


def test_get_object_stack_lists_runs_and_frames(with_real_testdata):
    from mcp_server import entities
    stacks, _, _ = _ids(with_real_testdata)
    sid = next(iter(stacks))
    s = entities.get_object("stack", sid, depth=2)
    entity = stacks[sid].get()
    assert [r["run_id"] for r in s["runs"]] == list(entity.contents)
    for r in s["runs"]:
        assert r["n_frames"] == len(r["frames"]) and not r["frames_truncated"]
        assert all(f["filename"].endswith(".fit") for f in r["frames"])
        assert "selected_frame_count" in r
    assert entities.get_object("stack", sid, depth=0).get("runs") is None


def test_quarantine_overrides_show_on_runs_and_objects(with_real_testdata):
    from context import local_context as lc
    from decisions.helpers import with_override
    from mcp_server import entities
    stacks, _, _ = _ids(with_real_testdata)
    sid, other = list(stacks)[:2]
    run_id = next(iter(stacks[sid].get().contents))
    lc.settings = with_override(
        lc.settings, DecisionKey.quarantine(ObjectType.STACK, run_id, sid),
        BinaryChoice.EXCLUDE)
    s = entities.get_object("stack", sid, depth=1)
    assert {r["run_id"]: r["quarantined"] for r in s["runs"]}[run_id] is True
    assert entities.get_object("stack", other, depth=1)["quarantined"] is False


def test_get_object_rejects_unknown_ids(with_real_testdata):
    from mcp_server import entities
    with pytest.raises(ValueError, match="no stack"):
        entities.get_object("stack", "0" * 16)


def test_list_objects_filters_sorts_and_caps(test_context):
    from context import local_context as lc
    from mcp_server import entities
    lc.set("ui_stacks", {
        "a" * 16: {"name": "Sadr", "filter": "H", "image_type": "light",
                   "newest_date": "2026-01-02", "filenames": ["x", "y"]},
        "b" * 16: {"name": "Sadr", "filter": "O", "image_type": "light",
                   "newest_date": "2026-01-03", "filenames": ["z"]},
        "c" * 16: {"name": "Flat", "filter": "H", "image_type": "flat",
                   "newest_date": "2025-12-01", "filenames": []},
    })
    out = entities.list_objects("stack", name_contains="sadr")
    assert [r["stable_hash"] for r in out["items"]] == ["b" * 16, "a" * 16]
    assert out["items"][1]["n_frames"] == 2 and "filenames" not in out["items"][1]
    assert entities.list_objects("stack", filter="H", image_type="flat")["count"] == 1
    capped = entities.list_objects("stack", limit=1)
    assert capped["truncated"] and len(capped["items"]) == 1
    with pytest.raises(NotAvailableYet):
        entities.list_objects("mosaic")


def test_products_for_object_never_traces(with_real_testdata, monkeypatch):
    import dependency_graph.trace_paths as tp
    from mcp_server import entities
    stacks, _, _ = _ids(with_real_testdata)
    sid = next(iter(stacks))

    def boom(*a, **k):
        raise AssertionError("a request-path read traced a pipeline")
    monkeypatch.setattr(tp, "traced_paths_for", boom)
    matrix = entities.products_for_object("stack", sid)
    assert matrix["pipelines"] == [] and matrix["default"] is None
    assert matrix["filters"] == [stacks[sid].get().filter]
    with pytest.raises(NotAvailableYet):
        entities.resolve_product("stack", sid)


def test_reads_never_touch_the_ray_context(with_real_testdata, monkeypatch):
    """Every read tool answers from local_context; the Ray proxy is for
    confirmed writes only (a read that attached would push a working-dir
    package to the cluster and register a driver)."""
    import sys
    from mcp_server import entities
    # `import context.context` would bind the package's re-exported singleton,
    # whose __getattr__ IS the Ray proxy — take the module from sys.modules.
    context_module = sys.modules["context.context"]

    def boom(self):
        raise AssertionError("a read tool used the Ray context proxy")
    monkeypatch.setattr(context_module.Context, "_actor", boom)
    stacks, targets, mosaics = _ids(with_real_testdata)
    entities.get_object("mosaic", next(iter(mosaics)), depth=3)
    entities.get_object("target", next(iter(targets)), depth=2)
    entities.products_for_object("mosaic", next(iter(mosaics)))
    entities.products_for_object("target", next(iter(targets)))
