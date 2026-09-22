"""`_overlay_durable_state` rewrites each node's state at serve time from
durable signals (on-disk output + the lean `node_outcome` record), and
`_strip_server_internal_fields` removes the server-only baked path.

We replace the module-level `context` with a fake (you can't monkeypatch the
LocalContext instance — its `__setattr__` writes attributes to the cache), so
the test needs no real context/cache/Ray. File presence is real (tmp_path).
"""
from web_server.backend import compute_graphs


class _FakeCtx:
    """Stand-in for `compute_graphs.context`; records which keys were read."""
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.reads = []

    def get(self, key, default=None):
        self.reads.append(key)
        return self.outcomes.get(key, default)

    def __getitem__(self, key):
        return self.get(key)


def _build(tmp_path, monkeypatch):
    present = tmp_path / "present.fits"
    present.write_text("x")
    missing = str(tmp_path / "missing.fits")

    ctx = _FakeCtx({
        ("node_outcome", "reclaimed"): {"result_type": "ImageMetadata", "failed": False},
        ("node_outcome", "failed"): {"result_type": "ValueError", "failed": True},
        ("node_outcome", "skipped"): {"result_type": "SkippedType", "failed": False},
        # "neverrun" intentionally absent -> get() returns None
    })
    monkeypatch.setattr(compute_graphs, "context", ctx)

    st = {
        "done@root": {"displayName": "stack_files_oss", "state": "NO_STATE_RUNTIME",
                      "options": {"expectedPath": str(present)}},
        "reclaimed@root": {"displayName": "place_image", "state": "NO_STATE_RUNTIME",
                           "options": {"expectedPath": missing}},
        "failed@root": {"displayName": "bxt_correct", "state": "NO_STATE_RUNTIME",
                        "options": {"expectedPath": missing}},
        "skipped@root": {"displayName": "sxt_stars", "state": "NO_STATE_RUNTIME",
                         "options": {"expectedPath": missing}},
        "neverrun@root": {"displayName": "hubble_image", "state": "NO_STATE_RUNTIME",
                          "options": {"expectedPath": missing}},
    }
    # A group over the reclaimed + never-run leaves (same dict objects, as the
    # real gather_state_from_visual builds groupMeta).
    st["grp"] = {"displayName": "color", "state": "NO_STATE_RUNTIME",
                 "groupMeta": {"reclaimed@root": st["reclaimed@root"],
                               "neverrun@root": st["neverrun@root"]}}
    return {"skeleton": [], "state": st}, ctx


def test_states_derived_from_disk_and_outcome(tmp_path, monkeypatch):
    snap, _ = _build(tmp_path, monkeypatch)
    compute_graphs._overlay_durable_state(snap)
    st = snap["state"]
    assert st["done@root"]["state"] == "SUCCEEDED"            # file present
    assert st["failed@root"]["state"] == "FAILED"
    assert st["skipped@root"]["state"] == "SKIPPED"
    assert st["neverrun@root"]["state"] == "NO_STATE_RUNTIME"
    # reclaimed: SUCCEEDED + the component's sanctioned "deleted" treatment
    r = st["reclaimed@root"]
    assert r["state"] == "SUCCEEDED"
    assert r["modifiers"] == ["deleted"]
    assert r["callout"] == "reclaimed"
    assert "reclaimed" in r["stateTooltip"]
    # group rolls up: reclaimed (done-like) + never-run -> partial -> RUNNING
    assert st["grp"]["state"] == "RUNNING"


def test_reads_only_node_outcome_and_skips_present_nodes(tmp_path, monkeypatch):
    snap, ctx = _build(tmp_path, monkeypatch)
    compute_graphs._overlay_durable_state(snap)
    # never touches the heavy node_status (no logs loaded)
    assert ctx.reads and all(k[0] == "node_outcome" for k in ctx.reads)
    # the present-file node short-circuits before any lookup
    assert ("node_outcome", "done") not in ctx.reads
    # each canonical id read at most once (per-request memo)
    assert len(ctx.reads) == len(set(ctx.reads))


def test_present_node_clears_stale_reclaimed_decoration(tmp_path, monkeypatch):
    # A reused snapshot where a now-present node was reclaimed last render:
    # the overlay must strip the stale `deleted` decoration.
    snap, _ = _build(tmp_path, monkeypatch)
    snap["state"]["done@root"].update(
        modifiers=["deleted"], callout="reclaimed", stateTooltip="old")
    compute_graphs._overlay_durable_state(snap)
    d = snap["state"]["done@root"]
    assert d["state"] == "SUCCEEDED"
    assert "modifiers" not in d and "callout" not in d and "stateTooltip" not in d


def test_strip_removes_expected_path(tmp_path, monkeypatch):
    snap, _ = _build(tmp_path, monkeypatch)
    compute_graphs._overlay_durable_state(snap)
    out = compute_graphs._strip_server_internal_fields(snap)
    for meta in out["state"].values():
        assert "expectedPath" not in (meta.get("options") or {})


def test_slice_reconstructs_shared_group_definition():
    """A shared group (fan-in from >1 filter's merge) bakes its `definition`
    onto only one skeleton copy; the others are bare stubs. If the slice prunes
    the definition-bearing copy (its parent is a hidden filter), the surviving
    stub must still be expandable — reconstructed + sliced to visible children.
    """
    snap = {
        "skeleton": [
            {"id": "mergeS", "type": "execution", "next": [
                {"id": "color", "type": "group", "definition": [   # full copy (under S)
                    {"id": "bxtS", "type": "execution"},
                    {"id": "bxtH", "type": "execution"},
                    {"id": "hubble", "type": "execution"},
                ]},
            ]},
            {"id": "mergeH", "type": "execution", "next": [
                {"id": "color", "type": "group"},                  # bare stub (under H)
            ]},
        ],
        "state": {
            "mergeS": {"displayName": "merge", "filters": ["S"]},
            "mergeH": {"displayName": "merge", "filters": ["H"]},
            "color":  {"displayName": "color_post_process", "filters": []},
            "bxtS":   {"displayName": "bxt_correct", "filters": ["S"]},
            "bxtH":   {"displayName": "bxt_correct", "filters": ["H"]},
            "hubble": {"displayName": "hubble_image", "filters": []},
        },
    }
    out = compute_graphs._slice_workflow_graph_by_filter(snap, "H")
    ids = [e["id"] for e in out["skeleton"]]
    assert "mergeH" in ids and "mergeS" not in ids       # S-only parent pruned

    color = out["skeleton"][ids.index("mergeH")]["next"][0]
    assert color["id"] == "color"
    # the fix: the formerly-bare surviving group is expandable again...
    assert isinstance(color.get("definition"), list), "sliced group must expand"
    # ...pruned to visible children: H's bxt + the shared deliverable, not S's
    defids = {c["id"] for c in color["definition"]}
    assert defids == {"bxtH", "hubble"}, defids


def test_every_hierarchy_level_strips_server_fields(tmp_path, monkeypatch):
    """The stack and target routes returned the RAW snapshot — `filters` and
    `expectedPath` included — and the frontend's DagNode constructor rejects
    unknown keys; every target page in the cart errored (2026-09-20). One
    preparation for all three levels."""
    snap, ctx = _build(tmp_path, monkeypatch)
    for meta in snap["state"].values():
        meta["filters"] = ["H"]
    ctx.outcomes["workflow_graphs"] = {("target", "t1", "P"): snap}
    monkeypatch.setattr(compute_graphs, "_resolve_ref", lambda ref: ref)
    for level, oid in (("target", "t1"), ("stack", "s1")):
        ctx.outcomes["workflow_graphs"] = {(level, oid, "P"): snap}
        out = compute_graphs._workflow_graph(level, oid, "P")
        assert set(out) == {"skeleton", "state"} and out["state"]
        for meta in out["state"].values():
            assert "filters" not in meta and "expectedPath" not in (meta.get("options") or {})
        assert out["state"]["done@root"]["state"] == "SUCCEEDED", "durable overlay must have run"
    assert compute_graphs._workflow_graph("target", "t1", None) == compute_graphs._EMPTY_WORKFLOW_GRAPH
