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
