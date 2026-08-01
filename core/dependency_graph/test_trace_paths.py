"""Unit tests for trace-time content_hash propagation.

The pure-Merkle invariant requires that the tracer's `_TracedOutput`
proxy hashes to its `content_hash` field (= the upstream's `call_hash`)
through the same `_maybe_use_content_hash` path the runtime uses for
real upstream `Path`/`ImageMetadata` values. If that fails, every
downstream call_hash diverges from runtime and the sweep produces
wrong filenames.
"""
from __future__ import annotations

from dependency_graph.trace_paths import _TracedOutput
from hashing import hash_call
from hashing.hashing import _maybe_use_content_hash


def test_traced_output_hashes_via_content_hash_attribute():
    """`_TracedOutput` must take the `.content_hash` attribute branch
    of `_maybe_use_content_hash`, NOT the dataclass-fields branch."""
    proxy = _TracedOutput(content_hash="abc123")

    result = _maybe_use_content_hash(proxy)

    # Returns the content_hash string directly, not a (name, value)
    # tuple from make_hashable's dataclass branch.
    assert result == "abc123"


def test_traced_output_is_not_a_contenthashable():
    """`_TracedOutput` must NOT be a `ContentHashable` subclass —
    otherwise `_maybe_use_content_hash` would call `get_content_hash`
    on it (which would try to read a file that doesn't exist)."""
    from file_io.types import ContentHashable

    proxy = _TracedOutput(content_hash="abc")
    assert not isinstance(proxy, ContentHashable)


def test_proxy_produces_same_call_hash_as_raw_string():
    """A downstream filename_fn hashing `_TracedOutput(content_hash=x)`
    must produce the same call_hash as hashing the raw string `x`.

    Equivalent to: trace-time and runtime-time hashing converge on
    the same downstream filename because both routes end up feeding
    the same string into `hash_list`.
    """
    proxy = _TracedOutput(content_hash="upstream_call_hash_xyz")
    fn_hash = "fnhash_abc"

    h_via_proxy = hash_call(fn_hash, "literal_arg", proxy)
    h_via_string = hash_call(fn_hash, "literal_arg", "upstream_call_hash_xyz")

    assert h_via_proxy == h_via_string


# --- trace/tap-snapshot cache key is versioned by the deployed code SHA ---


def test_pipeline_source_version_changes_with_code_version(monkeypatch):
    """The trace + tap-snapshot version component must move when the deployed
    code version changes. That is the whole mechanism: editing an op bumps the
    repo SHA on the next deploy, which moves this component, which makes the
    cached `trace_pipeline` / `build_main_processing_graph` / tap snapshot MISS
    and re-derive with the op's new call_hash. Without it the cached trace keeps
    the op's old call_hash and nothing rebuilds."""
    from dependency_graph.trace_paths import pipeline_source_version

    def fn():
        return 1

    monkeypatch.setattr("state.code_version.deployed_code_version", lambda: "SHA_A")
    va = pipeline_source_version(fn)
    monkeypatch.setattr("state.code_version.deployed_code_version", lambda: "SHA_B")
    vb = pipeline_source_version(fn)
    assert va != vb, "a code-version change must move the trace/tap version key"


def test_pipeline_source_version_empty_is_source_only(monkeypatch):
    """Empty code version (dev shells, tests — no frozen checkout) degrades to a
    pure pipeline-source hash, identical to the pre-change key. So the added
    versioning never spuriously re-derives in environments with no deploy."""
    from dependency_graph.trace_paths import pipeline_source_version
    from hashing import hash_source_code, hash_string

    def fn():
        return 1

    monkeypatch.setattr("state.code_version.deployed_code_version", lambda: "")
    assert pipeline_source_version(fn) == hash_string(hash_source_code(fn) + "")


def test_tap_snapshot_writer_and_reader_share_version_helper():
    """The writer (`compute_graph._store_tap_snapshot`) and the reader
    (`trace_paths._snapshot_key`) MUST compute the snapshot key through the SAME
    `pipeline_source_version` helper — otherwise the dashboard's key never
    matches the one the build wrote, and every tap silently misses. Guards the
    writer/reader lockstep at the source level."""
    import inspect
    import compute_graph.compute_graph as cg
    from dependency_graph import trace_paths as tp

    assert "pipeline_source_version(" in inspect.getsource(cg._store_tap_snapshot)
    assert "pipeline_source_version(" in inspect.getsource(tp._snapshot_key)


# --- display/filter tagging of the walk (the output_index replacement) ---


class _FakeEntity:
    """Minimal identity-bearing, filter-carrying stand-in for a Stack:
    `_find_identity_arg` keys off `stable_hash`, `filter_of` off `filter`."""

    def __init__(self, stable_hash: str, filter: str):
        self.stable_hash = stable_hash
        self.filter = filter


class _FakeTask:
    """Stand-in for a `@node`/`@makes_image` task. `_function` is an
    INSTANCE attribute (not a class attribute) so it isn't auto-bound by
    the descriptor protocol — `inspect.signature` then sees the real
    params, exactly as the framework's wrappers expose them."""

    def __init__(self, fn, display=None, final=False):
        self._function = fn
        self._display = display
        self._final = final
        self._path = f"/out/{fn.__name__}/deadbeefcafef00d.fits"

    def filename_fn(self, *args):
        return self._path


def test_walk_tags_tap_node_with_display_and_filter():
    """The display/filter extension of the walk: a `@makes_image(display=)`
    node surfaces with its `display`+`filter` (and `is_final`), while a plain
    filename_fn helper with no `display=` gets `display=None`/`filter=None`.

    This is the trace-derived data the dashboard now reads instead of the
    deleted stateful PipelineOutputIndex."""
    from dependency_graph.classes import Node
    from dependency_graph.trace_paths import _walk_and_resolve_filenames_tagged

    entity = _FakeEntity("st1", "Ha")

    def tapop(stack): ...     # one positional param → leaf, no deps appended
    def helper(stack): ...

    tap = Node(_FakeTask(tapop, display="Master", final=True), entity, name="tap")
    plain = Node(_FakeTask(helper), entity, name="plain")
    graph = {tap: [], plain: []}

    by_op = {e["op_name"]: e for e in _walk_and_resolve_filenames_tagged(graph)}

    assert by_op["tapop"]["display"] == "Master"
    assert by_op["tapop"]["filter"] == "Ha"
    assert by_op["tapop"]["is_final"] is True
    # Non-display helper is still walked (its path feeds downstream) but
    # carries no tap identity.
    assert by_op["helper"]["display"] is None
    assert by_op["helper"]["filter"] is None
    assert by_op["helper"]["is_final"] is False
    # The walk also carries the tap's routing entity, so the dashboard can
    # scope taps to the queried object (see test_routes_to_item_*).
    assert by_op["tapop"]["object_type"] == "_FakeEntity"
    assert by_op["tapop"]["object_id"] == "st1"


def test_routes_to_item_scopes_taps_by_type_AND_id():
    """A mosaic's trace contains per-panel `Master` taps that route to STACKS
    and per-target `Channel` taps that route to TARGETS; only the taps routing
    to the queried `(object_type, object_id)` belong on that entity's page.

    The match must check BOTH type and id: first-subframe-based `stable_hash`
    collides across a single-panel mosaic / its target / its stack, so id alone
    would wrongly surface a stack's `Master` on the mosaic page. This was the
    `/mosaic/<id>` regression (450 mixed taps showing instead of the mosaic's
    own)."""
    from dependency_graph.trace_paths import _routes_to_item

    mid = "75a1da6463e118e6"
    # The mosaic's own deliverable — kept.
    assert _routes_to_item({"object_type": "Mosaic", "object_id": mid}, "mosaic", mid)
    # A per-panel Master routing to a different stack — excluded.
    assert not _routes_to_item({"object_type": "Stack", "object_id": "other"}, "mosaic", mid)
    # The collision case: a Stack / Target SHARING the mosaic's id — excluded
    # by TYPE (id alone would have matched).
    assert not _routes_to_item({"object_type": "Stack", "object_id": mid}, "mosaic", mid)
    assert not _routes_to_item({"object_type": "Target", "object_id": mid}, "mosaic", mid)
    # snake_case URL/cart type vs PascalCase routing class name.
    assert _routes_to_item(
        {"object_type": "CalibrationStack", "object_id": "z"}, "calibration_stack", "z"
    )
    # A non-tap entry (no routing entity) is never claimed by any page.
    assert not _routes_to_item({"object_type": None, "object_id": None}, "mosaic", mid)


def test_tap_outputs_for_scopes_to_queried_entity(monkeypatch):
    """End-to-end guard for the `/mosaic/<id>` regression: `tap_outputs_for`
    (the helper the dashboard endpoint calls) must return ONLY the taps routing
    to the queried `(object_type, object_id)`. A mosaic's trace also contains
    per-panel `Master` taps (route to stacks) and per-target `Channel` taps
    (route to targets) — those belong on the stack/target pages, not here.

    The existing exhaustive endpoint test checked tap *structure* but not this
    scoping, which is why the regression shipped — so this is a focused unit
    test of the actual function, including the stable_hash collision."""
    import dependency_graph.trace_paths as tp

    mid = "m1"
    entries = [
        {"display": "Mosaic", "filter": "H", "object_type": "Mosaic",
         "object_id": mid, "path": "/out/mosaic.fits"},
        {"display": "Master", "filter": "H", "object_type": "Stack",
         "object_id": mid, "path": "/out/master.fits"},      # collides on id!
        {"display": "Channel", "filter": "H", "object_type": "Target",
         "object_id": "t9", "path": "/out/channel.fits"},
        {"display": None, "filter": None, "object_type": None,
         "object_id": None, "path": "/out/place.tiff"},      # non-tap intermediate
    ]
    monkeypatch.setattr(tp, "traced_paths_for",
                        lambda item, recompute=True, ancestors=(): entries)

    # Queried as the mosaic → only the Mosaic tap (the colliding Stack Master
    # is excluded by TYPE despite sharing the id).
    out = tp.tap_outputs_for(("mosaic", mid, "Hybrid: PI Master + OSS Mosaic"))
    assert [e["display"] for e in out] == ["Mosaic"]

    # Same trace, queried as the stack → the stack's Master (Mosaic excluded).
    out_stack = tp.tap_outputs_for(("stack", mid, "Hybrid: PI Master + OSS Mosaic"))
    assert [e["display"] for e in out_stack] == ["Master"]


def test_tap_path_for_scopes_to_queried_entity(monkeypatch):
    """`tap_path_for` resolves a tap's path only within the queried entity, so a
    colliding Stack `Master` is never served as the mosaic's path."""
    import dependency_graph.trace_paths as tp

    mid = "m1"
    entries = [
        {"display": "Mosaic", "filter": "H", "object_type": "Mosaic",
         "object_id": mid, "path": "/out/mosaic.fits"},
        {"display": "Master", "filter": "H", "object_type": "Stack",
         "object_id": mid, "path": "/out/master.fits"},
    ]
    monkeypatch.setattr(tp, "traced_paths_for",
                        lambda item, recompute=True, ancestors=(): entries)

    assert str(tp.tap_path_for(("mosaic", mid, "P"), "Mosaic", "H")) == "/out/mosaic.fits"
    # The mosaic owns no "Master" tap — the colliding stack's must not leak in.
    assert tp.tap_path_for(("mosaic", mid, "P"), "Master", "H") is None


def test_traced_paths_for_snapshot_miss_recompute_false_returns_empty(monkeypatch):
    """A read-only caller (viewer / catalog) that misses the snapshot gets `[]`
    and NEVER traces — no snapshot means the object×pipeline was never built, so
    there's nothing on disk to show. Tracing on an HTTP request is the failure
    mode this contract exists to prevent."""
    import dependency_graph.trace_paths as tp

    monkeypatch.setattr(tp, "_read_tap_snapshot", lambda item: None)
    # Reaching the recompute path at all would be the bug — make it explode.
    def _boom(item):
        raise AssertionError("recompute=False must not trace")
    monkeypatch.setattr(tp, "_resolve_pipeline_call", _boom)

    assert tp.traced_paths_for(("stack", "s", "P"), recompute=False) == []


def test_traced_paths_for_snapshot_miss_recompute_true_traces_inline(monkeypatch):
    """A head-driver caller (sweep / archive) that misses the snapshot recomputes
    the trace INLINE — resolve the call, `trace`, walk — with no Ray and no
    `@node` dispatch (the removed second cache)."""
    import dependency_graph.trace_paths as tp
    import dependency_graph.tracing as tracing

    seen = {}
    monkeypatch.setattr(tp, "_read_tap_snapshot", lambda item: None)
    monkeypatch.setattr(tp, "_resolve_pipeline_call",
                        lambda item: ("fn", "entity", "config"))

    def _fake_trace(fn, entity, config):
        seen["traced"] = (fn, entity, config)
        return "GRAPH"
    monkeypatch.setattr(tracing, "trace", _fake_trace)
    monkeypatch.setattr(tp, "_walk_and_resolve_filenames_tagged",
                        lambda graph: [{"path": "/o/x", "graph": graph}])

    result = tp.traced_paths_for(("stack", "s", "P"), recompute=True)

    assert seen["traced"] == ("fn", "entity", "config")   # traced inline
    assert result == [{"path": "/o/x", "graph": "GRAPH"}]  # walked the graph


def test_traced_paths_for_recompute_true_unresolvable_returns_empty(monkeypatch):
    """Inline recompute short-circuits to `[]` when the cart item is unresolvable
    (entity not in context / pipeline unregistered) — no crash, no trace."""
    import dependency_graph.trace_paths as tp

    monkeypatch.setattr(tp, "_read_tap_snapshot", lambda item: None)
    monkeypatch.setattr(tp, "_resolve_pipeline_call",
                        lambda item: (None, None, None))

    assert tp.traced_paths_for(("stack", "gone", "P"), recompute=True) == []


def test_traced_paths_for_falls_back_to_ancestor_snapshot(monkeypatch):
    """A child leaf with no snapshot of its own resolves from a named cart-item
    ANCESTOR's snapshot (the shared-graph trace holding its taps). The URL says
    which ancestor — no scan. The full ancestor list is returned; the caller's
    `_routes_to_item` filter scopes it to the leaf."""
    import dependency_graph.trace_paths as tp

    parent_entries = [
        {"object_type": "Stack", "object_id": "child", "display": "Master"},
        {"object_type": "Target", "object_id": "parent", "display": "Channel"},
    ]

    def _snap(item):
        # Only the TARGET (parent) has a snapshot; the child stack has none.
        return parent_entries if item == ("target", "parent", "P") else None
    monkeypatch.setattr(tp, "_read_tap_snapshot", _snap)

    result = tp.traced_paths_for(
        ("stack", "child", "P"), ancestors=(("target", "parent"),))
    assert result == parent_entries   # whole ancestor trace; caller filters


def test_traced_paths_for_ancestor_miss_then_recompute_false_empty(monkeypatch):
    """No own snapshot and no ancestor snapshot → read-only caller gets `[]`
    (never traces), even with ancestors named."""
    import dependency_graph.trace_paths as tp
    monkeypatch.setattr(tp, "_read_tap_snapshot", lambda item: None)
    assert tp.traced_paths_for(
        ("stack", "child", "P"), recompute=False,
        ancestors=(("target", "parent"),)) == []


def test_tap_outputs_for_scopes_ancestor_trace_to_leaf(monkeypatch):
    """`tap_outputs_for` on a child, given an ancestor, keeps only the display
    taps routing to the LEAF — the sibling target's own taps are dropped."""
    import dependency_graph.trace_paths as tp
    parent_entries = [
        {"object_type": "Stack", "object_id": "child",
         "display": "Master", "filter": "H", "path": "/o/child"},
        {"object_type": "Target", "object_id": "parent",
         "display": "Channel", "filter": "H", "path": "/o/parent"},
    ]
    monkeypatch.setattr(
        tp, "_read_tap_snapshot",
        lambda item: parent_entries if item == ("target", "parent", "P") else None)

    out = tp.tap_outputs_for(
        ("stack", "child", "P"), ancestors=(("target", "parent"),))
    assert [e["path"] for e in out] == ["/o/child"]   # sibling's Channel dropped
