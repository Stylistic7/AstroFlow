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
        self.content_hash = "c-" + stable_hash
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
    # The walk also carries the tap's ROUTING entity — URL type, stable hash
    # and content hash — which is what `_store_product_taps` files it under.
    assert by_op["tapop"]["object_type"] == "_fakeentity"
    assert by_op["tapop"]["object_id"] == "st1"
    assert by_op["tapop"]["content_hash"] == "c-st1"
    assert by_op["helper"]["object_type"] is None


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


# --------------------------------------------------------------------------- #
# Web-facing reads must NEVER live-trace (2026-08-16)                          #
# --------------------------------------------------------------------------- #
#
# `traced_paths_for(recompute=True)` re-executes the pipeline function in the
# CALLER's process. On the driver that is intended (sweep / archive of an item
# outside the current graph). In a web worker it meant
# calibrate -> resolve_calibration -> find_external_master walking calibration
# directories with is_dir(), measured at 31 s per /api/pipeline_outputs request
# on the Heart target (n=325, max 43 s) and 32 s for one thumbnail — while
# TargetPanel refires on every SSE tick, piling ~18 deep. The viewer's reads
# now go through `catalog.product_taps`, which cannot trace at all; this pins
# the driver-side contract from the other side.

def test_recompute_true_still_traces(monkeypatch):
    """The DRIVER-side path must keep tracing on demand — the sweep and archive
    depend on resolving items outside the current graph, so the fix above must
    not be applied globally."""
    import dependency_graph.trace_paths as tp

    monkeypatch.setattr(tp, "_read_tap_snapshot", lambda item: None)
    calls = []

    def _record(item):
        calls.append(item)
        return (None, None, None)      # fn is None -> tolerant [] return

    monkeypatch.setattr(tp, "_resolve_pipeline_call", _record)

    assert tp.traced_paths_for(("target", "abc", "P"), recompute=True) == []
    assert calls == [("target", "abc", "P")]


# --------------------------------------------------------------------------- #
# Per-entity product filing (the viewer's ONE lookup)                          #
# --------------------------------------------------------------------------- #

def test_product_taps_key_is_content_addressed_per_routed_entity(monkeypatch):
    """The key the writer files under and the reader asks for: deployed code
    version + the ROUTED entity's own identity (URL type, stable hash, content
    hash) + pipeline. A code edit or new frames move it; nothing else does."""
    import dependency_graph.trace_paths as tp
    import state.code_version as cv

    monkeypatch.setattr(cv, "deployed_code_version", lambda: "sha-1")
    k = tp.product_taps_key("calibration_stack", "b82f", "c0ff", "P")
    assert k == ("tap_snapshots", "product", "sha-1",
                 "calibration_stack", "b82f", "c0ff", "P")
    monkeypatch.setattr(cv, "deployed_code_version", lambda: "sha-2")
    assert tp.product_taps_key("calibration_stack", "b82f", "c0ff", "P") != k


def test_store_product_taps_files_each_tap_under_its_routed_entity(monkeypatch):
    """A mosaic's trace holds the mosaic's own tap, a per-panel stack Master and
    a star flat — three different entities. Each is filed under ITS key, with
    only display/filter/path, and non-tap intermediates are dropped."""
    import compute_graph.compute_graph as cg
    import state.code_version as cv

    monkeypatch.setattr(cv, "deployed_code_version", lambda: "v")
    written = {}
    # Class-level: the actor-backed context routes instance `setattr` to Ray.
    monkeypatch.setattr(type(cg.context), "set_many",
                        lambda self, items: written.update(items))

    entries = [
        {"object_type": "mosaic", "object_id": "m", "content_hash": "cm",
         "display": "Mosaic", "filter": "H", "path": "/o/mosaic", "op_name": "merge"},
        {"object_type": "stack", "object_id": "s", "content_hash": "cs",
         "display": "Master", "filter": "H", "path": "/o/master", "op_name": "stack_files"},
        {"object_type": "calibration_stack", "object_id": "f", "content_hash": "cf",
         "display": "Star Flat", "filter": "S", "path": "/o/flat", "op_name": "build_star_flat"},
        {"object_type": None, "object_id": None, "content_hash": None,
         "display": None, "filter": None, "path": "/o/place", "op_name": "place_image"},
    ]
    cg._store_product_taps("P", entries)

    assert written == {
        ("tap_snapshots", "product", "v", "mosaic", "m", "cm", "P"):
            [{"display": "Mosaic", "filter": "H", "path": "/o/mosaic"}],
        ("tap_snapshots", "product", "v", "stack", "s", "cs", "P"):
            [{"display": "Master", "filter": "H", "path": "/o/master"}],
        ("tap_snapshots", "product", "v", "calibration_stack", "f", "cf", "P"):
            [{"display": "Star Flat", "filter": "S", "path": "/o/flat"}],
    }


def test_store_product_taps_writes_nothing_for_a_tapless_trace(monkeypatch):
    import compute_graph.compute_graph as cg
    calls = []
    monkeypatch.setattr(type(cg.context), "set_many",
                        lambda self, items: calls.append(items))
    cg._store_product_taps("P", [{"display": None, "path": "/o/x"}])
    assert calls == []
