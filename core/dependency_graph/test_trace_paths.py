"""Unit tests for trace-time content_hash propagation and @node-caching.

The pure-Merkle invariant requires that the tracer's `_TracedOutput`
proxy hashes to its `content_hash` field (= the upstream's `call_hash`)
through the same `_maybe_use_content_hash` path the runtime uses for
real upstream `Path`/`ImageMetadata` values. If that fails, every
downstream call_hash diverges from runtime and the sweep produces
wrong filenames.

Plus: `_traced_paths` (the cached `@node` wrapping the actual trace) must
cache hits cheaply and invalidate when the pipeline source changes —
the trick that turns 17 s repeat-traces into sub-ms cache hits.
"""
from __future__ import annotations

import ray

from dependency_graph.trace_paths import _TracedOutput, _traced_paths
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


def test_traced_paths_is_node_wrapped(test_context):
    """Sanity: `_traced_paths` is a `@node`, so the cache machinery is
    available (`.remote`, `.is_cached`). Without these, the whole
    'cache the trace' design collapses."""
    assert hasattr(_traced_paths, "remote")
    assert hasattr(_traced_paths, "is_cached")


def test_traced_paths_caches_result_for_same_inputs(test_context):
    """Calling `_traced_paths` twice with identical args is a cache hit
    on the second call. Verified via `.is_cached`."""
    args = (
        "no_such_pipeline",  # unknown name → function short-circuits to []
        "entity-A", "config-A", "metadata-A", "source-hash-A",
    )
    # First call: cache miss; result is [] (unknown pipeline). Stored.
    result = ray.get(_traced_paths.remote(*args))
    assert result == []

    # Same args → cache hit.
    assert _traced_paths.is_cached(*args)


def test_traced_paths_invalidates_on_pipeline_source_hash_change(test_context):
    """The whole point of the `pipeline_source_hash` arg: changing it
    must produce a fresh cache key (i.e. miss on the new value)."""
    base = (
        "no_such_pipeline", "entity-A", "config-A", "metadata-A",
    )
    ray.get(_traced_paths.remote(*base, "source-hash-A"))
    assert _traced_paths.is_cached(*base, "source-hash-A")

    # Edit to the pipeline body → caller passes a new source hash.
    # That call must miss (different cache key), then store its own entry.
    assert not _traced_paths.is_cached(*base, "source-hash-B")
    ray.get(_traced_paths.remote(*base, "source-hash-B"))
    assert _traced_paths.is_cached(*base, "source-hash-B")
    # Original entry is untouched — both coexist.
    assert _traced_paths.is_cached(*base, "source-hash-A")


def test_traced_paths_invalidates_on_metadata_change(test_context):
    """Changing the `metadata_ref` arg's content_hash invalidates the
    cache. This is the mechanism by which adding/removing subframes or
    external masters triggers a re-trace."""
    base = (
        "no_such_pipeline", "entity-A", "config-A",
    )
    ray.get(_traced_paths.remote(*base, "metadata-v1", "source-hash"))
    assert _traced_paths.is_cached(*base, "metadata-v1", "source-hash")
    assert not _traced_paths.is_cached(*base, "metadata-v2", "source-hash")


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
