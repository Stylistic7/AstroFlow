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
