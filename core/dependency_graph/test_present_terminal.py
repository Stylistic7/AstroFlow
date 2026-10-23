"""`present_terminal` — builder ≡ terminal at trace time.

The runtime half of the identity already exists (a concluded child's
terminal result IS the builder node's result). These tests pin the
trace-time half: a presented builder must register its terminal's tap
(display/filter/entity → UI snapshot + version-supersede map), resolve to
its terminal's expected path (workflow UI, downstream keys), and answer
with its terminal's op name and finality (archive/sweep lookups).
"""
from types import SimpleNamespace

from dependency_graph.classes import Graph, Node
from dependency_graph.tracing import IdentityRef, present_terminal
from dependency_graph.trace_paths import (
    _op_name_for_node, _walk_and_resolve_filenames_tagged,
    resolve_output_paths,
)


class Mosaic:
    """Not a real Mosaic: `url_type_of` falls back to the lowercased class
    name for an unknown class, so the tap routes as "mosaic" all the same."""
    stable_hash = "m0sa1c"
    content_hash = "c0ntent"


def _builder_fn(mosaic, filter):
    """Signature is load-bearing: `filter_of` positional-binds to it."""


_builder_fn.__name__ = "merge_scope"


class _FakeRemote:
    def __init__(self, fn):
        self._function = fn


_TERMINAL = SimpleNamespace(_function=SimpleNamespace(
    __name__="merge_images_oss", _display="Mosaic", _final=True))


def _presented_node():
    n = Node(_FakeRemote(_builder_fn), Mosaic(), "H", name="scope0")
    return present_terminal(n, _TERMINAL, path="/out/merge_images_oss/abc123.tiff")


def test_presented_builder_registers_its_terminals_tap():
    from compute_graph.compute_graph import extract_pipeline_taps
    taps = list(extract_pipeline_taps(Graph({_presented_node(): []})))
    assert len(taps) == 1
    node, tap = taps[0]
    assert (tap["object_type"], tap["object_id"]) == ("mosaic", "m0sa1c")
    assert tap["content_hash"] == "c0ntent"
    assert tap["display"] == "Mosaic"
    assert tap["filter"] == "H"
    # Control: the same builder UNPRESENTED registers nothing.
    bare = Node(_FakeRemote(_builder_fn), Mosaic(), "H", name="scope1")
    assert list(extract_pipeline_taps(Graph({bare: []}))) == []


def test_presented_builder_lands_in_the_supersede_map():
    from compute_graph.compute_graph import _version_supersede_map
    n = _presented_node()
    mapping = _version_supersede_map("PI Master + OSS Mosaic",
                                     Graph({n: []}))
    assert n in mapping, "presented deliverable missing from N-1 supersede"


def test_presented_builder_resolves_to_its_terminals_path():
    n = _presented_node()
    paths = resolve_output_paths(Graph({n: []}))
    assert paths.get(n) == "/out/merge_images_oss/abc123.tiff"
    # content_hash (== the path stem) is what downstream filename_fn args
    # fold — the same stem IdentityRef carries at the call site.
    assert n.content_hash == "abc123"
    assert IdentityRef("abc123").content_hash == n.content_hash


def test_presented_builder_answers_with_terminal_op_name_and_finality():
    n = _presented_node()
    assert _op_name_for_node(n) == "merge_images_oss"
    entries = _walk_and_resolve_filenames_tagged(Graph({n: []}))
    assert len(entries) == 1
    assert entries[0]["is_final"] is True
    assert entries[0]["op_name"] == "merge_images_oss"
    assert entries[0]["display"] == "Mosaic"
