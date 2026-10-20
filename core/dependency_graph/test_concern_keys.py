"""Where a node's concerns are filed (`dependency_graph.logging._concerns_keys_for`):
under its first argument's object for a registered producer, and for an op
declared `@node(reports_concerns=True)` — whichever wrapper layer asks."""
from types import SimpleNamespace

from decorators import node
from dependency_graph.logging import _concerns_keys_for


class Thing:
    def __init__(self, h):
        self.stable_hash = h


def test_a_plain_node_files_no_concerns():
    @node
    def plain(x: Thing) -> int:
        return 1
    assert _concerns_keys_for(plain, (Thing("abc"),), {}) == []


def test_a_reporting_node_files_under_its_first_argument_at_every_layer():
    @node(reports_concerns=True)
    def reporting(x: Thing) -> int:
        return 1
    key = [("concerns", "Thing", "abc")]
    assert _concerns_keys_for(reporting, (Thing("abc"),), {}) == key
    # the layer `capture_logs` actually sees is an inner wrapper that `wraps` the raw function
    inner = reporting
    while getattr(inner, "__wrapped__", None) is not None:
        inner = inner.__wrapped__
        assert _concerns_keys_for(inner, (Thing("abc"),), {}) == key
    assert _concerns_keys_for(reporting, (SimpleNamespace(),), {}) == [], "no stable hash, nowhere to file"
