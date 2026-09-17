"""Two graph runtimes asking for the SAME call while it is in flight share
one execution. A runtime dedups within itself and the @node cache catches a
repeat only once the first call has landed — so two cart pipelines for one
mosaic, each owning an identical `stack_files` node, ran WBPP twice (boot 91:
three masters twice, ~75 min of PixInsight). The dispatcher now keys
in-flight work by the cache key and rides the existing ref.
"""
import time

import ray

from decorators import node
from dependency_graph import run_graph
from dependency_graph.classes import Graph
from dependency_graph.tracing import trace


@node
def slow_shared(tag: str) -> str:
    time.sleep(3.0)  # long enough that both runtimes see it in flight
    return f"{tag}@{time.time():.6f}"


@ray.remote(num_cpus=0.1)
def child(tag: str, dry_run: bool = False):
    del dry_run

    def _body():
        return slow_shared(tag)
    return Graph(trace(_body))


def _count_submits(monkeypatch):
    import dependency_graph.dependency_graph as dd
    counts: dict[str, int] = {}
    real = dd._submit_node

    def counting(rt, node, *a, **kw):
        name = getattr(getattr(node.task, "_function", node.task), "__name__", "?")
        counts[name] = counts.get(name, 0) + 1
        return real(rt, node, *a, **kw)

    monkeypatch.setattr(dd, "_submit_node", counting)
    return counts


def test_identical_calls_in_two_runtimes_run_once(test_context, monkeypatch):
    counts = _count_submits(monkeypatch)
    from dependency_graph.classes import Node
    a = Node(child, "shared", name="child-a")
    b = Node(child, "shared", name="child-b")
    run_graph(Graph({a: set(), b: set()}))
    assert counts.get("slow_shared") == 1, counts
    # and both children got the ONE result (a cache hit afterwards agrees)
    hit, value = slow_shared.cached_value("shared")
    assert hit and value.startswith("shared@")


def test_different_calls_are_not_conflated(test_context, monkeypatch):
    counts = _count_submits(monkeypatch)
    from dependency_graph.classes import Node
    a = Node(child, "one", name="child-one")
    b = Node(child, "two", name="child-two")
    run_graph(Graph({a: set(), b: set()}))
    assert counts.get("slow_shared") == 2, counts
