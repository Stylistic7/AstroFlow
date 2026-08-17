"""Tests for GraphRuntime — the recursive, idempotent graph scheduler."""
import pytest
from dependency_graph.classes import Graph, Node, GraphStatus
from dependency_graph.graph_runtime import GraphRuntime


# Dummy remote functions for building test graphs.
# These are never actually called — we just need distinct function objects
# with _function attributes for Node hashing.
class _FakeRemote:
    def __init__(self, name):
        self._function = type(name, (), {
            "__module__": "test",
            "__name__": name,
        })()

a_task = _FakeRemote("a")
b_task = _FakeRemote("b")
c_task = _FakeRemote("c")
d_task = _FakeRemote("d")
gather_task = _FakeRemote("gather")


def _node(task, *args, **kwargs):
    return Node(task, *args, **kwargs)


class TestBasicTopologicalOrder:

    def test_linear_chain(self):
        """A → B → C: nodes become ready one at a time."""
        A, B, C = _node(a_task), _node(b_task), _node(c_task)
        rt = GraphRuntime(Graph({C: [B], B: [A]}))

        assert rt.get_ready() == [A]
        rt.start(A)
        rt.node_completed(A, "result_a")

        assert rt.get_ready() == [B]
        rt.start(B)
        rt.node_completed(B, "result_b")

        assert rt.get_ready() == [C]
        rt.start(C)
        rt.node_completed(C, "result_c")

        assert rt.get_ready() == []
        assert not rt.is_active()
        assert rt.final_result == "result_c"

    def test_diamond(self):
        """A → {B, C} → D: B and C are ready in parallel after A."""
        A = _node(a_task)
        B = _node(b_task)
        C = _node(c_task)
        D = _node(d_task)
        rt = GraphRuntime(Graph({D: {B, C}, C: {A}, B: {A}}))

        assert rt.get_ready() == [A]
        rt.start(A)
        rt.node_completed(A, "a")

        ready = set(rt.get_ready())
        assert ready == {B, C}

        rt.start(B)
        rt.start(C)
        rt.node_completed(B, "b")
        assert rt.get_ready() == []  # D not ready yet, C still running

        rt.node_completed(C, "c")
        assert rt.get_ready() == [D]

    def test_single_node(self):
        A = _node(a_task)
        rt = GraphRuntime(Graph({A: []}))

        assert rt.get_ready() == [A]
        rt.start(A)
        rt.node_completed(A, 42)
        assert not rt.is_active()
        assert rt.final_result == 42


class TestIdempotentGetReady:

    def test_repeated_calls_return_same_nodes(self):
        """get_ready() is idempotent until start() is called."""
        A = _node(a_task)
        B = _node(b_task)
        rt = GraphRuntime(Graph({B: [A]}))

        assert rt.get_ready() == [A]
        assert rt.get_ready() == [A]  # Same result
        assert rt.get_ready() == [A]  # Still same

        rt.start(A)
        assert rt.get_ready() == []  # A is now in progress

    def test_start_removes_from_ready(self):
        A = _node(a_task)
        B = _node(b_task)
        C = _node(c_task)
        rt = GraphRuntime(Graph({C: {A, B}}))

        assert set(rt.get_ready()) == {A, B}
        rt.start(A)
        assert rt.get_ready() == [B]
        rt.start(B)
        assert rt.get_ready() == []


class TestSubgraphExpansion:

    def test_node_returning_graph_creates_child(self):
        """When a node completes with a Graph, a child runtime is created."""
        A = _node(a_task)
        B = _node(b_task)
        rt = GraphRuntime(Graph({B: [A]}))

        rt.start(A)
        # A returns a subgraph with a single node
        inner = _node(c_task)
        rt.node_completed(A, Graph({inner: []}))

        # A is NOT done yet (waiting for child to complete)
        assert A not in rt._done
        # The child's ready nodes are visible
        all_ready = rt.get_all_ready()
        assert len(all_ready) == 1
        child_rt, node = all_ready[0]
        assert node == inner
        assert child_rt is not rt  # It's the child runtime

    def test_child_completion_unblocks_parent(self):
        """When a child runtime completes, the parent node becomes done."""
        A = _node(a_task)
        B = _node(b_task)
        rt = GraphRuntime(Graph({B: [A]}))

        rt.start(A)
        inner = _node(c_task)
        rt.node_completed(A, Graph({inner: []}))

        # Complete the inner node
        child_rt = rt._children[A]
        child_rt.start(inner)
        child_rt.node_completed(inner, "inner_result")

        # Now B should be ready (A is done via child propagation)
        assert rt.get_ready() == [B]
        assert rt.results[A] == "inner_result"

    def test_nested_subgraphs(self):
        """Three levels deep: A → subgraph → subgraph → leaf."""
        A = _node(a_task)
        rt = GraphRuntime(Graph({A: []}))

        # Level 1: A returns a graph
        rt.start(A)
        mid = _node(b_task)
        rt.node_completed(A, Graph({mid: []}))

        # Level 2: mid returns another graph
        child1 = rt._children[A]
        all_ready = rt.get_all_ready()
        assert len(all_ready) == 1
        assert all_ready[0] == (child1, mid)

        child1.start(mid)
        leaf = _node(c_task)
        child1.node_completed(mid, Graph({leaf: []}))

        # Level 3: leaf is ready
        all_ready = rt.get_all_ready()
        assert len(all_ready) == 1
        child2 = child1._children[mid]
        assert all_ready[0] == (child2, leaf)

        # Complete the leaf — should propagate all the way up
        child2.start(leaf)
        child2.node_completed(leaf, "deep_result")

        assert not rt.is_active()
        assert rt.final_result == "deep_result"

    def test_subgraph_ready_nodes_immediate(self):
        """Subgraph expansion makes child nodes available immediately."""
        A = _node(a_task)
        B = _node(b_task)
        C = _node(c_task)
        rt = GraphRuntime(Graph({C: {A, B}}))

        # Start and complete A with a subgraph
        rt.start(A)
        inner_a = _node(d_task, name="inner_a")
        rt.node_completed(A, Graph({inner_a: []}))

        # B is still ready at the parent level, inner_a is ready in the child
        all_ready = rt.get_all_ready()
        nodes = {n for _, n in all_ready}
        assert B in nodes
        assert inner_a in nodes


class TestSingleTerminalNodeInvariant:

    def test_multiple_terminals_rejected(self):
        A = _node(a_task)
        B = _node(b_task)
        # A and B are both terminals (nothing depends on them)
        with pytest.raises(AssertionError, match="terminal nodes"):
            parent = GraphRuntime(Graph({_node(c_task): []}))
            parent.start(_node(c_task))
            parent.node_completed(
                _node(c_task), Graph({A: [], B: []})
            )

    def test_empty_subgraph_allowed(self):
        A = _node(a_task)
        rt = GraphRuntime(Graph({A: []}))
        rt.start(A)
        rt.node_completed(A, Graph({}))
        # Empty graph — no child created, A stays as-is


class TestExceptionPropagation:

    def test_node_failure_marks_done(self):
        A = _node(a_task)
        B = _node(b_task)
        rt = GraphRuntime(Graph({B: [A]}))

        rt.start(A)
        err = ValueError("boom")
        rt.node_failed(A, err)

        assert A in rt._done
        assert rt.results[A] is err

    def test_final_result_set_on_failure(self):
        A = _node(a_task)
        rt = GraphRuntime(Graph({A: []}))
        rt.start(A)
        err = RuntimeError("fail")
        rt.node_failed(A, err)
        assert rt.final_result is err


class TestIsActive:

    def test_active_when_work_remains(self):
        A = _node(a_task)
        rt = GraphRuntime(Graph({A: []}))
        assert rt.is_active()

    def test_inactive_when_all_done(self):
        A = _node(a_task)
        rt = GraphRuntime(Graph({A: []}))
        rt.start(A)
        rt.node_completed(A, 1)
        assert not rt.is_active()

    def test_active_while_child_active(self):
        A = _node(a_task)
        rt = GraphRuntime(Graph({A: []}))
        rt.start(A)
        rt.node_completed(A, Graph({_node(b_task): []}))
        assert rt.is_active()  # child still running

    def test_inactive_after_child_completes(self):
        A = _node(a_task)
        rt = GraphRuntime(Graph({A: []}))
        rt.start(A)
        inner = _node(b_task)
        rt.node_completed(A, Graph({inner: []}))

        child = rt._children[A]
        child.start(inner)
        child.node_completed(inner, "done")

        # Need to trigger propagation
        rt.get_ready()
        assert not rt.is_active()


class TestStatus:

    def test_status_includes_children(self):
        A = _node(a_task)
        B = _node(b_task)
        rt = GraphRuntime(Graph({B: [A]}))

        rt.start(A)
        inner = _node(c_task)
        rt.node_completed(A, Graph({inner: []}))

        status = rt.get_status()
        # B is waiting (dep A not done yet), inner is ready in child
        assert len(status.ready_nodes) == 1  # inner
        assert status.waiting_count == 1  # B


class TestSubgraphsByParent:

    def test_collects_children_recursively(self):
        A = _node(a_task)
        rt = GraphRuntime(Graph({A: []}))

        rt.start(A)
        mid = _node(b_task)
        rt.node_completed(A, Graph({mid: []}))

        child = rt._children[A]
        child.start(mid)
        leaf = _node(c_task)
        child.node_completed(mid, Graph({leaf: []}))

        by_parent = rt.subgraphs_by_parent
        assert A in by_parent
        assert mid in by_parent
        assert len(by_parent) == 2


class TestEmptyGraph:

    def test_empty_graph_is_inactive(self):
        rt = GraphRuntime(Graph({}))
        assert not rt.is_active()
        assert rt.get_ready() == []


class TestVersionSupersede:
    """The graceful-N/N-1 completion hook: a display-tap node landing its new
    output triggers `version_log.supersede` for exactly that node, only on
    real success — the deterministic replacement for the deleted global
    prune/auto-sweep."""

    def _capture(self, monkeypatch):
        import products.version_log as vl
        calls = []
        monkeypatch.setattr(vl, "supersede",
                            lambda k, p: calls.append((k, p)) or 0)
        return calls

    def test_supersede_fires_on_success_with_runtime_path(self, monkeypatch):
        """Phase 1b: the map carries the product KEY only; the landed path
        comes from the node's ACTUAL result (never a prediction)."""
        from types import SimpleNamespace
        calls = self._capture(monkeypatch)
        A, B = _node(a_task), _node(b_task)
        g = Graph({B: [A]})
        g.version_supersede = {B: ("product_versions", "x")}
        rt = GraphRuntime(g)

        rt.node_completed(A, "ra")
        assert calls == []                      # unmapped node: no supersede
        rt.node_completed(B, SimpleNamespace(filename="/out/RUNTIME_b.tiff"))
        assert calls == [(("product_versions", "x"), "/out/RUNTIME_b.tiff")]

    def test_supersede_skipped_on_failure_and_sentinels(self, monkeypatch):
        from dependency_graph.graph_runtime import Skipped
        calls = self._capture(monkeypatch)
        A, B = _node(a_task), _node(b_task)
        g = Graph({B: [A]})
        g.version_supersede = {
            A: ("product_versions", "a"),
            B: ("product_versions", "b"),
        }
        rt = GraphRuntime(g)

        rt.node_failed(A, RuntimeError("boom"))
        rt.node_completed(B, Skipped)   # sentinel: no filename/path → no call
        assert calls == []
        # A failure or skip makes the census an incomplete picture of what's
        # live — storing it would condemn real files as orphans.


    def test_child_subgraph_map_merges_into_root(self, monkeypatch):
        """run_pipeline attaches the map to its RETURNED sub-graph; the child
        runtime must surface it to the root so completion of a child node
        supersedes."""
        calls = self._capture(monkeypatch)
        parent_node = _node(a_task)
        rt = GraphRuntime(Graph({parent_node: []}))

        C, done = _node(c_task), _node(gather_task)
        sub = Graph({done: [C]})
        sub.version_supersede = {C: ("product_versions", "c")}
        monkeypatch.setattr(
            "dependency_graph.dependency_graph._apply_preflight",
            lambda g, is_subgraph=False: g,
        )
        rt.start(parent_node)
        rt.node_completed(parent_node, sub)    # splices child runtime

        from types import SimpleNamespace
        child = rt._children[parent_node]
        child.node_completed(C, SimpleNamespace(filename="/out/RUNTIME_c.tiff"))
        assert calls == [(("product_versions", "c"), "/out/RUNTIME_c.tiff")]


