"""Tests for the GraphRuntime cleanup hook + cleanup_ephemeral_artifact.

We use a stub `cleanup_fn` for the runtime tests so we can assert on
*when* cleanup fires; the actual file-deleting logic lives in
`cache/ephemeral_cleanup.py` and is exercised separately
(`test_cleanup_action` below).
"""
from pathlib import Path

import pytest

from cache.ephemeral_cleanup import cleanup_ephemeral_artifact
from dependency_graph.classes import Graph, Node
from dependency_graph.graph_runtime import GraphRuntime


# ──────────────────────────────────────────────────────────────────────────
# Test fixtures: dummy tasks with the attributes the runtime touches.
# ──────────────────────────────────────────────────────────────────────────

class _FakeFn:
    def __init__(self, name: str, n_params: int = 0):
        self.__module__ = "test_ephemeral_cleanup"
        self.__name__ = name
        # `make_fn_args` calls `inspect.signature(node.task._function)`.
        # Use real Python signatures by exec'ing a stub function.
        params = ", ".join(f"_p{i}" for i in range(n_params))
        ns: dict = {}
        exec(f"def {name}({params}): pass", ns)
        fn = ns[name]
        self._sig_fn = fn

    def __call__(self, *args, **kwargs):
        return None


class _FakeTask:
    def __init__(self, name: str, n_params: int = 0):
        self._function = _FakeFn(name, n_params)._sig_fn
        self._function.__name__ = name
        self._function.__module__ = "test_ephemeral_cleanup"


def _node(task, *args):
    return Node(task, *args)


def _ephemeral(node):
    node._ephemeral = True
    return node


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────

class TestCleanupHook:
    """The runtime-side trigger: when does cleanup_fn fire?"""

    def test_no_cleanup_when_node_not_ephemeral(self):
        a, b = _FakeTask("a"), _FakeTask("b")
        A, B = _node(a), _node(b)
        called: list = []

        def cleanup(node, result, args):
            called.append(node)

        rt = GraphRuntime(Graph({B: [A]}), cleanup_fn=cleanup)
        rt.start(A); rt.node_completed(A, "ra")
        rt.start(B); rt.node_completed(B, "rb")
        assert called == []

    def test_cleanup_after_only_consumer_completes(self):
        """A → B (ephemeral) → C. Cleanup of B fires after C completes."""
        a, b, c = _FakeTask("a"), _FakeTask("b"), _FakeTask("c")
        A, B, C = _node(a), _ephemeral(_node(b)), _node(c)
        called: list = []

        def cleanup(node, result, args):
            called.append((node, result))

        rt = GraphRuntime(Graph({C: [B], B: [A]}), cleanup_fn=cleanup)
        rt.start(A); rt.node_completed(A, "ra")
        rt.start(B); rt.node_completed(B, "rb")
        # B has one dependent (C). Until C completes, no cleanup.
        assert called == []
        rt.start(C); rt.node_completed(C, "rc")
        # Now both of B's dependents are done — exactly one cleanup call.
        assert called == [(B, "rb")]

    def test_cleanup_waits_for_all_consumers(self):
        """A → B (ephemeral) → {C, D}. Cleanup waits for BOTH C and D."""
        a, b, c, d = _FakeTask("a"), _FakeTask("b"), _FakeTask("c"), _FakeTask("d")
        A, B, C, D = _node(a), _ephemeral(_node(b)), _node(c), _node(d)
        called: list = []

        def cleanup(node, result, args):
            called.append(node)

        rt = GraphRuntime(
            Graph({D: [B], C: [B], B: [A]}),
            cleanup_fn=cleanup,
        )
        rt.start(A); rt.node_completed(A, "ra")
        rt.start(B); rt.node_completed(B, "rb")
        rt.start(C); rt.node_completed(C, "rc")
        # Only one consumer (C) done — D still pending.
        assert called == []
        rt.start(D); rt.node_completed(D, "rd")
        # Both done now.
        assert called == [B]

    def test_terminal_ephemeral_not_cleaned(self):
        """An ephemeral node with no dependents (a user mistake — e.g. a
        `final=True` op inside `ephemeral_intermediates()`) is preserved
        by the safety guard."""
        a = _FakeTask("a")
        A = _ephemeral(_node(a))
        called: list = []

        def cleanup(node, result, args):
            called.append(node)

        rt = GraphRuntime(Graph({A: []}), cleanup_fn=cleanup)
        rt.start(A); rt.node_completed(A, "ra")
        assert called == []  # not cleaned — no consumer

    def test_cleanup_fires_only_once(self):
        """If two consumers complete back-to-back, cleanup runs exactly once."""
        a, b, c, d = _FakeTask("a"), _FakeTask("b"), _FakeTask("c"), _FakeTask("d")
        A, B, C, D = _node(a), _ephemeral(_node(b)), _node(c), _node(d)
        called: list = []

        def cleanup(node, result, args):
            called.append(node)

        rt = GraphRuntime(
            Graph({D: [B], C: [B], B: [A]}),
            cleanup_fn=cleanup,
        )
        rt.start(A); rt.node_completed(A, "ra")
        rt.start(B); rt.node_completed(B, "rb")
        rt.start(C); rt.node_completed(C, "rc")
        rt.start(D); rt.node_completed(D, "rd")
        # Even if some implementation detail re-fires the trigger, the
        # _cleaned set guards us.
        assert called == [B]

    def test_cleanup_exception_does_not_break_run(self):
        a, b, c = _FakeTask("a"), _FakeTask("b"), _FakeTask("c")
        A, B, C = _node(a), _ephemeral(_node(b)), _node(c)

        def cleanup(node, result, args):
            raise RuntimeError("simulated cleanup failure")

        rt = GraphRuntime(Graph({C: [B], B: [A]}), cleanup_fn=cleanup)
        rt.start(A); rt.node_completed(A, "ra")
        rt.start(B); rt.node_completed(B, "rb")
        # Cleanup raises, but node_completed must still succeed.
        rt.start(C); rt.node_completed(C, "rc")
        assert C in rt.done_nodes


class TestCleanupAction:
    """The cleanup function itself: deletes the right files, calls invalidate."""

    def test_non_tiff_master_deletes_master_and_sidecar_keeps_tiff(self, tmp_path):
        from file_io.sidecar import sidecar_path_for

        master = tmp_path / "abc123.fits"
        master.write_bytes(b"fake fits")
        sidecar = sidecar_path_for(master)
        sidecar.write_text('{"astroflow": {}}')
        tiff = master.with_suffix(".tiff")
        tiff.write_bytes(b"fake tiff")

        # Fake node: has display => UI-relevant; cleanup keeps the tiff.
        class Task:
            class _function:
                __name__ = "merge"
                __module__ = "test"
            _display = "Merged"
            invalidate_called = []
            @classmethod
            def invalidate(cls, *args):
                cls.invalidate_called.append(args)
        node = Node(Task, "literal")
        result = master   # bare Path

        cleanup_ephemeral_artifact(node, result, fn_args=["literal"])

        assert not master.exists()
        assert not sidecar.exists()
        assert tiff.exists()
        assert Task.invalidate_called == [("literal",)]

    def test_tiff_master_no_display_deletes_tiff(self, tmp_path):
        from file_io.sidecar import sidecar_path_for

        master = tmp_path / "abc123.tiff"
        master.write_bytes(b"fake tiff that's the master")
        sidecar = sidecar_path_for(master)
        sidecar.write_text('{"astroflow": {}}')

        class Task:
            class _function:
                __name__ = "place"
                __module__ = "test"
            invalidate_called = []
            @classmethod
            def invalidate(cls, *args):
                cls.invalidate_called.append(args)
        node = Node(Task, "lit")

        cleanup_ephemeral_artifact(node, master, fn_args=["lit"])

        assert not master.exists()
        assert not sidecar.exists()
        assert Task.invalidate_called == [("lit",)]

    def test_tiff_master_with_display_preserves(self, tmp_path):
        master = tmp_path / "abc.tiff"
        master.write_bytes(b"tiff that IS the tile pyramid")

        class Task:
            class _function:
                __name__ = "subframe_store"
                __module__ = "test"
            _display = "Subframe"
            invalidate_called = []
            @classmethod
            def invalidate(cls, *args):
                cls.invalidate_called.append(args)
        node = Node(Task, "lit")

        cleanup_ephemeral_artifact(node, master, fn_args=["lit"])

        # Master + display → keep everything, UI needs the tiff.
        assert master.exists()
        assert Task.invalidate_called == []

    def test_withsidecar_wrapped_result(self, tmp_path):
        from file_io.types import WithSidecar, FitsFilePath

        master_path = tmp_path / "wrapped.fits"
        master_path.write_bytes(b"x")
        wrapped = WithSidecar(FitsFilePath(str(master_path)))

        class Task:
            class _function:
                __name__ = "merge"; __module__ = "test"
            _display = "Merged"
            @staticmethod
            def invalidate(*args): pass
        node = Node(Task, "lit")

        cleanup_ephemeral_artifact(node, wrapped, fn_args=["lit"])
        assert not master_path.exists()

    def test_missing_file_no_error(self, tmp_path):
        """Unlink on a missing file is a no-op, not a failure."""
        master = tmp_path / "never_existed.fits"
        class Task:
            class _function:
                __name__ = "merge"; __module__ = "test"
            _display = "Merged"
            @staticmethod
            def invalidate(*args): pass
        node = Node(Task, "lit")
        # Must not raise.
        cleanup_ephemeral_artifact(node, master, fn_args=["lit"])

    def test_unknown_result_type_no_error(self):
        class Task:
            class _function:
                __name__ = "x"; __module__ = "test"
        node = Node(Task)
        # Result has no .path/.filename — bail silently.
        cleanup_ephemeral_artifact(node, "not a path at all", fn_args=[])


class TestCrossPipelineFusion:
    """The A6/A7 fix: a producer's cleanup must wait for ALL
    structurally-equal consumers across every child runtime, not just
    the consumers in one runtime's local graph."""

    def test_two_child_runtimes_sharing_an_ephemeral_producer(self):
        """Pipeline A graph: P (ephemeral) → CA. Pipeline B graph: P → CB.
        P is the same producer in both — structurally equal Node. Without
        the global registry, runtime A would clean P as soon as CA
        completes, leaving CB FNF when it later tries to read P's file.
        With the registry, P's pending set is {CA, CB} unioned across
        both child runtimes; cleanup fires only after both."""
        p_task, ca_task, cb_task = (
            _FakeTask("p"), _FakeTask("ca"), _FakeTask("cb"),
        )
        P_for_a = _ephemeral(_node(p_task))
        P_for_b = _ephemeral(_node(p_task))
        # Structural identity: both Ps hash + equal-compare the same.
        assert P_for_a == P_for_b
        assert hash(P_for_a) == hash(P_for_b)

        CA, CB = _node(ca_task), _node(cb_task)
        called: list = []

        def cleanup(node, result, args):
            called.append(node)

        # Root runtime has no ephemerals — just a placeholder.
        root = GraphRuntime(Graph({}), cleanup_fn=cleanup)
        # Two child sub-Graphs simulate two pipelines that both
        # consume the same producer. Each child runtime is created
        # the way `node_completed` would create it.
        child_a = GraphRuntime(Graph({CA: [P_for_a]}), parent=root)
        child_b = GraphRuntime(Graph({CB: [P_for_b]}), parent=root)

        # P "runs" in pipeline A: producer_done fires from child_a.
        child_a.start(P_for_a); child_a.node_completed(P_for_a, "rp")
        # Pipeline A's only local consumer completes.
        child_a.start(CA); child_a.node_completed(CA, "rca")
        # Without cross-pipeline fusion, cleanup would fire here.
        # With the global registry, it shouldn't — CB still pending.
        assert called == []
        # Pipeline B's consumer completes.
        child_b.start(CB); child_b.node_completed(CB, "rcb")
        # NOW the producer's pending set is empty across both pipelines.
        assert called == [P_for_a]


class TestChildRuntimeInheritsCleanup:
    """A subgraph should inherit its parent's cleanup_fn so ephemeral
    nodes nested inside a `@graph_builder` still get reclaimed."""

    def test_child_inherits_when_not_overridden(self):
        a = _FakeTask("a")
        A = _node(a)
        called: list = []

        def cleanup(node, result, args):
            called.append(node)

        parent = GraphRuntime(Graph({A: []}), cleanup_fn=cleanup)
        child = GraphRuntime(Graph({A: []}), parent=parent)
        assert child.cleanup_fn is cleanup

    def test_child_explicit_cleanup_wins(self):
        a = _FakeTask("a")
        A = _node(a)

        def parent_cleanup(node, result, args): pass
        def child_cleanup(node, result, args): pass

        parent = GraphRuntime(Graph({A: []}), cleanup_fn=parent_cleanup)
        child = GraphRuntime(
            Graph({A: []}), parent=parent, cleanup_fn=child_cleanup,
        )
        assert child.cleanup_fn is child_cleanup

    def test_no_cleanup_when_neither_set(self):
        a = _FakeTask("a")
        A = _node(a)
        rt = GraphRuntime(Graph({A: []}))
        assert rt.cleanup_fn is None
