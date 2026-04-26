"""Recursive graph runtime with eager subgraph expansion.

GraphRuntime replaces TopologicalSorterWithStatus + the flat subgraphs list
in run_graph(). It manages a tree of subgraphs internally:

- When a node completes with a Graph result, a child GraphRuntime is
  created immediately. Its ready nodes are visible on the next get_ready().
- When a child completes, its final_result propagates to the parent
  and unblocks dependents automatically.
- get_ready() is idempotent and recursive — it returns ready nodes from
  all levels of the tree.
"""
from __future__ import annotations

import inspect
from collections import Counter, defaultdict
from typing import Any

from dependency_graph.classes import (
    Graph,
    Node,
    NodeStatusFn,
    GraphStatus,
    GraphUpdateFn,
    ResultDict,
    canonicalize_graph,
    canonicalize_node,
)
from cache.types import NeverRun
from dependency_graph.classes import Skipped


class GraphRuntime:
    def __init__(
        self,
        graph: Graph,
        parent: GraphRuntime | None = None,
        parent_node: Node | None = None,
    ) -> None:
        canonical = canonicalize_graph(graph)

        # Validate single terminal node for subgraphs.
        if parent is not None and canonical:
            all_nodes = set(canonical.keys())
            depended_on = {dep for deps in canonical.values() for dep in deps}
            terminal = all_nodes - depended_on
            assert len(terminal) == 1, (
                f"Subgraph has {len(terminal)} terminal nodes, expected 1: "
                f"{[str(n) for n in terminal]}"
            )

        self.parent = parent
        self.parent_node = parent_node

        # The dependency graph: node -> list of dependency nodes
        self._graph: dict[Node, list[Node]] = defaultdict(list)
        for node, deps in canonical.items():
            self._graph[node] = list(deps)
            for dep in deps:
                if dep not in self._graph:
                    self._graph[dep] = []

        self._started: set[Node] = set()
        self._done: set[Node] = set()

        # Reverse dependencies: node -> set of nodes that depend on it.
        self._dependents: dict[Node, set[Node]] = defaultdict(set)
        for node, deps in self._graph.items():
            for dep in deps:
                self._dependents[dep].add(node)

        # Incrementally-maintained ready set: nodes whose deps are all
        # done and that haven't been started yet.
        self._ready: set[Node] = {
            n for n in self._graph
            if not self._graph[n]  # no dependencies
        }

        # Children: nodes that returned a Graph get a child runtime.
        self._children: dict[Node, GraphRuntime] = {}
        self._child_done_count: int = 0

        # Per-function-name counters used for "place_image 12/34" style
        # progress display. Total is fixed for this runtime; done is
        # bumped in _mark_done.
        self._total_per_fn: Counter = Counter(
            n.task._function.__name__ for n in self._graph
        )
        self._done_per_fn: Counter = Counter()

        # Results and metadata
        self.results = ResultDict()
        self.logs = ResultDict()
        self.filenames: dict[Node, Any] = {}
        self.final_result: Any = None

        # Callbacks inherited from Graph
        self.graph_update_fn: GraphUpdateFn | None = graph.graph_update_fn
        self.node_status_fn: NodeStatusFn | None = (
            graph.node_status_fn
            or (parent.node_status_fn if parent else None)
        )

    @property
    def graph(self) -> dict[Node, list[Node]]:
        """The dependency graph dict. Used by workflow_graph_adapter."""
        return self._graph

    @property
    def done_nodes(self) -> list[Node]:
        return list(self._done)

    def get_ready(self) -> list[Node]:
        """Return nodes that are ready to run (all deps done, not yet started).

        Idempotent — safe to call multiple times.
        """
        self._propagate_completed_children()
        return list(self._ready)

    def start(self, node: Node) -> None:
        """Mark a node as started (submitted to Ray)."""
        self._started.add(node)
        self._ready.discard(node)

    def node_completed(
        self, node: Node, result: Any, logs: list[Any] | None = None,
    ) -> None:
        """Report that a node has completed with the given result.

        If result is a Graph, creates a child runtime whose ready nodes
        will be visible on the next get_ready() call.
        """
        if logs is None:
            logs = []

        if isinstance(result, Graph):
            child = GraphRuntime(result, parent=self, parent_node=node)
            self._children[node] = child
        else:
            self._mark_done(node, result, logs)

    def node_failed(self, node: Node, error: Exception, logs: list[Any] | None = None) -> None:
        """Report that a node raised an exception."""
        if logs is None:
            logs = []
        self._mark_done(node, error, logs)
        self.final_result = error

    def ancestor_results(self, node: Node) -> list[Any]:
        """Get results from this node's dependencies."""
        return [self.results[dep] for dep in self._graph.get(node, [])]

    def make_fn_args(self, node: Node) -> list[Any]:
        """Build the argument list for a node's remote function."""
        sig = inspect.signature(node.task._function)
        args = list(node.args)
        if len(sig.parameters) > len(node.args):
            args += self.ancestor_results(node)
        return args

    def _mark_done(self, node: Node, result: Any, logs: list[Any]) -> None:
        self.results[node] = result
        self.logs[node] = logs
        self._done.add(node)
        self._done_per_fn[node.task._function.__name__] += 1
        self._ready.discard(node)

        # Check if completing this node unblocks any dependents.
        for dependent in self._dependents.get(node, ()):
            if (
                dependent not in self._started
                and all(d in self._done for d in self._graph[dependent])
            ):
                self._ready.add(dependent)

        # Opportunistically compute the output filename for the workflow UI.
        if hasattr(node.task, "filename_fn"):
            try:
                args = self.make_fn_args(node)
                self.filenames[node] = node.task.filename_fn(*args)
            except (AttributeError, KeyError, TypeError):
                pass

        # Don't let Skipped overwrite a meaningful result or exception.
        if result is not Skipped:
            self.final_result = result

        if self.node_status_fn is not None:
            self.node_status_fn(node, result, logs)

    def _propagate_completed_children(self) -> None:
        """Check children; if any are done, fold their results into this runtime."""
        for node, child in list(self._children.items()):
            child._propagate_completed_children()
            if not child.is_active():
                self._child_done_count += len(child._done) + child._child_done_count
                # Roll the child's per-fn counters up so they survive
                # the child's removal from _children. Without this, the
                # UI's "place_image 12/34" denominator would shrink as
                # subgraphs finish.
                self._done_per_fn += child._done_per_fn
                self._total_per_fn += child._total_per_fn
                self._mark_done(node, child.final_result, child.logs)
                del self._children[node]

    def is_active(self) -> bool:
        """True if there is still work to do at this level or any child."""
        self._propagate_completed_children()
        if self._ready or (len(self._done) + len(self._children) < len(self._graph)):
            return True
        return any(c.is_active() for c in self._children.values())

    def get_all_ready(self) -> list[tuple[GraphRuntime, Node]]:
        """Return (runtime, node) pairs for all ready nodes in the tree.

        This is the main interface for run_graph — it returns ready nodes
        from all levels, each tagged with the runtime that owns them.
        """
        self._propagate_completed_children()

        result = [(self, n) for n in self.get_ready()]
        for child in self._children.values():
            if child.is_active():
                result.extend(child.get_all_ready())
        return result

    # --- Status / introspection ---

    def waiting_count(self) -> int:
        return len(self._graph) - len(self._started) - len(self._ready)

    def get_status(self) -> GraphStatus:
        done_count = len(self._done) + self._child_done_count
        # Parent nodes whose function returned a child Graph aren't doing
        # work themselves; the actual in-flight work is in the child's
        # in_flight_nodes when its get_status() is folded in below.
        in_flight = self._started - self._done - self._children.keys()
        status = GraphStatus(
            done_count=done_count,
            ready_nodes=self.get_ready(),
            in_flight_nodes=list(in_flight),
            waiting_count=self.waiting_count(),
            done_per_fn=Counter(self._done_per_fn),
            total_per_fn=Counter(self._total_per_fn),
        )
        for child in self._children.values():
            status = status + child.get_status()
        return status

    @property
    def subgraphs_by_parent(self) -> dict[Node, "GraphRuntime"]:
        """Recursively collect all child runtimes indexed by parent node."""
        result = dict(self._children)
        for child in self._children.values():
            result.update(child.subgraphs_by_parent)
        return result
