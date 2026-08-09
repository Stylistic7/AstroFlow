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
from typing import Any, Callable

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
from dependency_graph.ephemeral_registry import EphemeralFiles


def _landed_path(result) -> "str | None":
    """The primary on-disk path of a node result, from the RESULT itself.

    Producer results are ImageMetadata (`.filename`) after the metadata
    wrap; raw file-path results (WithSidecar/TiffFilePath) stringify to
    their path. Non-file results (dicts, graphs, None) return None."""
    import os
    fname = getattr(result, "filename", None)
    if fname:
        return str(fname)
    path = getattr(result, "path", None)
    if path is not None:
        return str(path)
    if isinstance(result, (str, os.PathLike)) and str(result):
        return os.fspath(result)
    return None
from cache.types import NeverRun
from dependency_graph.classes import Skipped


class GraphRuntime:
    def __init__(
        self,
        graph: Graph,
        parent: GraphRuntime | None = None,
        parent_node: Node | None = None,
        cleanup_fn: Callable[[Node, Any, list], None] | None = None,
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
        # Active children participate in scheduling; completed ones are
        # retained in `_completed_children` purely so the workflow-graph
        # UI can still expand a subgraph after it finishes (otherwise a
        # completed parent would visually collapse to a "simple" node).
        self._children: dict[Node, GraphRuntime] = {}
        self._completed_children: dict[Node, GraphRuntime] = {}
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

        # Ephemeral output cleanup. Nodes flagged `_ephemeral` (by the
        # trace-mode wrapper in `cache/cached_decorator.py` when the call
        # was inside an `ephemeral_intermediates()` scope) get their
        # on-disk output reclaimed once every dependent across every
        # pipeline has completed. The bookkeeping lives in a single
        # `EphemeralFiles` registry shared across the entire run via the
        # parent chain — see `dependency_graph/ephemeral_registry.py`.
        # `cleanup_fn` is the actual delete-the-file callable —
        # `cleanup_ephemeral_artifact` in `cache/ephemeral_cleanup.py`
        # in production, mockable for tests.
        self.cleanup_fn: Callable[[Node, Any, list], None] | None = (
            cleanup_fn
            if cleanup_fn is not None
            else (parent.cleanup_fn if parent else None)
        )
        # Root holds the live registry; children share it via parent.
        if parent is None:
            self._ephemeral_files: EphemeralFiles = EphemeralFiles()
        else:
            self._ephemeral_files = parent._root._ephemeral_files

        # Deterministic N-1 supersession map ({node: (version_log_key,
        # expected_path)}), attached by `run_pipeline` to its returned
        # sub-graph. Root holds the union across all sub-graphs; `_mark_done`
        # consults it so a display tap's previous version is deleted at the
        # MOMENT its new output lands (`version_log.supersede`).
        incoming_versions = getattr(graph, "version_supersede", None) or {}
        if parent is None:
            self._version_supersede: dict = dict(incoming_versions)
        else:
            self._version_supersede = parent._root._version_supersede
            self._version_supersede.update(incoming_versions)

        # If this Graph carries its own pre-traced registry (built at
        # eager-trace time inside a graph-builder @node), union it into
        # the shared one.
        incoming = getattr(graph, "ephemeral_files", None)
        if incoming is not None:
            self._ephemeral_files.merge(incoming)
        # Also register from this runtime's own graph — but skip the
        # work entirely when no cleanup_fn is wired up (snapshot
        # runtimes used by the workflow-graph adapter, structure-only
        # test fixtures, dry-run paths). Without a cleanup_fn nothing
        # consults the registry, so populating it is pure waste.
        # For production this is redundant with eager-trace
        # contributions (structural Node identity → idempotent set
        # union). For tests and stand-alone use without an eager-trace
        # pass, it's the only source of bookkeeping.
        if self.cleanup_fn is not None:
            self._ephemeral_files.register_from_graph(
                self._graph, fallback_to_node_key=True,
            )

    @property
    def graph(self) -> dict[Node, list[Node]]:
        """The dependency graph dict. Used by workflow_graph_adapter."""
        return self._graph

    @property
    def _root(self) -> "GraphRuntime":
        """Walk up the parent chain to the root runtime — the one that
        owns the shared `EphemeralFiles` registry."""
        rt = self
        while rt.parent is not None:
            rt = rt.parent
        return rt

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
            # Preflight the dynamic subgraph before splicing it in, exactly as
            # run_graph does for the top-level graph. Without this, a subgraph
            # whose deliverables are ALREADY on disk still re-dispatches its
            # entire body bottom-up — and an ephemeral intermediate whose output
            # was reclaimed then cache-MISSES and needlessly recomputes (the
            # real "bxt_correct_only ran 134x/run" churn). Lazy import:
            # `_apply_preflight` lives in dependency_graph.dependency_graph,
            # which imports this module — importing at load time would cycle.
            from dependency_graph.dependency_graph import _apply_preflight
            # is_subgraph=True: restore the single-terminal invariant if the
            # reverse-walk dropped this subgraph's funnel sentinel (a pure
            # successor of an unsatisfied final master whose consumers are still
            # on disk), else the __init__ assertion below trips with multiple
            # bare terminals.
            pruned = _apply_preflight(result, is_subgraph=True)
            if pruned:
                child = GraphRuntime(pruned, parent=self, parent_node=node)
                self._children[node] = child
            else:
                # Fully satisfied — every deliverable is already on disk, so
                # there is no work to spin up. An empty subgraph would trip the
                # single-terminal assertion in GraphRuntime.__init__, so don't
                # create a child: mark the builder node done directly. Builder
                # results are consumed only by no-output completion sentinels
                # (main_processing_graph_done / run_pipeline_done) that ignore
                # the value, so `None` is the correct result here.
                self._mark_done(node, None, logs, succeeded=True)
        else:
            # Skipped/NeverRun sentinels propagate through node_completed
            # but don't represent a real, file-producing success — gate
            # cleanup decisions accordingly.
            succeeded = result is not Skipped and result is not NeverRun
            self._mark_done(node, result, logs, succeeded=succeeded)

    def node_failed(self, node: Node, error: Exception, logs: list[Any] | None = None) -> None:
        """Report that a node raised an exception."""
        if logs is None:
            logs = []
        self._mark_done(node, error, logs, succeeded=False)
        self.final_result = error

    def ancestor_results(self, node: Node) -> list[Any]:
        """Get results from this node's dependencies."""
        return [self.results[dep] for dep in self._graph.get(node, [])]

    @property
    def landed_paths(self) -> set:
        """RUNTIME census: the primary path of every successful completion
        (hits and misses alike) this run, collected at `_mark_done` and
        shared root-wide. Stored durably at full-run completion as the
        orphan sweep's live-set — the single ledger of what the runtime
        actually touched, replacing trace predictions (Phase 1b)."""
        root = self._root
        if not hasattr(root, "_landed_paths"):
            root._landed_paths = set()
        return root._landed_paths

    @property
    def census_sound(self) -> bool:
        """False once ANY node in ANY runtime (this one or a child) failed
        or was Skipped this run.

        A census is a DELETION authority: files not in it are orphans. A
        failed or skipped node contributes no landed path, yet its output
        may well exist on disk from an earlier run and still be live — so
        one failure anywhere makes the census unsafe to store. Set at the
        `_mark_done` chokepoint, which every runtime funnels through, so
        child-subgraph failures count too (the root graph alone can't see
        them: a child's failure surfaces as a FAILED parent node, never as
        a never-run one)."""
        root = self._root
        return getattr(root, "_census_sound", True)

    def make_fn_args(self, node: Node) -> list[Any]:
        """Build the argument list for a node's remote function.

        Data-dep results are placed at their ORIGINAL arg positions
        (`node._dep_positions`), so a node result can sit anywhere in the
        call — first, middle, or last — not only as a trailing arg. Nodes
        built directly (no `_dep_positions`, e.g. in tests) fall back to
        the historical literals-then-deps layout."""
        from dependency_graph.classes import reconstruct_call_args
        positions = getattr(node, "_dep_positions", None)
        literals = list(node.args)
        if positions is None:
            sig = inspect.signature(node.task._function)
            if len(sig.parameters) > len(literals):
                return literals + self.ancestor_results(node)
            return literals
        # `_dep_positions` counts the data deps exactly (robust to
        # `*cal_masters`); they sit before any ordering-only `_after`
        # deps in `graph[node]`.
        deps = self._graph.get(node, [])[: len(positions)]
        dep_values = [self.results[d] for d in deps]
        return reconstruct_call_args(literals, dep_values, positions)

    def _notify_ephemeral_registry(
        self, node: Node, result: Any, succeeded: bool,
    ) -> None:
        """Two notifications fire here in succession:

        1. If `node` is itself an ephemeral producer, register the
           cleanup data (filename, result, fn_args) so the registry can
           fire `cleanup_fn` once every consumer is also done.
        2. `node` may be a consumer of someone else's ephemeral
           producer — discard from the registry's pending sets, fire
           cleanup for any filename whose set just emptied.

        Gated on `succeeded`: a failed producer wrote no file (skip
        producer registration); a failed consumer still needs the
        upstream file for the next manual re-run (skip consumer
        decrement). Failures leave intermediates around for one extra
        run, intentionally.

        The producer-side key is the result's master path (extracted
        from `ImageMetadata.filename` / `WithSidecar.path`) when one
        exists — matches the eager trace's `expected_output_path` so
        cross-pipeline consumer sets unify. When the result has no
        extractable path (test stubs, ops without `filename_fn`), we
        fall back to `f"node:{node.name}"`, mirroring the same fallback
        in `EphemeralFiles.register_from_graph`.
        """
        if self.cleanup_fn is None or not succeeded:
            return
        registry = self._root._ephemeral_files
        if getattr(node, "_ephemeral", False):
            from cache.ephemeral_cleanup import _master_path
            master = _master_path(result)
            key = str(master) if master is not None else f"node:{node.name}"
            registry.producer_done(
                key, node, result,
                self.make_fn_args(node), self.cleanup_fn,
            )
        registry.consumer_done(node, self.cleanup_fn)

    def _mark_done(
        self, node: Node, result: Any, logs: list[Any],
        succeeded: bool = True,
    ) -> None:
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

        self._notify_ephemeral_registry(node, result, succeeded)

        # Graceful N/N-1, rebased on RUNTIME truth (Phase 1b): the landed
        # path comes from the node's ACTUAL result, never a prediction.
        # `supersede` records the landed entry as the product's current
        # version and deletes every other logged version's files — the only
        # deletion moment where supersession is a local fact. Best-effort:
        # a versioning hiccup must never fail the run. Every landed path
        # also feeds the run's census (`landed_paths`) — the runtime ledger
        # the orphan sweep's live-set is built from.
        if not succeeded:
            # Failure or Skipped sentinel: the census can no longer claim to
            # be a complete picture of what's live (see `census_sound`).
            self._root._census_sound = False
        if succeeded:
            landed = _landed_path(result)
            if landed is not None:
                self._root.landed_paths.add(landed)
                key = self._root._version_supersede.get(node)
                if key is not None:
                    try:
                        from products import version_log
                        version_log.supersede(key, landed)
                    except Exception as e:
                        print(f"### version supersede failed (non-fatal) for "
                              f"{node.name}: {type(e).__name__}: {e}")

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
                # The parent node "succeeds" iff the child subgraph
                # reached a real result — failed children propagate an
                # Exception in final_result, which we treat as failure
                # for cleanup-gating purposes (don't decrement consumer
                # sets for a failed parent's spot in the parent graph).
                parent_succeeded = not isinstance(
                    child.final_result, Exception,
                )
                self._mark_done(
                    node, child.final_result, child.logs,
                    succeeded=parent_succeeded,
                )
                # Move out of _children so we stop scheduling against
                # it, but keep it in _completed_children so the
                # workflow-graph UI can still render the subgraph.
                self._completed_children[node] = self._children.pop(node)

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
        """Recursively collect all child runtimes indexed by parent node.
        Includes completed subgraphs so the workflow-graph UI can still
        expand a subgraph after it finishes.
        """
        result = {**self._children, **self._completed_children}
        for child in self._children.values():
            result.update(child.subgraphs_by_parent)
        for child in self._completed_children.values():
            result.update(child.subgraphs_by_parent)
        return result
