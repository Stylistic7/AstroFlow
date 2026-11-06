"""Recursive graph runtime with eager subgraph expansion.

GraphRuntime replaces TopologicalSorterWithStatus + the flat subgraphs list
in run_graph(). It manages a tree of subgraphs internally:

- When a node completes with a Graph result, a child GraphRuntime is
  created immediately. Its ready nodes are visible on the next get_ready()
  and surface through the next drain_newly_ready().
- When a child completes, its final_result propagates to the parent
  and unblocks dependents automatically.
- get_ready() is idempotent and recursive — it returns ready nodes from
  all levels of the tree.
- Readiness is an EVENT: every node that becomes ready is recorded at that
  moment, and drain_newly_ready() hands run_graph exactly those nodes, so
  the dispatcher never rescans the graph to find work.
"""
from __future__ import annotations

import inspect
import time
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


def _is_ephemeral(node: Node) -> bool:
    """Whether `node`'s op declared `@node(ephemeral=True)`.

    Read off the task's wrapper. Defaults False, so an op that never opted in
    keeps its output — the safe direction, and it means a node built directly in
    a test (no @node wrapper) is never reclaimed."""
    task = getattr(node, "task", None)
    return bool(getattr(task, "_ephemeral", False))


def _terminal_cache_hit(graph: Graph) -> "tuple[bool, Any]":
    """`(hit, value)` for a subgraph's single terminal — a read-only cache
    probe that never executes anything.

    The terminal's positional args are reconstructed the way the runtime
    would bind them (`reconstruct_call_args` + `_dep_positions`), except the
    dep slots hold the traced dep HANDLES instead of results. That yields the
    same cache key: a key resolves any object carrying `content_hash` through
    `_maybe_use_content_hash`, and a handle is stamped at trace time with
    exactly the `content_hash` its result will have (the
    `content_hash := producer.call_hash` convention).

    Never raises: a failed probe just means "expand the subgraph", which is
    always correct and only slower.
    """
    try:
        from dependency_graph.classes import reconstruct_call_args

        depended = {d for deps in graph.values() for d in deps}
        terminals = [n for n in graph if n not in depended]
        if len(terminals) != 1:
            return (False, None)
        terminal = canonicalize_node(terminals[0])
        cached_value = getattr(terminal.task, "cached_value", None)
        if cached_value is None:
            return (False, None)   # not a cached op (sentinel, raw remote)
        positions = getattr(terminals[0], "_dep_positions", None)
        deps = list(graph.get(terminals[0], []) or [])
        if positions is not None:
            deps = deps[: len(positions)]
        args = reconstruct_call_args(list(terminal.args), deps, positions)
        return cached_value(*args)
    except Exception:
        return (False, None)


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
        # (started_at, ray node id or None) per started node — what
        # `running_snapshot` publishes as "running where, since when".
        self._placement: dict[Node, tuple[float, str | None]] = {}

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
        # Readiness transitions not yet handed to the dispatcher, oldest
        # first — the initial frontier in graph (trace) order, then every
        # dependent as it becomes ready. See drain_newly_ready().
        self._newly_ready: list[Node] = [n for n in self._graph if n in self._ready]

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

        # Deterministic N-1 supersession map ({node: version_log_key}),
        # attached by `run_pipeline` to its returned sub-graph. Root holds the
        # union across all sub-graphs; `_mark_done` consults it so a display
        # tap's previous version is deleted at the MOMENT its new output lands
        # (`version_log.supersede`, called with the node's RUNTIME result).
        incoming_versions = getattr(graph, "version_supersede", None) or {}
        if parent is None:
            self._version_supersede: dict = dict(incoming_versions)
        else:
            self._version_supersede = parent._root._version_supersede
            self._version_supersede.update(incoming_versions)
        # Dispatch priority (`Graph.priority`): a sub-graph's own, else its
        # parent's — a starred pipeline's nested scopes are starred too.
        self.priority: int = getattr(graph, "priority", 0) or (parent.priority if parent is not None else 0)

    @property
    def graph(self) -> dict[Node, list[Node]]:
        """The dependency graph dict. Used by workflow_graph_adapter."""
        return self._graph

    @property
    def _root(self) -> "GraphRuntime":
        """Walk up the parent chain to the root runtime — the one that owns
        the run-wide state (supersession map, landed-path census)."""
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

    def start(self, node: Node, placed_on: str | None = None) -> None:
        """Mark a node as started (submitted to Ray), pinned to Ray node
        `placed_on` or unpinned (None)."""
        self._started.add(node)
        self._ready.discard(node)
        self._placement[node] = (time.time(), placed_on)

    def started(self, node: Node) -> bool:
        return node in self._started

    def drain_newly_ready(self) -> list[tuple[GraphRuntime, Node]]:
        """Nodes that became ready since the last drain, oldest first, each
        tagged with the runtime that owns it — across the whole tree, a
        child's initial frontier included.

        run_graph's only source of work. A readiness transition is
        something `_mark_done` already observes, so it is recorded there
        and handed over here instead of being rediscovered by rescanning
        the graph: O(dependents) per completion, independent of how many
        nodes are ready or retained. Nodes started or completed meanwhile
        (driver-side cache hits) are dropped.
        """
        self._propagate_completed_children()
        out = [(self, n) for n in self._newly_ready if n in self._ready]
        self._newly_ready.clear()
        for child in self._children.values():
            if child.is_active():
                out.extend(child.drain_newly_ready())
        return out

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
            # SKIP THE WHOLE SUBGRAPH when its terminal is already a cache hit.
            # Only decidable here, at runtime: the Graph exists, so its terminal
            # and that terminal's dep handles are in hand. This matters
            # specifically for graphs holding EPHEMERAL nodes — their outputs
            # were reclaimed after the last run, so expanding would re-derive
            # every one of them just to hand the terminal a cache hit and
            # reclaim them again. Skipping hands the consumer the terminal's
            # cached value and never dispatches anything.
            if any(_is_ephemeral(n) for n in result):
                hit, value = _terminal_cache_hit(result)
                if hit:
                    self._mark_done(node, value, logs, succeeded=True)
                    return
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
                and dependent not in self._ready
                and all(d in self._done for d in self._graph[dependent])
            ):
                self._ready.add(dependent)
                self._newly_ready.append(dependent)


        # Graceful N/N-1, rebased on RUNTIME truth: the landed path comes from
        # the node's ACTUAL result, never a prediction. `supersede` records the
        # landed entry as the product's current version and deletes every other
        # logged version's files — the only deletion moment where supersession
        # is a local fact. Best-effort: a versioning hiccup must never fail the
        # run.
        #
        # No runtime census is collected here any more (2026-08-17): the orphan
        # sweep's live set is the TRACE, which over-pins (safe) where a census
        # under-pins (needs a completeness guard, and deadlocks without one).
        # See cache/orphan_sweep.py.
        if succeeded:
            landed = _landed_path(result)
            if landed is not None:
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
                # The one moment a subgraph concludes, so the one place its
                # ephemeral outputs can be reclaimed. Fires for EVERY concluded
                # child: what is reclaimable is now declared per-OP
                # (`@node(ephemeral=True)`), not per-graph, so the runtime no
                # longer needs the builder to have flagged the scope. Success
                # only: a failed graph keeps its files so a retry resumes from
                # real data instead of re-deriving everything.
                if parent_succeeded:
                    child._reclaim_ephemerals()
                # Move out of _children so we stop scheduling against
                # it, but keep it in _completed_children so the
                # workflow-graph UI can still render the subgraph.
                self._completed_children[node] = self._children.pop(node)

    def _reclaim_ephemerals(self) -> None:
        """Reclaim this graph's EPHEMERAL outputs, sparing its terminal.

        Ephemerality is declared by the OP, not by whoever built the graph:
        `@node(ephemeral=True)` means "my output is a working value, not a
        deliverable". The op's author is the one who knows whether it is cheap
        to recompute, so the decision lives with the code that knows the answer
        instead of in a `Graph(delete_intermediates=True)` in some pipeline file
        — a different file, and duplicated at every scope site.

        THE TERMINAL IS SPARED, even when marked ephemeral. Its result is what
        escapes to consumers outside this graph, and inside this graph it has no
        dependents at all, so "everything that needed it is done" is vacuously
        true and reclaiming it here would delete a value about to be read. That
        is a DEFERRAL, not an exemption: one level up the same value is an
        interior node, and the enclosing graph reclaims it then. This is what
        lets `blend_panels` be reclaimed at pipeline completion while
        `crop_panel` still dies per-panel.

        Runs only after the graph concluded SUCCESSFULLY — a failed graph keeps
        its files so a retry resumes from real data rather than re-deriving
        everything.

        Each file comes from that node's own RUNTIME result, never a predicted
        path, so this can only ever delete something this graph actually
        produced. Nodes that produce no image (orchestration, sentinels) are
        skipped by `reclaim_artifact` itself.

        Sharing is deliberately untracked: an intermediate another pipeline
        wants is re-derived. Never raises — reclaim is a side effect, never
        load-bearing.
        """
        from cache.reclaim import reclaim_artifact
        from dependency_graph.dependency_graph import _terminals

        keep = set(_terminals(self._graph))
        deleted = 0
        for node in self._graph:
            if node in keep or not _is_ephemeral(node):
                continue
            result = self.results.get(node)
            if result is None:
                continue
            try:
                reclaim_artifact(node, result, self.make_fn_args(node))
                deleted += 1
            except Exception as e:
                print(f"### ephemeral reclaim failed (non-fatal) for "
                      f"{node.name}: {type(e).__name__}: {e}")
        if deleted:
            print(f"### reclaimed {deleted} ephemeral output(s) from a "
                  f"concluded graph")

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
            in_flight_placement={n: self._placement[n] for n in in_flight
                                 if n in self._placement},
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
