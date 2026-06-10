"""Global ephemeral-file lifetime tracker.

A run_graph()-scoped registry that keys an on-disk filename to the set
of Node consumers (across every pipeline's traced graph) that still need
to read it. Per-pipeline GraphRuntimes share the same instance via the
parent chain.

The point: today each child GraphRuntime tracks its own
`_dependents`/`_ephemeral`/`_cleaned` and decides cleanup locally. Two
pipelines with structurally-equal Nodes for the same producer (cache
hit, work runs once) keep separate consumer bookkeeping — one
pipeline's "all dependents done" fires cleanup while the other
pipeline's consumer is still mid-flight, and the consumer hits a
FileNotFoundError on read.

Filename + set of consumer Nodes (structural identity) IS the global
view. Build it at trace time, decrement as nodes complete, unlink when
a set empties.

Trace-time population uses `dependency_graph.preflight` to predict each
ephemeral producer's filename without running anything; runtime
producer-completion records the actual on-disk write target via
`cleanup_ephemeral_artifact._master_path(result)`, which must agree
with the trace-time prediction (content-addressing invariant).
"""
from __future__ import annotations

import structlog
from typing import Any, Callable, Iterable

from dependency_graph.classes import Node


log = structlog.get_logger()


class EphemeralFiles:
    """Owns the global filename → consumer-set mapping for one
    `run_graph` invocation. Created on the root GraphRuntime; shared
    with every descendant via `parent._ephemeral_files`.
    """

    def __init__(self) -> None:
        # filename → set of consumer Nodes that haven't completed yet.
        # Populated at trace time (eager) per producer Node marked
        # _ephemeral; decremented at runtime in GraphRuntime._mark_done.
        self._pending: dict[str, set[Node]] = {}
        # filename → (producer_node, producer_result, producer_fn_args).
        # Captured at producer-completion time; consumed when the
        # pending set empties and cleanup fires.
        self._cleanup_data: dict[str, tuple[Node, Any, list]] = {}

    def register_from_graph(self, graph, fallback_to_node_key: bool = False) -> None:
        """Walk `graph` for `_ephemeral` producers, predict each
        producer's on-disk filename via `preflight`, and record every
        dependent Node as a pending consumer.

        Idempotent: re-registering the same graph or a structurally
        overlapping one is a set union — no double-counting.

        Tolerant: skips cycles. Producers with no consumers in the
        graph are skipped (the topology safety guard — terminals are
        never auto-cleaned, since they have no dependents).

        `fallback_to_node_key=True` keys producers without a
        predictable filename (test stubs, ops with no `filename_fn`)
        on `f"node:{producer.name}"` instead of skipping. Used when
        constructing a runtime from a local graph as the only source
        of ephemeral bookkeeping; production calls from eager trace
        leave the default off.
        """
        from dependency_graph.preflight import (
            predict_call_hashes, expected_output_path,
        )

        try:
            predicted = predict_call_hashes(graph)
        except ValueError as e:
            log.warning("ephemeral_files: skipping registry build, graph has cycle/orphan",
                        error=str(e))
            return

        # Reverse adjacency: dep → consumers (both data + ordering).
        dependents: dict[Node, set[Node]] = {}
        for n, deps in graph.items():
            for d in deps:
                if isinstance(d, Node):
                    dependents.setdefault(d, set()).add(n)

        for p in graph:
            if not isinstance(p, Node) or not getattr(p, "_ephemeral", False):
                continue
            consumers = dependents.get(p, set())
            if not consumers:
                continue  # terminal — topology safety guard
            filename = expected_output_path(p, predicted, graph)
            if filename is None:
                if not fallback_to_node_key:
                    continue
                key = f"node:{p.name}"
            else:
                key = str(filename)
            self._pending.setdefault(key, set()).update(consumers)

    def merge(self, other: "EphemeralFiles") -> None:
        """Union another registry's pending sets into ours. Called when
        a child Graph carries its own ephemeral_files (built at eager
        trace time in a graph-builder @node)."""
        if other is self:
            return
        for filename, pending in other._pending.items():
            self._pending.setdefault(filename, set()).update(pending)
        # Cleanup data is producer-scoped; transfer any already-captured
        # entries too, then re-evaluate (the producer may have completed
        # before this merge fires).
        for filename, data in other._cleanup_data.items():
            self._cleanup_data.setdefault(filename, data)

    def producer_done(
        self,
        filename: str,
        node: Node,
        result: Any,
        fn_args: list,
        cleanup_fn: Callable[[Node, Any, list], None],
    ) -> None:
        """Record that `node` (an ephemeral producer) finished writing
        `filename`. Fires cleanup if every consumer is already done."""
        if filename not in self._pending:
            return  # not tracked as ephemeral — nothing to do
        self._cleanup_data[filename] = (node, result, fn_args)
        self._maybe_cleanup(filename, cleanup_fn)

    def consumer_done(
        self,
        node: Node,
        cleanup_fn: Callable[[Node, Any, list], None] | None,
    ) -> None:
        """`node` finished — discard it from every pending set. Fire
        cleanup for any filename whose set just emptied (and whose
        producer is already done)."""
        if cleanup_fn is None:
            return
        # Iterate over a copy of the keys: _maybe_cleanup may del entries.
        for filename in list(self._pending.keys()):
            pending = self._pending.get(filename)
            if pending is None:
                continue
            pending.discard(node)
            if not pending:
                self._maybe_cleanup(filename, cleanup_fn)

    def _maybe_cleanup(
        self,
        filename: str,
        cleanup_fn: Callable[[Node, Any, list], None],
    ) -> None:
        if self._pending.get(filename):
            return  # still has pending consumers
        if filename not in self._pending:
            return  # already cleaned (or never registered)
        if filename not in self._cleanup_data:
            return  # producer not done yet
        node, result, fn_args = self._cleanup_data.pop(filename)
        del self._pending[filename]
        try:
            cleanup_fn(node, result, fn_args)
        except Exception as e:
            log.warning(
                "ephemeral cleanup hook raised",
                filename=filename, node=str(node), error=str(e),
            )

    # -- Introspection (tests / debugging) --

    def is_tracked(self, filename: str) -> bool:
        return filename in self._pending

    def pending_count(self, filename: str) -> int:
        return len(self._pending.get(filename, ()))

    def all_filenames(self) -> Iterable[str]:
        return self._pending.keys()
