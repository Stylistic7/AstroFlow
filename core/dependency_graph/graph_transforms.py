"""Pure `VisualGraph → VisualGraph` transforms.

Each transform is a single-purpose, side-effect-free rewrite of the
visual graph. They compose: `sanitize_for_display` runs the full
stack the frontend sees. Bypassing the stack (via the
`/api/workflow_graph/.../?raw=1` query param) shows the raw graph
straight off the trace — useful for debugging when the sanitized
view looks wrong.
"""
from __future__ import annotations

from collections import defaultdict

from dependency_graph.visual_graph import VisualGraph, VisualNode


# Op names hidden in the sanitized view. These are system-glue
# nodes the user didn't write — trace hooks (`compute_image_metadata`,
# `_register_tap_path_op`) and sub-runtime sentinels
# (`run_pipeline_done`). `remove_transparent_nodes` stitches each
# transparent node's upstreams directly to its downstreams.
TRANSPARENT_OP_NAMES = frozenset({
    "compute_image_metadata",
    "_register_tap_path_op",
    "run_pipeline_done",
})


def remove_transparent_nodes(vg: VisualGraph) -> VisualGraph:
    """Drop VisualNodes whose op name is in `TRANSPARENT_OP_NAMES`.
    Stitch incoming edges to outgoing edges so an `A → T → B` chain
    becomes `A → B`; chains of transparent nodes
    (`A → T1 → T2 → B`) collapse to `A → B` in one pass."""
    def is_transparent(vn: VisualNode) -> bool:
        return vn.op_name in TRANSPARENT_OP_NAMES

    def resolve_through(start: VisualNode, seen=None) -> set[VisualNode]:
        if seen is None:
            seen = set()
        if start in seen:
            return set()
        seen.add(start)
        if not is_transparent(start):
            return {start}
        out: set[VisualNode] = set()
        for d in vg.get(start, []):
            out |= resolve_through(d, seen)
        return out

    new: VisualGraph = {}
    for downstream, deps in vg.items():
        if is_transparent(downstream):
            continue
        resolved: list[VisualNode] = []
        seen_per_node: set[VisualNode] = set()
        for d in deps:
            for r in resolve_through(d):
                if r not in seen_per_node:
                    seen_per_node.add(r)
                    resolved.append(r)
        new[downstream] = resolved
    return new


def duplicate_shared_into_consumer_scopes(vg: VisualGraph) -> VisualGraph:
    """If a VisualNode has direct consumers in two or more
    DISTINCT scopes, replace it with one copy per consumer scope.
    Each consumer is rewired to the copy at its own scope. A
    producer with one consumer (or many consumers all sharing one
    scope) is left alone — the cross-scope edge is rendered as a
    boundary crossing.

    Iterates to fixpoint: a producer whose direct consumers all
    sit in one scope might still need duplication once those
    consumers themselves get duplicated by a later pass."""

    def consumers_of(g: VisualGraph) -> dict[VisualNode, list[VisualNode]]:
        out: dict[VisualNode, list[VisualNode]] = defaultdict(list)
        for downstream, deps in g.items():
            for d in deps:
                out[d].append(downstream)
        return out

    current = dict(vg)
    while True:
        consumers = consumers_of(current)
        to_duplicate: dict[VisualNode, dict[VisualNode, tuple]] = {}
        for producer, prod_consumers in consumers.items():
            scope_of_consumer = {c: c.scope for c in prod_consumers}
            distinct_scopes = set(scope_of_consumer.values())
            distinct_scopes.discard(producer.scope)
            if len(distinct_scopes) < 2:
                continue
            # consumer → target scope (== consumer's own scope).
            to_duplicate[producer] = scope_of_consumer

        if not to_duplicate:
            break

        # For each producer to duplicate, build one VisualNode per
        # distinct consumer scope. Consumers sharing a scope share
        # a copy.
        copy_for: dict[tuple[VisualNode, tuple], VisualNode] = {}
        for producer, scope_of_consumer in to_duplicate.items():
            for s in set(scope_of_consumer.values()):
                copy_for[(producer, s)] = VisualNode(
                    canonical=producer.canonical, scope=s,
                )

        new: VisualGraph = {}
        for downstream, deps in current.items():
            if downstream in to_duplicate:
                # The producer being duplicated: its outgoing dep
                # edges flow into EACH copy.
                for s in set(to_duplicate[downstream].values()):
                    new[copy_for[(downstream, s)]] = list(deps)
                continue
            new_deps: list[VisualNode] = []
            for d in deps:
                if d in to_duplicate:
                    # Pick the copy of d that matches THIS
                    # consumer's scope.
                    target = copy_for.get((d, downstream.scope), d)
                    new_deps.append(target)
                else:
                    new_deps.append(d)
            new[downstream] = new_deps

        # Preserve dep-only nodes that aren't keys yet.
        keys = set(new.keys())
        for deps in list(new.values()):
            for d in deps:
                if d not in keys:
                    new.setdefault(d, [])
                    keys.add(d)

        current = new

    return current


def sanitize_for_display(vg: VisualGraph) -> VisualGraph:
    """The default transform pipeline applied before rendering.
    Bypassed by `?raw=1` on the workflow-graph endpoint."""
    return duplicate_shared_into_consumer_scopes(
        remove_transparent_nodes(vg)
    )
