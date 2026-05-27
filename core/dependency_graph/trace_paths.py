"""Per-cart-item file-path enumeration for the orphan-sweep workflow.

`paths_for_item((object_type, object_id, pipeline_name))` traces the
pipeline for the named cart item and walks the resulting `Graph` in
topological order, calling each Node's `filename_fn` with upstream
file paths substituted for upstream Node refs. The returned list is
every on-disk path the pipeline would (or has) produced.

The walk skips Nodes whose upstream files don't exist on disk yet:
`_make_call_hash_filename_fn` hashes Path args via
`get_content_hash`, which reads the file. Missing upstream file →
unresolvable downstream call_hash → skip. Those paths can't be
orphan candidates anyway — the sweep cares about files that ARE on
disk.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from context import local_context as context
from dependency_graph.classes import Graph, Node, canonicalize_graph


def paths_for_item(
    item: tuple[str, str, str], *, finals_only: bool = False,
) -> list[Path]:
    """All paths the cart item `(object_type, object_id, pipeline_name)`
    would write to disk.

    `finals_only=True` filters to Nodes whose task has `_final == True`
    (set via `@makes_image(final=True)`). Used by the cart→archive
    flow's "preserve finals only" mode.

    Returns `[]` for an unresolvable item (unknown entity, unknown
    pipeline, trace failure). Never raises.
    """
    graph = _trace_for_item(item)
    if graph is None:
        return []
    return _enumerate_paths(graph, finals_only=finals_only)


def _trace_for_item(item: tuple[str, str, str]) -> Graph | None:
    """Resolve `item` against the current context and trace its
    pipeline. Returns the resulting `Graph` or `None`."""
    from decisions.settings import ProcessingConfig
    from pipelines.registry import get_pipeline

    object_type_str, object_id, pipeline_name = item
    entity = _resolve_entity(object_type_str, object_id)
    if entity is None:
        return None
    settings = context["settings"]
    if settings is None:
        return None
    config = ProcessingConfig(
        calibration_stacks=context["calibration_stacks"] or {},
        development_settings=settings.development_settings,
        min_frames_in_stack=settings.clustering_settings.min_frames_in_stack,
    )
    try:
        runner = get_pipeline(pipeline_name, processes=type(entity))
    except KeyError:
        return None
    try:
        return runner(entity, config)
    except Exception:
        return None


def _resolve_entity(object_type_str: str, object_id: str) -> Any:
    """Look up the entity object for a cart item. Mirrors the
    backend's `metadata._resolve_object` minus the identity-class
    plumbing; returns the entity itself (Stack / Target / Mosaic /
    CalibrationStack) or None."""
    try:
        if object_type_str == "stack":
            return context.stacks[object_id].get()
        if object_type_str == "calibration_stack":
            return context.calibration_stacks[object_id]
        if object_type_str == "target":
            return context.targets[object_id].get()
        if object_type_str == "mosaic":
            return context.mosaics[object_id].get()
    except (KeyError, TypeError, AttributeError):
        return None
    return None


def _enumerate_paths(graph: Graph, *, finals_only: bool) -> list[Path]:
    """Walk `graph` in topological order, resolving each Node's
    `filename_fn` against upstream resolved paths."""
    canonical = canonicalize_graph(graph)
    resolved: dict[Node, Path] = {}
    paths: list[Path] = []

    for node in _topological_order(canonical):
        path = _resolve_node_filename(node, canonical, resolved)
        if path is None:
            continue
        resolved[node] = path
        if finals_only and not getattr(node.task, "_final", False):
            continue
        paths.append(path)

    return paths


def _topological_order(graph: dict[Node, list[Node]]) -> list[Node]:
    """Kahn's algorithm. Stable order: nodes are ordered by their
    first appearance in `graph.keys()` among those with satisfied
    deps."""
    in_degree = {n: 0 for n in graph}
    for n in graph:
        for d in graph[n]:
            if d not in in_degree:
                in_degree[d] = 0
    for n, deps in graph.items():
        in_degree[n] = len(deps)

    ordered: list[Node] = []
    pending = [n for n in graph if in_degree[n] == 0]
    seen: set[Node] = set()
    while pending:
        nxt = pending.pop(0)
        if nxt in seen:
            continue
        seen.add(nxt)
        ordered.append(nxt)
        for n in graph:
            if nxt in graph[n]:
                in_degree[n] -= 1
                if in_degree[n] == 0:
                    pending.append(n)
    return ordered


def _resolve_node_filename(
    node: Node,
    graph: dict[Node, list[Node]],
    resolved: dict[Node, Path],
) -> Path | None:
    """Call `node.task.filename_fn` with explicit args + resolved
    upstream paths. Returns the path or None if anything goes
    wrong (no filename_fn, unresolvable upstream, call_hash
    computation failed because an upstream file is missing on
    disk, etc.)."""
    task = node.task
    if not hasattr(task, "filename_fn"):
        return None
    deps = graph.get(node, [])
    if any(d not in resolved for d in deps):
        return None
    try:
        sig = inspect.signature(task._function)
    except (TypeError, ValueError):
        sig = None
    args = list(node.args)
    if sig is not None and len(sig.parameters) > len(args):
        args = args + [resolved[d] for d in deps]
    try:
        return task.filename_fn(*args)
    except Exception:
        return None
