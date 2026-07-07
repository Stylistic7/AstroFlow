"""Thin adapter between the runtime's `Graph` and the
`{skeleton, state}` dict the workflow-graph viewer consumes.

The heavy lifting lives elsewhere:

* `visual_graph.py` — VisualNode/VisualGraph types and the
  serializer that turns a VisualGraph into the frontend's dict
  shape.
* `graph_transforms.py` — pure `VisualGraph → VisualGraph`
  rewrites (remove transparent nodes, duplicate shared into
  consumer scopes). `sanitize_for_display` composes them.

This module just wires those together and exposes the two
runtime-callback entry points the rest of the codebase already
imports: `store_workflow_graph` (fires from the runtime's
`graph_update_fn`) and `store_node_status` (fires from
`node_status_fn`).
"""
from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Any

import ray

from cache.types import NeverRun
from context import context
from dependency_graph import GraphRuntime, Node
from dependency_graph.graph_transforms import sanitize_for_display
from dependency_graph.visual_graph import (
    VisualGraph,
    build_skeleton_from_visual,
    gather_state_from_visual,
    to_visual_graph,
)
from file_io.types import WithSidecar


def node_to_id(node: Node) -> str:
    """Canonical id of a Node, used as the unmangled key for
    `/api/node_status/<id>`. Matches `VisualNode.visual_id`'s
    pre-`@` prefix, so `node_id.partition("@")[0]` unmangles a
    skeleton id back to this."""
    return f"{node.task._function.__name__}_{hash(node)}"


def to_workflow_graph(rt: GraphRuntime, *, raw: bool = False) -> dict:
    """Serialize a runtime's graph to the viewer's
    `{skeleton, state}` shape.

    `raw=False` (default) runs the `sanitize_for_display`
    transform pipeline first: drop transparent ops, duplicate
    shared Nodes into each consumer scope. `raw=True` skips the
    pipeline — useful for `?raw=1` debugging when the sanitized
    view looks wrong."""
    vg = to_visual_graph(rt.graph)
    if not raw:
        vg = sanitize_for_display(vg)
    by_parent = rt.subgraphs_by_parent
    return {
        "skeleton": build_skeleton_from_visual(vg, rt, by_parent),
        "state": gather_state_from_visual(vg, rt, by_parent),
    }


def store_workflow_graph(key: Any, rt: GraphRuntime):
    """`graph_update_fn` callback: store the sanitized workflow-
    graph snapshot at `key` in the workflow_graphs ContextActor
    dict. Called periodically by the runtime as nodes complete,
    and once at end of run."""
    from cache.context_ref import ContextRef

    wf_graph = to_workflow_graph(rt)
    cache_key = ("workflow_graphs", str(key))
    new_ref = ContextRef.store(cache_key, wf_graph)
    context.or_equals("workflow_graphs", {key: new_ref})


# ---------------------------------------------------------------------------
# node_status — keyed by canonical node_to_id (one Node = one status).
# ---------------------------------------------------------------------------
# The UI's NodeStatusDialog fetches `/api/node_status/<id>` where
# `<id>` is a skeleton entry id (mangled `<node_to_id>@<suffix>`);
# the endpoint strips the `@suffix` and looks up by canonical id.
# So this stays a single dict entry per underlying Node, even when
# the Node renders as multiple visual copies.

# Captured structlog event dicts can carry arbitrarily large
# objects (a logged Graph/Stack/MetadataDict), which made single
# entries 100+ MB. Stringify and bound everything before it
# reaches the cache — the UI only ever shows this as text anyway.
_LOG_FIELD_MAX = 4000
_LOG_MAX_ENTRIES = 1000


def _stringify(v: Any) -> str:
    if isinstance(v, str):
        return v
    try:
        return repr(v)
    except Exception:
        return f"<unreprable {type(v).__name__}>"


def _truncate(s: str) -> str:
    if len(s) <= _LOG_FIELD_MAX:
        return s
    return s[:_LOG_FIELD_MAX] + f"…[+{len(s) - _LOG_FIELD_MAX} chars]"


def _bounded_logs(logs: Any) -> list:
    if not isinstance(logs, list):
        return [{"event": _truncate(_stringify(logs))}]
    out: list = []
    if len(logs) > _LOG_MAX_ENTRIES:
        out.append(
            {"event": f"[{len(logs) - _LOG_MAX_ENTRIES} earlier entries elided]"}
        )
    for e in logs[-_LOG_MAX_ENTRIES:]:
        if isinstance(e, dict):
            out.append({k: _truncate(_stringify(v)) for k, v in e.items()})
        else:
            out.append({"event": _truncate(_stringify(e))})
    return out


def store_node_status(node: Node, result: Any, logs: list[Any]):
    """`node_status_fn` callback: record one entry per canonical
    Node into `context[("node_status", <id>)]`."""
    if isinstance(result, ray.ObjectRef):
        result = ray.get(result)

    status_dict = dict(
        result_type=str(type(result).__name__), logs=_bounded_logs(logs)
    )

    if isinstance(result, Exception):
        # repr, not the raw exception: a pickled exception drags its
        # traceback frames/locals (more 100 MB blobs).
        status_dict["exception"] = _truncate(_stringify(result))
        status_dict["stack_trace"] = traceback.format_exc()
    elif isinstance(result, (Path, WithSidecar)):
        status_dict["path"] = str(result)

    nid = node_to_id(node)
    context[("node_status", nid)] = status_dict

    # Lean companion the workflow-graph serve overlay reads INSTEAD of the
    # heavyweight status above: {result_type, failed} only, so deriving a
    # node's display state never deserializes the (bounded but still large)
    # `logs` blob. Same evictable cache — best-effort, like node_status.
    context[("node_outcome", nid)] = {
        "result_type": status_dict["result_type"],
        "failed": "exception" in status_dict,
    }
