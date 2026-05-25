import ray
from flask import Blueprint, Response
import json
import subprocess

from context import local_context as context

compute_graphs_bp = Blueprint("compute_graphs", __name__)

_EMPTY_WORKFLOW_GRAPH = {"skeleton": [], "state": {}}


@compute_graphs_bp.route("/api/compute_graph")
def compute_graph():
    # TODO: Don't convert to a workflow graph here, do it through a
    # graph_update_fn.
    # return to_workflow_graph(context.compute_graphs)
    return {}


@compute_graphs_bp.route("/api/compute_graph/status")
def compute_graph_status():
    return context.compute_graph_status or {}


# Workflow-graph routes. Today user pipelines don't call
# `set_workflow_key(...)` from their bodies, so the `workflow_graphs`
# context dict is empty and these routes return the empty-graph
# sentinel. The pipeline / filter segments are accepted but unused;
# they reserve the URL shape for when pipelines start tagging their
# trace with workflow keys.
@compute_graphs_bp.route("/api/workflow_graph/stack/<stack_id>")
@compute_graphs_bp.route("/api/workflow_graph/stack/<stack_id>/<pipeline>")
def workflow_graph_for_stack(stack_id, pipeline=None):
    wf = context["workflow_graphs"] or {}
    graph = _resolve_ref(wf.get(("stack", stack_id, pipeline))) if pipeline else None
    return graph if graph is not None else _EMPTY_WORKFLOW_GRAPH


@compute_graphs_bp.route("/api/workflow_graph/target/<target_id>")
@compute_graphs_bp.route("/api/workflow_graph/target/<target_id>/<pipeline>")
def workflow_graph_for_target(target_id, pipeline=None):
    wf = context["workflow_graphs"] or {}
    graph = _resolve_ref(wf.get(("target", target_id, pipeline))) if pipeline else None
    return graph if graph is not None else _EMPTY_WORKFLOW_GRAPH


@compute_graphs_bp.route("/api/workflow_graph/mosaic/<mosaic_id>")
@compute_graphs_bp.route("/api/workflow_graph/mosaic/<mosaic_id>/<pipeline>")
@compute_graphs_bp.route("/api/workflow_graph/mosaic/<mosaic_id>/<pipeline>/<filter>")
def workflow_graph_for_mosaic(mosaic_id, pipeline=None, filter=None):
    """One workflow graph per `(mosaic, id, pipeline)`, snapshotted
    at pipeline-trace time and refreshed with live runtime state
    by the runtime's `graph_update_fn`. The optional `<filter>`
    segment slices the snapshot to nodes whose ancestor closure
    reaches a `filter`-bearing sink — sliced on read here rather
    than written as a separate snapshot, so `graph_update_fn`
    stays filter-unaware."""
    wf = context["workflow_graphs"] or {}
    if not pipeline:
        return _EMPTY_WORKFLOW_GRAPH
    snapshot = _resolve_ref(wf.get(("mosaic", mosaic_id, pipeline)))
    if snapshot is None:
        return _EMPTY_WORKFLOW_GRAPH
    if filter:
        snapshot = _slice_workflow_graph_by_filter(snapshot, filter)
    return _strip_server_internal_fields(snapshot)


def _slice_workflow_graph_by_filter(snapshot: dict, filter: str) -> dict:
    """Prune the snapshot's `state` and `skeleton` to entries whose
    `filters` list contains `filter` (or is empty — empty means
    "shared / global", visible in every slice)."""
    state = snapshot.get("state", {})

    def visible(node_id):
        filters = state.get(node_id, {}).get("filters", [])
        return not filters or filter in filters

    new_state = {nid: m for nid, m in state.items() if visible(nid)}

    def prune(entries):
        out = []
        for e in entries:
            nid = e.get("id")
            if nid is not None and not visible(nid):
                continue
            new_e = dict(e)
            if "next" in new_e:
                new_e["next"] = prune(new_e["next"])
            if "definition" in new_e:
                new_e["definition"] = prune(new_e["definition"])
            out.append(new_e)
        return out

    return {
        "skeleton": prune(snapshot.get("skeleton", [])),
        "state": new_state,
    }


def _strip_server_internal_fields(snapshot: dict) -> dict:
    """Drop `filters` from every meta entry before returning to the
    frontend — the DagNode constructor rejects unknown fields.
    `filters` is used by `_slice_workflow_graph_by_filter` and is
    not consumed by the client. Recurses into `groupMeta`, which
    nests meta dicts for synthetic-group / subgraph entries."""
    def strip(meta):
        out = {k: v for k, v in meta.items() if k != "filters"}
        if "groupMeta" in out:
            out["groupMeta"] = {
                nid: strip(m) for nid, m in out["groupMeta"].items()
            }
        return out

    state = {nid: strip(m) for nid, m in snapshot.get("state", {}).items()}
    return {"skeleton": snapshot.get("skeleton", []), "state": state}


def _resolve_ref(ref):
    """Pull a workflow_graphs ContextRef's payload. Returns None
    if the ref itself is None or its underlying entry has been
    evicted from the cache — the index dict can outlive a wipe
    of the per-graph data."""
    if ref is None:
        return None
    try:
        return ref.get()
    except Exception:
        return None


def _json_default(obj):
    try:
        return str(obj)
    except Exception:
        return repr(obj)


@compute_graphs_bp.route("/api/node_status/<node_id>")
def node_status(node_id):
    return json.dumps(context[("node_status", node_id)], default=_json_default)


@compute_graphs_bp.route("/api/open_node_result_path/<node_id>")
def open_node_result_path(node_id):
    status = context[("node_status", node_id)]
    if "path" in status:
        filename = str(status["path"])
        print(f"### opening filename {filename}")
        subprocess.Popen(
            [
                "/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight",
                filename,
            ]
        )

        return Response(status=200)
