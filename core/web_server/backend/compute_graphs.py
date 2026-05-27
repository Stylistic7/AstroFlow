import ray
from flask import Blueprint, Response, request
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
    stays filter-unaware.

    Query param `?raw=1` skips the `sanitize_for_display` transform
    pipeline and serializes the trace's raw graph (transparent ops
    visible, shared Nodes not duplicated). Re-traces the pipeline
    on the spot, so it's slower but only used for debugging when
    the sanitized view looks wrong."""
    if not pipeline:
        return _EMPTY_WORKFLOW_GRAPH
    raw = request.args.get("raw") == "1"
    if raw:
        snapshot = _build_raw_workflow_graph(mosaic_id, pipeline)
    else:
        wf = context["workflow_graphs"] or {}
        snapshot = _resolve_ref(wf.get(("mosaic", mosaic_id, pipeline)))
    if snapshot is None:
        return _EMPTY_WORKFLOW_GRAPH
    if filter:
        snapshot = _slice_workflow_graph_by_filter(snapshot, filter)
    return _strip_server_internal_fields(snapshot)


def _build_raw_workflow_graph(mosaic_id: str, pipeline_name: str) -> dict | None:
    """Re-trace `pipeline_name` for the mosaic and serialize WITHOUT
    the sanitize-for-display transforms. Returns `None` if the
    mosaic or pipeline isn't found.

    Local imports keep the heavy module deps off the import-time
    fast path."""
    from dependency_graph.graph_runtime import GraphRuntime
    from dependency_graph.workflow_graph_adapter import to_workflow_graph
    from decisions.settings import ProcessingConfig
    from mosaics.types import Mosaic
    from pipelines.registry import get_pipeline

    mosaic_ref = (context["mosaics"] or {}).get(mosaic_id)
    if mosaic_ref is None:
        return None
    mosaic = mosaic_ref.get()
    settings = context["settings"]
    config = ProcessingConfig(
        calibration_stacks=context["calibration_stacks"] or {},
        development_settings=settings.development_settings,
        min_frames_in_stack=settings.clustering_settings.min_frames_in_stack,
    )
    try:
        runner = get_pipeline(pipeline_name, processes=Mosaic)
    except Exception:
        return None
    graph = runner(mosaic, config)
    if graph is None:
        return None
    rt = GraphRuntime(graph)
    return to_workflow_graph(rt, raw=True)


def _slice_workflow_graph_by_filter(snapshot: dict, filter: str) -> dict:
    """Prune the snapshot's `state` and `skeleton` to entries whose
    `filters` list contains `filter` (or is empty — empty means
    "shared / global", visible in every slice).

    Group entries' `state` is RE-AGGREGATED from their surviving
    children, so a group that contained a FAILED descendant only
    reachable through a hidden filter shows the state of what's
    still visible (instead of the pre-computed cross-filter
    aggregate the snapshot baked in)."""
    state = snapshot.get("state", {})

    def visible(meta):
        filters = (meta or {}).get("filters", [])
        return not filters or filter in filters

    def aggregate(states):
        if not states:
            return "NO_STATE_RUNTIME"
        if "FAILED" in states:
            return "FAILED"
        if any(s not in ("SUCCEEDED", "SKIPPED", "NO_STATE_RUNTIME") for s in states):
            return "RUNNING"
        succeeded = any(s in ("SUCCEEDED", "SKIPPED") for s in states)
        not_run = any(s == "NO_STATE_RUNTIME" for s in states)
        if succeeded and not_run:
            return "RUNNING"
        if succeeded:
            return "SUCCEEDED"
        return "NO_STATE_RUNTIME"

    def slice_meta(m):
        """Return a copy of `m` with its `groupMeta` filtered to
        visible children, and its `state` recomputed from those
        surviving children's states. Returns None if `m` itself
        isn't visible."""
        if not visible(m):
            return None
        out = dict(m)
        gm = m.get("groupMeta")
        if gm:
            new_gm = {}
            for k, child in gm.items():
                sliced = slice_meta(child)
                if sliced is not None:
                    new_gm[k] = sliced
            out["groupMeta"] = new_gm
            if new_gm:
                out["state"] = aggregate([c.get("state") for c in new_gm.values()])
        return out

    new_state: dict[str, dict] = {}
    for nid, m in state.items():
        sliced = slice_meta(m)
        if sliced is not None:
            new_state[nid] = sliced

    def visible_id(node_id):
        return visible(state.get(node_id, {}))

    def prune(entries):
        out = []
        for e in entries:
            nid = e.get("id")
            if nid is not None and not visible_id(nid):
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
    # Skeleton ids are mangled `<node_to_id>@<scope_suffix>` so the
    # same underlying Node can render in multiple consumer groups
    # (visual duplication). Status entries are keyed by the
    # canonical `node_to_id` alone — one Node has one status — so
    # strip the scope suffix before lookup.
    node_id = node_id.partition("@")[0]
    return json.dumps(context[("node_status", node_id)], default=_json_default)


@compute_graphs_bp.route("/api/open_node_result_path/<node_id>")
def open_node_result_path(node_id):
    node_id = node_id.partition("@")[0]
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
