import ray
from flask import Blueprint, Response
import json
import subprocess

from context import context

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


@compute_graphs_bp.route("/api/workflow_graph/stack/<stack_id>/<quality>")
def workflow_graph_for_stack(stack_id, quality):
    wf = context["workflow_graphs"] or {}
    ref = wf.get(("stack", stack_id, quality))
    return ref.get() if ref is not None else _EMPTY_WORKFLOW_GRAPH


@compute_graphs_bp.route("/api/workflow_graph/target/<target_id>/<quality>")
def workflow_graph_for_target(target_id, quality):
    wf = context["workflow_graphs"] or {}
    ref = wf.get(("target", target_id, quality))
    return ref.get() if ref is not None else _EMPTY_WORKFLOW_GRAPH


@compute_graphs_bp.route("/api/workflow_graph/mosaic/<mosaic_id>/<filter>/<quality>")
def workflow_graph_for_mosaic(mosaic_id, filter, quality):
    wf = context["workflow_graphs"] or {}
    ref = wf.get(("mosaic", mosaic_id, filter, quality))
    return ref.get() if ref is not None else _EMPTY_WORKFLOW_GRAPH


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
