import ray
from flask import Blueprint

from context import context
from dependency_graph.workflow_graph_adapter import to_workflow_graph
from dependency_graph.progress import graph_status_to_progress

compute_graphs_bp = Blueprint("compute_graphs", __name__)


@compute_graphs_bp.route("/compute_graph")
def compute_graph():
    return to_workflow_graph(context.compute_graphs)


@compute_graphs_bp.route("/compute_graph_status")
def compute_graph_status():
    return graph_status_to_progress(context.compute_graph_status)


@compute_graphs_bp.route("/workflow_graph/stack/<stack_id>")
def workflow_graph_for_stack(stack_id):
    return context.ui_stacks[stack_id]["workflow_graph"]


@compute_graphs_bp.route("/workflow_graph/target/<target_id>")
def workflow_graph_for_target(target_id):
    return context.ui_targets[target_id]["workflow_graph"]


@compute_graphs_bp.route("/workflow_graph/mosaic/<mosaic_id>")
def workflow_graph_for_mosaic(mosaic_id):
    return context.ui_mosaics[mosaic_id]["workflow_graph"]
