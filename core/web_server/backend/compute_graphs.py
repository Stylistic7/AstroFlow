import ray
from flask import Blueprint

from context import context
from dependency_graph import run_graph
from dependency_graph.workflow_graph_adapter import to_workflow_graph
from stacks.compute_graph import build_stacking_subgraph
from targets.compute_graph import build_target_subgraph
from mosaics.compute_graph import build_mosaic_subgraph, stacks_for_mosaic, targets_for_mosaic, build_placement_subgraph
from previews.types import QualityLevel
from dependency_graph.progress import graph_status_to_progress

compute_graphs_bp = Blueprint("compute_graphs", __name__)

@compute_graphs_bp.route('/compute_graphs')
def compute_graphs():
  return to_workflow_graph(context.compute_graphs)


@compute_graphs_bp.route('/compute_graph_status')
def compute_graph_status():
  return graph_status_to_progress(context.compute_graph_status)
  

@compute_graphs_bp.route('/workflow_graph/stack/<stack_id>')
def workflow_graph_for_stack(stack_id):

  graph_status = None

  def graph_update_fn(g):
    nonlocal graph_status
    graph_status = g

  graph = ray.get(build_stacking_subgraph.remote(context.stacks[stack_id]))

  results, _ = run_graph(graph,
                         graph_update_fn=graph_update_fn,
                         dry_run=True,
                         debug=True)

  return to_workflow_graph(graph_status)


@compute_graphs_bp.route('/workflow_graph/target/<target_id>')
def workflow_graph_for_target(target_id):

  graph_status = None

  def graph_update_fn(g):
    nonlocal graph_status
    graph_status = g

  graph = ray.get(build_target_subgraph.remote(context.targets[target_id]))

  results, _ = run_graph(graph,
                         graph_update_fn=graph_update_fn,
                         dry_run=True,
                         debug=True)

  return to_workflow_graph(graph_status)


@compute_graphs_bp.route('/workflow_graph/mosaic/<mosaic_id>')
def workflow_graph_for_mosaic(mosaic_id):

  graph_status = None

  def graph_update_fn(g):
    nonlocal graph_status
    graph_status = g

  mosaic = context.mosaics[mosaic_id]

  graph = ray.get(
      build_mosaic_subgraph.remote(
          QualityLevel.MASTER, mosaic,
          stacks_for_mosaic(mosaic, context.stacks, context.targets),
          targets_for_mosaic(mosaic, context.targets)))

  results, _ = run_graph(graph,
                         graph_update_fn=graph_update_fn,
                         dry_run=True,
                         debug=True)

  return to_workflow_graph(graph_status)
