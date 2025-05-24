import ray
from flask import Blueprint, Response
import datetime
import threading

from context import context
from compute_graph import build_compute_graph
from dependency_graph import Graph, run_graph, graph_status_to_progress
from dependency_graph.debugging import node_to_string
from stacks.compute_graph import build_stacking_subgraph
from web_server.backend.announcer import announcer
from web_server.backend.object_cache import stacks, targets, mosaics

compute_graphs_bp = Blueprint("compute_graphs", __name__)

compute_graph_status = {}
workflow_graph = {'skeleton': [], 'state': {}}


def populate_context():
  # TODO: Put these knobs in the UI.
  context.cosine_similarity_threshold = 0.99999
  context.min_component_members = 3
  context.max_run_gap = datetime.timedelta(hours=8)
  context.min_frames_in_run = 3
  context.min_frames_in_stack = 3
  context.subframe_thumbnail_width = 1200
  context.master_thumbnail_width = 2160


# TODO: Pull all of this transform code out into a separate module and test
# it properly.
def status_fn(status):
  global compute_graph_status
  print(f'### status_fn {len(status.ready_nodes)} ready '
        f'{len(status.waiting_nodes)} waiting '
        f'{len(status.done_nodes)} done')

  compute_graph_status = graph_status_to_progress(status)
  announcer.announce('compute_graph_status')

  if len(status.ready_nodes) == 0 and len(status.waiting_nodes) == 0:
    print('### Compute graph is done!')


def workflow_graph_fn(wg):
  global workflow_graph
  workflow_graph = wg


# TODO: Maybe rename this.
@compute_graphs_bp.route('/compute_graph')
def get_compute_graph():
  return workflow_graph


@compute_graphs_bp.route('/compute_graph_status')
def get_compute_graph_status():
  return compute_graph_status


# TODO: Make sure a second graph can't be started when one is already in
# progress.
#
# TODO: Something in here is causing `download_asiair()` to be called and
# preventing this from returning a response immediately.
@compute_graphs_bp.route('/run_graph')
def run_compute_graph():

  populate_context()

  # TODO: Put these in the UI.
  context.ignore_cached_results = True
  context.dangerously_accept_existing_files = True

  graph = build_compute_graph()

  def _run_graph(*args, **kwargs):
    # TODO: Perhaps the results and logs should be saved incrementally,
    # whenever a node finishes?
    results, logs = run_graph(*args, **kwargs)

    print('### Writing graph results and logs into context')
    context.last_run_results = {
        node_to_string(node): result
        for node, result in results.flatten().items()
    }

    context.last_run_logs = {
        node_to_string(node): log for node, log in logs.flatten().items()
    }

  print('### Running compute graph!')
  thread = threading.Thread(target=_run_graph,
                            args=(graph,),
                            kwargs={
                                'status_fn': status_fn,
                                'workflow_graph_fn': workflow_graph_fn,
                                'debug': False
                            })
  thread.daemon = True
  thread.start()

  return Response(status=204)


@compute_graphs_bp.route('/workflow_graph/stack/<stack_id>')
def workflow_graph_for_stack(stack_id):

  workflow_graph = None

  def workflow_graph_fn(wg):
    nonlocal workflow_graph
    workflow_graph = wg


#   graph = ray.get(build_stacking_subgraph.remote(stacks()[stack_id]))

  graph = Graph({(build_stacking_subgraph, stacks()[stack_id]): {}})
  results, _ = run_graph(graph,
                         workflow_graph_fn=workflow_graph_fn,
                         dry_run=True,
                         debug=True)

  return workflow_graph
