import ray
import inspect
from copy import copy
from functools import reduce
from operator import or_

from dependency_graph import Graph, GraphStatus, ResultDict, TopologicalSorterWithStatus, canonicalize_node
from .debugging import print_graph


def matching_nodes(graph, fn):

  def _matches(node):
    return (node.task._function.__module__ == fn._function.__module__ and
            node.task._function.__name__ == fn._function.__name__)

  return [node for node in graph if _matches(node)]


def run_graph(graph, status_fn=None, debug=False):
  num_nodes_run = 0
  max_parallelism_seen = 0
  active = []
  completed = []

  # TODO: Maybe support nodes with one ancestor, like this {done: process},
  # instead of requiring {done: {process}}.
  def canonicalize_graph(graph):
    output = Graph({})
    for node, dependents in graph.items():
      output[canonicalize_node(node)] = {
          canonicalize_node(x) for x in dependents
      }
    return output

  def add_subgraph(graph, parent_ts=None, parent_node=None):

    def done_callback(subgraph_ts):
      if debug:
        print(f'Subgraph is done {subgraph_ts.graph}')
      if parent_ts is None:
        return

      parent_ts.done(parent_node)
      parent_ts.results[parent_node] = subgraph_ts.final_result
      parent_ts.final_result = subgraph_ts.final_result

    graph = canonicalize_graph(graph)

    if debug:
      print(f'### Creating new subgraph:')
      print_graph(graph)

    ts = TopologicalSorterWithStatus(graph)
    ts.prepare()
    ts.done_callback = lambda subgraph_ts=ts: done_callback(subgraph_ts)
    ts.graph = graph
    ts.results = ResultDict()
    ts.final_result = None
    active.append(ts)

    return ts

  def ancestor_results(ts, node):
    nodes = ts.graph.get(node, [])
    if debug:
      print(f'Node with task {node.task._function.__name__} has '
            f'{len(nodes)} ancestors')

    return [ts.results[node] for node in nodes]

  def make_run_fn(ts, node):
    if debug:
      print(f'Making run fn for task {node.task._function.__name__}')

    # TODO: Is this copy necessary?
    full_args = []
    full_args.extend(node.args)

    sig = inspect.signature(node.task._function)
    if 'ancestor_results' in sig.parameters:
      return node.task.remote(*full_args,
                              ancestor_results=ancestor_results(ts, node))
    else:
      return node.task.remote(*full_args)

  def delete_finished_subgraphs():
    for ts in active:
      if not ts.is_active():
        if ts.done_callback:
          ts.done_callback()
        active.remove(ts)
        completed.append(ts)

  def get_ready_nodes():
    return [(ts, n)
            for ts in active
            for n in sorted(ts.get_ready(), key=lambda n: n.priority)]

  def get_status():
    return reduce(or_, [ts.get_status() for ts in active + completed],
                  GraphStatus())

  toplevel_ts = add_subgraph(graph)

  while active:
    ready = get_ready_nodes()
    max_parallelism_seen = max(len(ready), max_parallelism_seen)

    if status_fn:
      status_fn(get_status())

    if debug:
      print(f'Top of loop, {len(active)} active graphs, '
            f'{len(ready)} ready nodes')

    # This starts all ready tasks in parallel.
    ref_to_ts_and_node = {
        make_run_fn(ts, node): (ts, node) for ts, node in ready
    }
    remaining_refs = list(ref_to_ts_and_node.keys())

    while len(remaining_refs):
      ready_refs, remaining_refs = ray.wait(remaining_refs, num_returns=1)
      assert len(ready_refs) == 1
      ts, node = ref_to_ts_and_node[ready_refs[0]]
      if debug:
        print(f'Task {node.task._function.__name__} finished')

      num_nodes_run += 1

      try:
        result = ray.get(ready_refs[0])

        # There are two possibilities here: the task can return a result,
        # or a (dynamically-generated) Graph to run as a sub-graph.
        if isinstance(result, Graph):
          add_subgraph(result, ts, node)
        else:
          ts.results[node] = result
          ts.final_result = result
          ts.done(node)
      except ray.exceptions.RayTaskError as e:
        print(f'Node {node} raised exception {e}')
        ts.results[node] = e.cause
        ts.final_result = e.cause
        ts.done(node)

    delete_finished_subgraphs()

  # All subgraphs, including the given graph, must be done here.
  assert len(active) == 0

  if status_fn:
    status_fn(get_status())

  toplevel_ts.results['num_nodes_run'] = num_nodes_run
  toplevel_ts.results['max_parallelism_seen'] = max_parallelism_seen

  return toplevel_ts.results
