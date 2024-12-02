import ray
import inspect
from copy import copy
from functools import reduce
from operator import or_
import time
from typing import Callable, Any
from ray.remote_function import RemoteFunction
import structlog

from dependency_graph import Node, Graph, CanonicalGraph, GraphStatus, ResultDict, TopologicalSorterWithStatus, GenericRemoteFunction, canonicalize_graph
from dependency_graph.logging import LogContainer
from .debugging import print_graph, node_to_string, node_to_string_tuple

MIN_STATUS_UPDATE_INTERVAL_SECS: float = 5.0

StatusFn = Callable[[GraphStatus], Any]

# TODO: Make a type alias for AncestorResults


def matching_nodes(graph: CanonicalGraph,
                   fn: GenericRemoteFunction) -> list[Node]:

  def _matches(node: Node) -> bool:
    assert isinstance(fn, RemoteFunction)
    assert isinstance(node.task, RemoteFunction)
    return (node.task._function.__module__ == fn._function.__module__ and
            node.task._function.__name__ == fn._function.__name__)

  return [node for node in graph if _matches(node)]


def run_graph(graph: Graph,
              status_fn: StatusFn | None = None,
              status_interval: float = MIN_STATUS_UPDATE_INTERVAL_SECS,
              debug: bool = False) -> ResultDict:
  num_nodes_run: int = 0
  max_parallelism_seen: int = 0
  active: list[TopologicalSorterWithStatus] = []
  completed: list[TopologicalSorterWithStatus] = []
  last_status_time: float = 0

  def add_subgraph(
      graph: Graph,
      parent_ts: TopologicalSorterWithStatus | None = None,
      parent_node: Node | None = None) -> TopologicalSorterWithStatus:

    def done_callback(subgraph_ts: TopologicalSorterWithStatus) -> None:
      if debug:
        print(f'Subgraph is done {subgraph_ts.graph}')
      if parent_ts is None:
        return

      if parent_node is not None:
        parent_ts.done(parent_node)
        parent_ts.results[parent_node] = subgraph_ts.final_result
        parent_ts.logs[parent_node] = subgraph_ts.logs

      parent_ts.final_result = subgraph_ts.final_result

    canonical_graph = canonicalize_graph(graph)

    if debug:
      print(f'### Creating new subgraph:')
      print_graph(canonical_graph)

    dc: Callable[[], None] = lambda: done_callback(ts)
    ts: TopologicalSorterWithStatus = TopologicalSorterWithStatus(
        canonical_graph, done_callback=dc)
    ts.prepare()
    active.append(ts)

    return ts

  def ancestor_results(ts: TopologicalSorterWithStatus,
                       node: Node) -> list[Any]:
    nodes: set[Node] = ts.graph.get(node, set())
    if debug:
      assert isinstance(node.task, RemoteFunction)
      print(f'Node with task {node.task._function.__name__} has '
            f'{len(nodes)} ancestors')

    return [ts.results[node] for node in nodes]

  def make_run_fn(ts: TopologicalSorterWithStatus, node: Node) -> Any:
    if debug:
      assert isinstance(node.task, RemoteFunction)
      print(f'Making run fn for task {node.task._function.__name__}')

    # TODO: Is this copy necessary?
    full_args = []
    full_args.extend(node.args)

    assert isinstance(node.task, RemoteFunction)
    sig = inspect.signature(node.task._function)
    if 'ancestor_results' in sig.parameters:
      try:
        return node.task.remote(*full_args,
                                ancestor_results=ancestor_results(ts, node))
      except Exception as e:
        print(f'### node {node_to_string(node)} raised exception {e}')
        raise e
    else:
      return node.task.remote(*full_args)

  def delete_finished_subgraphs() -> None:
    for ts in active:
      if not ts.is_active():
        if ts.done_callback:
          ts.done_callback()
        active.remove(ts)
        completed.append(ts)

  def get_ready_nodes() -> list[tuple[TopologicalSorterWithStatus, Node]]:
    return [(ts, n)
            for ts in active
            for n in sorted(ts.get_ready(), key=lambda n: n.priority)]

  def get_status() -> GraphStatus:
    return reduce(or_, [ts.get_status() for ts in active + completed],
                  GraphStatus())

  toplevel_ts = add_subgraph(graph)

  while active:
    ready = get_ready_nodes()
    max_parallelism_seen = max(len(ready), max_parallelism_seen)

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
        assert isinstance(node.task, RemoteFunction)
        print(f'Task {node.task._function.__name__} finished')

      num_nodes_run += 1

      # Maybe call the user's status_fn, but rate limit the frequency of these
      # calls, which may be expensive.
      current_time = time.time()
      if status_fn and current_time > (last_status_time + status_interval):
        status_fn(get_status())
        last_status_time = current_time

      try:
        result = ray.get(ready_refs[0])

        # If the function captured logs, split them off the returned result.
        log = []
        if isinstance(result, tuple) and isinstance(result[0], LogContainer):
          log, result = result
          print(f'### node {node_to_string(node)} produced '
                f'a log of length {len(log)}')

        # There are two possibilities here: the task can return a result,
        # or a (dynamically-generated) Graph to run as a sub-graph.
        if isinstance(result, Graph):
          add_subgraph(result, ts, node)
        else:
          ts.results[node] = result
          ts.logs[node] = log
          ts.final_result = result
          ts.done(node)
      except ray.exceptions.RayTaskError as e:
        print(f'Node {node} raised exception {e}')
        ts.results[node] = e.cause
        ts.logs[node] = []
        ts.final_result = e.cause
        ts.done(node)

    delete_finished_subgraphs()

  # All subgraphs, including the given graph, must be done here.
  assert len(active) == 0

  if status_fn:
    status_fn(get_status())

  toplevel_ts.results.num_nodes_run = num_nodes_run
  toplevel_ts.results.max_parallelism_seen = max_parallelism_seen

  return toplevel_ts.results, toplevel_ts.logs
