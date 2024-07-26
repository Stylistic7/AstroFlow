import graphlib
import ray
import inspect
from dataclasses import dataclass, field
from copy import copy
from functools import reduce
from operator import or_


class Graph(dict):
  pass


class Node():

  def __init__(self, task, *args, name=None, priority=1):
    self.task = task
    self.name = name
    self.priority = priority
    self.args = args

  def __eq__(self, other):
    return (self.task._function.__module__ == other.task._function.__module__
            and
            self.task._function.__name__ == other.task._function.__name__ and
            self.name == other.name and self.priority == other.priority and
            self.args == other.args)

  def __hash__(self):
    return hash((self.task, self.name, self.priority, self.args))


def _unpack_node(node):

  if type(node) is Node:
    return node
  elif type(node) is tuple:
    return Node(*node)
  else:
    return Node(node)


class ResultDict(dict):

  def __init__(self, *args, **kwargs):
    super(ResultDict, self).__init__(*args, **kwargs)

  def __getattr__(self, k):
    return self[k]

  def __getitem__(self, key):
    if type(key) is str:
      return self.get(key)

    key = _unpack_node(key)

    for node, result in self.items():
      node = _unpack_node(node)
      if node == key:
        return result

    raise KeyError(
        f'Key {key.task._function.__module__}.{key.task._function.__name__} '
        f'args {key.args} not found')


@ray.remote
def done(ancestor_results):
  return ancestor_results[0]


def make_graph_from_list(tasks):
  graph = Graph()
  prev_done_node = None
  for i, task in enumerate(tasks):
    done_node = Node(done, name=f'done_{i}')
    graph[done_node] = [task]

    if prev_done_node:
      graph[task] = [prev_done_node]

    prev_done_node = done_node

  return graph


def print_result_dict(results, indent=0):
  for node, result in results.items():
    node = _unpack_node(node)

    if type(node.task) is str:
      print(f'{" " * indent}{node.task} = {result}')
      continue

    if type(result) is not ResultDict:
      print(
          f'{" " * indent}{node.task._function.__name__} {node.args}: {result}')
    else:
      print(f'{" " * indent}{node.task._function.__name__} {node.args}:')
      print_result_dict(result, indent + 2)


def print_graph(graph):
  if not graph:
    print('Compute graph is empty')
    return

  print(f'Printing compute graph with {len(graph)} edges:')
  for k, v in graph.items():

    def type_or_value(x):
      if type(x) in [int, float, str]:
        return f"'{str(x)}'"

      return str(type(x))

    def tuple_values(tup):
      return ', '.join([type_or_value(t) for t in tup])

    def to_string(node):
      node = _unpack_node(node)
      return f'({node.task._function.__name__}, {tuple_values(node.args)})'

    if type(v) in [list, set]:
      strings = [to_string(i) for i in v]
      print(f'{to_string(k)}: [{", ".join(strings)}]')
      continue

    assert False, f'Node dependencies have unsupported type {type(v)}: {v}'

  print('Done printing compute graph')


@dataclass
class GraphStatus:
  done_nodes: set[Node] = field(default_factory=set)
  ready_nodes: set[Node] = field(default_factory=set)
  waiting_nodes: set[Node] = field(default_factory=set)

  def __or__(self, rhs):
    self.done_nodes |= rhs.done_nodes
    self.ready_nodes |= rhs.ready_nodes
    self.waiting_nodes |= rhs.waiting_nodes

    return self


class TopologicalSorterWithStatus(graphlib.TopologicalSorter):

  def __init__(self, graph):
    super().__init__(graph)

    self.done_nodes = set()
    self.ready_nodes = set()
    self.waiting_nodes = set(graphlib.TopologicalSorter(graph).static_order())

  def get_status(self):
    return GraphStatus(self.done_nodes, self.ready_nodes, self.waiting_nodes)

  def get_ready(self):
    ready_nodes = super().get_ready()

    self.ready_nodes = set(ready_nodes)
    self.waiting_nodes -= self.ready_nodes

    return ready_nodes

  def done(self, node):
    super().done(node)
    self.done_nodes.add(node)
    if node in self.ready_nodes:
      self.ready_nodes.remove(node)


def run_graph(graph, status_fn=None, debug=False):
  num_nodes_run = 0
  max_parallelism_seen = 0
  active = []
  completed = []

  def canonicalize_graph(graph):
    output = {}
    for node, dependents in graph.items():
      output[_unpack_node(node)] = [_unpack_node(x) for x in dependents]
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
      print(f'Creating new subgraph {graph}')

    ts = TopologicalSorterWithStatus(graph)
    ts.prepare()
    ts.done_callback = lambda subgraph_ts=ts: done_callback(subgraph_ts)
    ts.graph = graph
    ts.results = ResultDict()
    ts.final_result = None
    active.append(ts)

    return ts

  def ancestor_results(ts, node):
    node = _unpack_node(node)
    nodes = ts.graph.get(node, [])
    if debug:
      print(f'Node with task {node.task._function.__name__} has '
            f'{len(nodes)} ancestors')

    return [ts.results[node] for node in nodes]

  def make_run_fn(ts, node):
    node = _unpack_node(node)
    if debug:
      print(f'Making run fn for task {node.task._function.__name__}')

    # TODO: Is this copy necessary?
    full_args = []
    full_args.extend(node.args)

    # TODO: This confuses optional arguments with ancestor results;
    # maybe there's a cleaner way to do this?
    sig = inspect.signature(node.task._function)
    if len(sig.parameters) > len(full_args):
      full_args.append(ancestor_results(ts, node))

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
        unpacked = _unpack_node(node)
        print(f'Task {unpacked.task._function.__name__} finished')

      num_nodes_run += 1

      try:
        result = ray.get(ready_refs[0])

        # There are two possibilities here: the task can return a result,
        # or a (dynamically-generated) Graph to run as a sub-graph.
        if type(result) is Graph:
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

  if status_fn:
    status_fn(get_status())

  toplevel_ts.results['num_nodes_run'] = num_nodes_run
  toplevel_ts.results['max_parallelism_seen'] = max_parallelism_seen

  return toplevel_ts.results
