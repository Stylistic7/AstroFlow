import graphlib
import ray
import inspect
import copy


class Graph(dict):
  pass


class Node():

  def __init__(self, task, *args, priority=1):
    self.task = task
    self.priority = priority
    self.args = args

  def __eq__(self, other):
    return (self.task._function.__module__ == other.task._function.__module__
            and
            self.task._function.__name__ == other.task._function.__name__ and
            self.priority == other.priority and self.args == other.args)

  def __hash__(self):
    return hash((self.task, self.priority, self.args))


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


def make_graph_from_list(tasks):
  graph = Graph()
  prev_task = None
  for task in tasks:
    if prev_task:
      graph[task] = {prev_task}
    else:
      graph[task] = {}

    prev_task = task

  return graph


def print_result_dict(results, indent=0):
  for node, result in results.items():
    if type(node) is tuple:
      task, *args = node
    else:
      task, args = node, []

    if type(task) is str:
      print(f'{" " * indent}{task} = {result}')
      continue

    if type(result) is not ResultDict:
      print(f'{" " * indent}{task._function.__name__} {args}: {result}')
    else:
      print(f'{" " * indent}{task._function.__name__} {args}:')
      print_result_dict(result, indent + 2)


def run_graph(graph, context=None, debug=False):
  use_context = context is not None
  num_nodes_run = 0
  max_parallelism_seen = 0
  active = []

  def canonicalize_graph(graph):
    output = {}
    for node, dependents in graph.items():
      output[_unpack_node(node)] = [_unpack_node(x) for x in dependents]
    return output

  def add_subgraph(graph, parent_ts=None, parent_node=None):

    def done_callback(subgraph_ts):
      if debug:
        print(f'subgraph is done {subgraph_ts.graph}')
      if parent_ts is None:
        return

      parent_ts.done(parent_node)
      parent_ts.context = subgraph_ts.context
      parent_ts.results[parent_node] = subgraph_ts.final_result
      parent_ts.final_result = subgraph_ts.final_result

    graph = canonicalize_graph(graph)

    if debug:
      print(f'creating new subgraph {graph}')

    ts = graphlib.TopologicalSorter(graph)
    ts.prepare()
    ts.done_callback = lambda subgraph_ts=ts: done_callback(subgraph_ts)
    ts.graph = graph
    ts.context = context if parent_ts is None else parent_ts.context
    ts.results = ResultDict()
    ts.final_result = None
    active.append(ts)

    return ts

  def ancestor_results(ts, node):
    node = _unpack_node(node)
    nodes = ts.graph.get(node, [])
    return [ts.results[node] for node in nodes]

  def make_run_fn(ts, node):
    node = _unpack_node(node)
    if debug:
      print(f'Making run fn for task {node.task._function.__name__}')

    full_args = []
    if use_context:
      full_args.append(ts.context)

    full_args.extend(node.args)

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

  def get_ready_nodes():
    return [(ts, n)
            for ts in active
            for n in sorted(ts.get_ready(), key=lambda n: n.priority)]

  toplevel_ts = add_subgraph(graph)

  while active:
    if debug:
      print(f'top of loop, {len(active)} active graphs')

    ready = get_ready_nodes()
    max_parallelism_seen = max(len(ready), max_parallelism_seen)

    ref_to_ts_and_node = {
        make_run_fn(ts, node): (ts, node) for ts, node in ready
    }
    remaining_refs = list(ref_to_ts_and_node.keys())

    while len(remaining_refs):
      ready_refs, remaining_refs = ray.wait(remaining_refs, num_returns=1)
      assert len(ready_refs) == 1
      ts, node = ref_to_ts_and_node[ready_refs[0]]
      if use_context:
        ts.context, result = ray.get(ready_refs[0])
      else:
        result = ray.get(ready_refs[0])

      num_nodes_run += 1

      # There are two possibilities here: the task can return a result,
      # or a (dynamically-generated) Graph to run as a sub-graph.
      if type(result) is Graph:
        add_subgraph(result, ts, node)
      else:
        ts.results[node] = result
        ts.final_result = result
        ts.done(node)

    delete_finished_subgraphs()

  toplevel_ts.results['num_nodes_run'] = num_nodes_run
  toplevel_ts.results['max_parallelism_seen'] = max_parallelism_seen

  if use_context:
    return toplevel_ts.context, toplevel_ts.results
  else:
    return toplevel_ts.results
