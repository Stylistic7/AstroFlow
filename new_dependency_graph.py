import graphlib
import ray
import inspect
import copy


class Graph(dict):
  pass


def _unpack_node(node):
  if type(node) is tuple:
    return node
  else:
    return (node,)


class ResultDict(dict):

  def __init__(self, *args, **kwargs):
    super(ResultDict, self).__init__(*args, **kwargs)

  def __getattr__(self, k):
    return self[k]

  def __getitem__(self, input):
    if type(input) is tuple:
      key, *input_args = input
    else:
      key, input_args = input, []

    if type(key) is not ray.remote_function.RemoteFunction:
      return self.get(key)

    for node, result in self.items():
      task, *args = _unpack_node(node)
      if type(task) is not ray.remote_function.RemoteFunction:
        continue

      if (task._function.__module__ == key._function.__module__) and (
          task._function.__name__
          == key._function.__name__) and (input_args is None or
                                          args == input_args):
        return result

    raise KeyError(f'Key {key._function.__module__}.{key._function.__name__} '
                   f'input_args {input_args} not found')


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


def run_graph(graph, context=None):
  use_context = context is not None
  num_nodes_run = 0
  max_parallelism_seen = 0
  active = []

  def add_subgraph(graph, parent_ts=None, parent_node=None):

    def done_callback(subgraph_ts):
      print(f'### subgraph is done {subgraph_ts.graph}')
      if parent_ts is None:
        return

      parent_ts.done(parent_node)
      parent_ts.context = subgraph_ts.context
      parent_ts.results[parent_node] = subgraph_ts.final_result
      parent_ts.final_result = subgraph_ts.final_result

    print(f'### creating new subgraph {graph}')
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
    nodes = ts.graph.get(node, [])
    return [ts.results[node] for node in nodes]

  def make_run_fn(ts, node):
    task, *args = _unpack_node(node)
    print(f'### Making run fn for task {task._function.__name__}')

    full_args = []
    if use_context:
      full_args.append(ts.context)

    full_args.extend(args)

    sig = inspect.signature(task._function)
    if len(sig.parameters) > len(full_args):
      full_args.append(ancestor_results(ts, node))

    return task.remote(*full_args)

  def delete_finished_subgraphs():
    for ts in active:
      if not ts.is_active():
        if ts.done_callback:
          ts.done_callback()
        active.remove(ts)

  def get_ready_nodes():
    return [(ts, n) for ts in active for n in ts.get_ready()]

  toplevel_ts = add_subgraph(graph)

  while active:
    print(f'### top of loop, {len(active)} active graphs')
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
