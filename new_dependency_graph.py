import graphlib
import ray
import inspect


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


def make_graph_from_list(tasks):
  graph = {}
  prev_task = None
  for task in tasks:
    if prev_task:
      graph[task] = {prev_task}
    else:
      graph[task] = {}

    prev_task = task

  return graph


def run_graph(graph, context=None):
  num_nodes_run = 0
  max_parallelism_seen = 0

  graph_for_ts = {}
  context_for_ts = {}
  results_for_ts = {}

  def add_topological_sorter(graph, context, results):
    ts = graphlib.TopologicalSorter(graph)
    ts.prepare()
    graph_for_ts[ts] = graph
    context_for_ts[ts] = context
    results_for_ts[ts] = results

  results = ResultDict()
  add_topological_sorter(graph, context, results)

  while any(ts.is_active() for ts in graph_for_ts.keys()):
    to_run = []
    for ts, node in [
        (ts, n) for ts in graph_for_ts.keys() for n in ts.get_ready()
    ]:
      task, *args = _unpack_node(node)
      if hasattr(task, 'artifact'):
        assert hasattr(task, 'artifact_exists')
        artifact = task.artifact(*args)
        if task.artifact_exists(artifact):
          results[node] = artifact
          ts.done(node)
          continue

      to_run.append((ts, node))

    max_parallelism_seen = max(len(to_run), max_parallelism_seen)

    def ancestor_results(ts, node):
      nodes = graph_for_ts[ts].get(node, [])
      values = [results_for_ts[ts][node] for node in nodes]

      if len(values) == 1:
        return values[0]

      return values

    def run(ts, node):
      task, *args = _unpack_node(node)

      full_args = []
      if context is not None:
        full_args.append(context_for_ts[ts])

      full_args.extend(args)

      sig = inspect.signature(task._function)
      if len(sig.parameters) > len(full_args):
        full_args.append(ancestor_results(ts, node))

      return task.remote(*full_args)

    ref_to_ts_and_node = {run(ts, node): (ts, node) for ts, node in to_run}
    remaining_refs = list(ref_to_ts_and_node.keys())

    while len(remaining_refs):
      ready_refs, remaining_refs = ray.wait(remaining_refs)

      for ref in ready_refs:
        ts, node = ref_to_ts_and_node[ref]
        num_nodes_run += 1

        if context is not None:
          context_for_ts[ts], result = ray.get(ref)
          context = context_for_ts[ts]
        else:
          result = ray.get(ref)

        # There are two possibilities here: the task can return a result,
        # or a (dynamically-generated) Graph to run as a sub-graph.
        if type(result) is Graph:
          subgraph_results = ResultDict()
          results_for_ts[ts][node] = subgraph_results
          add_topological_sorter(result, context, subgraph_results)
        else:
          results_for_ts[ts][node] = result

        ts.done(node)

  results['num_nodes_run'] = num_nodes_run
  results['max_parallelism_seen'] = max_parallelism_seen

  if context is not None:
    return context, results

  return results
