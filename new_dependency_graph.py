import graphlib
import ray


class ResultDict(dict):

  def __init__(self, *args, **kwargs):
    super(ResultDict, self).__init__(*args, **kwargs)

  def __getattr__(self, k):
    return self[k]

  def __getitem__(self, input):
    if type(input) is tuple:
      key, input_args = input
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
          == key._function.__name__) and (not input_args or args == input_args):
        return result


class Graph(dict):
  pass


def _unpack_node(node):
  if type(node) is tuple:
    return node
  else:
    return (node,)


def run_graph(graph, context=None):
  ts = graphlib.TopologicalSorter(graph)
  ts.prepare()

  results = {}

  while ts.is_active():
    to_run = []
    for node in ts.get_ready():
      to_run.append(node)

    def ancestor_results(node):
      # TODO: What happens to tasks that failed or didn't run?
      values = [results[node] for node in graph.get(node, [])]

      if len(values) == 0:
        return None

      if len(values) == 1:
        return values[0]

      return values

    def run(node):
      task, *args = _unpack_node(node)

      full_args = []
      if context is not None:
        full_args.append(context)

      full_args.extend(args)
      full_args.append(ancestor_results(node))

      return task.remote(*full_args)

    ref_to_node = {run(node): node for node in to_run}
    remaining_refs = list(ref_to_node.keys())

    while len(remaining_refs):
      ready_refs, remaining_refs = ray.wait(remaining_refs)

      for ref in ready_refs:
        node = ref_to_node[ref]
        if context is not None:
          context, result = ray.get(ref)
        else:
          result = ray.get(ref)

        # There are two possibilities here: the task can return a result,
        # or a (dynamically-generated) Graph to run as a sub-graph.
        if type(result) is Graph:
          # TODO: This blocks the parent graph; it should run in parallel
          # with any other runnable tasks in the parent.
          if context is not None:
            context, subgraph_result = run_graph(result, context)
          else:
            subgraph_result = run_graph(result)

          results[node] = ResultDict(subgraph_result)
          ts.done(node)
        else:
          results[node] = result
          ts.done(node)

  if context is not None:
    return context, results

  return results
