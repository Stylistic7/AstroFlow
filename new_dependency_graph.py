from collections import defaultdict
import graphlib
import ray

# Some design philosophy:
#
# This dependency graph has no persistent state. It does not save anything to
# disk.
#
# The entire compute graph is formed in full on every run.
#
# Each node consists of two functions:
#
# node.remote(): a function that does the node's work, which can be done
# locally or remotely. This must be a pure function (e.g. it may not use the
# diskcache) because remote workers are stateless.
#
# node.should_run(): an optional function that returns True if the node should
# run. The default should_run() function always returns True. This function
# runs locally, and may check the diskcache.
#
# A node's remote() function is therefore called when this Boolean expression
# is True:
#
#   node.should_run() and (node has no predecessors or any predecessor says
#     successors should run).


# TODO: If explicit local or remote scheduling is supported, then it'll be
# easy to break up big functions (like stacking.run()) into smaller pieces.
#
# The `should_run` mechanism could maybe be eliminated. `should_run` is instead
# just task that runs locally, and either enables or disables the downstream
# task. This is conceptually simpler.
#
# Within Ray, I can also just specify that some tasks require custom resources,
# and the custom resource could be the diskcache that contains image hashes
# and plate solutions. That could be really nice...
#
# On the other hand, the plate_solutions are about 125 MB for 20k images,
# and image_hashes only about 10 MB. This is such a small amount of data that
# passing it around the network is ~free, compared to the amount of image data
# that has to be read from NAS (easily 10-100 GB per stack). There can still be
# scheduling overhead, etc. that would unnecessarily slow down the entire
# compute graph.
#
# So there are basically three options:
#
# - Provide a should_run() function that is explicitly outside the Ray system.
#   It only ever runs locally, and select whether or not tasks will even be
#   sent to Ray at all. This is extremely fast, but a little ugly.
#
# - Split tasks into `should_run` and `run` sub-tasks. The `should_run` task
#   enables or disables the `run` task. This is conceptually beautiful, and
#   is likely fast enough. The overhead to run `should_run` remotely is large,
#   but Ray should automatically avoid sending large amounts of data over the
#   network.
#
# - Split tasks as before, but also set a custom resource (e.g. diskcache) for
#   steps that use it: `should_run`, but not `run`. This is conceptually nice,
#   and guarantees that the diskcache-related tasks will only be run on the
#   head node.
class ResultDict(dict):

  def __init__(self, *args, **kwargs):
    super(ResultDict, self).__init__(*args, **kwargs)

    self.setdefault('ran', False)
    self.setdefault('run_successors', False)

  def __getattr__(self, k):
    return self[k]

  def __getitem__(self, input):
    if type(input) is tuple:
      key, input_arg = input
    else:
      key, input_arg = input, None

    if type(key) is not ray.remote_function.RemoteFunction:
      return self.get(key)

    for node, result in self.result.items():
      task, arg = _unpack_node(node)
      if type(task) is not ray.remote_function.RemoteFunction:
        continue

      if (task._function.__module__ == key._function.__module__) and (
          task._function.__name__
          == key._function.__name__) and (not input_arg or arg == input_arg):
        return result


class FromAncestors:
  pass


class Graph(dict):
  pass


def _unpack_node(node):
  if type(node) is tuple:
    return node
  else:
    return (node, FromAncestors)


def run_graph(graph):
  ts = graphlib.TopologicalSorter(graph)
  ts.prepare()

  results = {}

  def should_run(node):
    task, arg = _unpack_node(node)

    if hasattr(task, 'should_run') and not task.should_run(arg):
      return False

    if node in graph and len(graph[node]) and all(
        not results[node].run_successors for node in graph[node]):
      return False

    return True

  while ts.is_active():
    to_run = []
    for node in ts.get_ready():
      if should_run(node):
        to_run.append(node)
      else:
        results[node] = ResultDict()
        ts.done(node)

    def ancestor_results(node):
      # TODO: What happens to tasks that failed or didn't run?
      values = [
          results[node].result
          for node in graph.get(node, [])
          if 'result' in results[node]
      ]

      if len(values) == 0:
        return None

      if len(values) == 1:
        return values[0]

      return values

    def run(node):
      task, arg = _unpack_node(node)
      if arg == FromAncestors:
        return task.remote(ancestor_results(node))
      else:
        return task.remote(arg, ancestor_results(node))

    ref_to_node = {run(node): node for node in to_run}
    remaining_refs = list(ref_to_node.keys())

    while len(remaining_refs):
      ready_refs, remaining_refs = ray.wait(remaining_refs)

      for ref in ready_refs:
        node = ref_to_node[ref]
        result = ray.get(ref)

        # There are two possibilities here: the task can return a ResultDict,
        # or a (dynamically-generated) Graph to run as a sub-graph.
        if type(result) is ResultDict:
          results[node] = result
          ts.done(node)
        elif type(result) is Graph:
          subgraph_result = run_graph(result)
          results[node] = ResultDict(ran=True,
                                     run_successors=True,
                                     result=subgraph_result)
          ts.done(node)
        else:
          assert False, f"Task return type {type(result)} is unsupported"

  return results
