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


class ResultDict(dict):
  def __init__(self, *args, **kwargs):
    super(ResultDict, self).__init__(*args, **kwargs)

    self.setdefault('ran', True)
    self.setdefault('run_successors', False)

  def __getattr__(self, k):
    return self[k]


def run_graph(graph):
  ts = graphlib.TopologicalSorter(graph)
  ts.prepare()

  results = {}

  def should_run(node):
    task, arg = node

    if hasattr(task, 'should_run') and not task.should_run(arg):
      return False

    if node in graph and all(
        not results[(task._function.__name__, arg)].run_successors
        for task, arg in graph[node]):
      return False

    return True

  while ts.is_active():
    to_run = []
    for node in ts.get_ready():
      if should_run(node):
        to_run.append(node)
      else:
        task, arg = node
        results[(task._function.__name__, arg)] = ResultDict(ran=False)
        ts.done(node)

    ref_to_node = {task.remote(arg): (task, arg) for task, arg in to_run}
    remaining_refs = list(ref_to_node.keys())

    while len(remaining_refs):
      ready_refs, remaining_refs = ray.wait(remaining_refs)

      for ref in ready_refs:
        node = ref_to_node[ref]
        task, arg = node
        result = ray.get(ref)
        assert type(result) is ResultDict
        results[(task._function.__name__, arg)] = result
        ts.done(node)

  return results