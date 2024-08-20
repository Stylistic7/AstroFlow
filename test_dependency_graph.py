from ray_infra import ray_local

import os
import unittest
import ray
import tempfile
import time

from cache import cache
from context import context
from dependency_graph import Graph, Node, GraphStatus, TopologicalSorterWithStatus, run_graph, make_graph_from_list


@ray.remote
def my_remote_task(arg):
  return arg


@ray.remote(resources={'diskcache': 0.001})
def my_local_task(arg):
  return arg


@ray.remote(resources={'diskcache': 0.001})
def my_task_with_global(arg):
  cache['foo'] = cache.get('foo', 0) + 1
  return arg


@ray.remote
def my_first_task():
  return 1


@ray.remote
def my_later_task(*, ancestor_results):
  return ancestor_results[0] + 1


@ray.remote
def my_last_task(*, ancestor_results):
  return ancestor_results[0] + 3


@ray.remote
def my_last_task_with_context(*, ancestor_results):
  return ancestor_results[0] + 3


@ray.remote
def increment_foo():
  context.foo += 1
  return context.foo


@ray.remote
def increment_bar():
  context.bar += 1
  return context.bar


@ray.remote
def increment_baz():
  context.baz += 1
  return context.baz


@ray.remote
def set_foo(value):
  context.foo = value
  return value


@ray.remote
def scatter():
  return Graph({gather: [(process, i) for i in range(7)]})


@ray.remote
def scatter_huge(count):
  return Graph({gather: [(process, i) for i in range(count)]})


@ray.remote
def process(item):
  return item


@ray.remote
def scatter_raises_exception():
  return Graph({gather: [(process_raises_exception, i) for i in range(11)]})


@ray.remote
def process_raises_exception(item):
  raise TimeoutError("oh noes")


@ray.remote
def gather(*, ancestor_results):
  return sum(ancestor_results)


@ray.remote
def passthrough(*, ancestor_results):
  return ancestor_results


@ray.remote
def build_subgraph(item):
  return Graph({passthrough: {(process, item)}})


@ray.remote
def gather_from_subgraphs(*, ancestor_results):
  return sum(r[0] for r in ancestor_results)


def scatter_in_subgraphs():
  return Graph(
      {gather_from_subgraphs: [(build_subgraph, i) for i in range(13)]})


@ray.remote
def scatter_with_context():
  s = context.starting_value or 0
  context.processed = []
  return Graph({gather_with_context: [(process, i) for i in range(s, s + 7)]})


@ray.remote
def gather_with_context(*, ancestor_results):
  context.processed += ancestor_results
  context.sum = sum(ancestor_results)
  return context.sum


@ray.remote
def noop():
  pass


@ray.remote
def produce_file(output_dir, filename):
  filename = os.path.join(output_dir, filename)

  # No work to do.
  if os.path.exists(filename):
    return None

  open(filename, 'w').write('foo')
  return filename


@ray.remote
def fn_with_multiple_inputs(a, b, c):
  return a + b + c


@ray.remote
def maybe_run_dynamic_graph(should_run):
  if should_run:
    return Graph({gather: [(process, i) for i in range(7)]})
  else:
    return {}


@ray.remote
def initial_task():
  return None


class TestDependencyGraph(unittest.TestCase):

  def test_simple_graph(self):
    A = (my_remote_task, 'A')
    B = (my_remote_task, 'B')
    C = (my_remote_task, 'C')
    D = (my_remote_task, 'D')

    graph = {D: {B, C}, C: {A}, B: {A}}
    results = run_graph(graph)

    self.assertEqual(results[A], 'A')
    self.assertEqual(results[B], 'B')
    self.assertEqual(results[C], 'C')
    self.assertEqual(results[D], 'D')

    self.assertEqual(results.num_nodes_run, 4)
    self.assertEqual(results.max_parallelism_seen, 2)

  def test_resources(self):
    A = (my_local_task, 'A')
    B = (my_remote_task, 'B')
    C = (my_local_task, 'C')
    D = (my_remote_task, 'D')

    graph = {D: {B, C}, C: {A}, B: {A}}
    results = run_graph(graph)

    self.assertEqual(results[A], 'A')
    self.assertEqual(results[B], 'B')
    self.assertEqual(results[C], 'C')
    self.assertEqual(results[D], 'D')

    self.assertEqual(results.num_nodes_run, 4)
    self.assertEqual(results.max_parallelism_seen, 2)

  def test_global_variable(self):
    cache['foo'] = 10

    A = (my_task_with_global, 'A')
    B = (my_task_with_global, 'B')
    C = (my_task_with_global, 'C')
    D = (my_task_with_global, 'D')

    graph = {D: {C}, C: {B}, B: {A}}
    results = run_graph(graph)

    self.assertEqual(cache['foo'], 14)

    self.assertEqual(results.num_nodes_run, 4)
    self.assertEqual(results.max_parallelism_seen, 1)

  def test_pass_down(self):
    A = my_first_task
    B = my_later_task
    C = my_last_task

    graph = {C: {B}, B: {A}}
    results = run_graph(graph)

    self.assertEqual(results[A], 1)
    self.assertEqual(results[B], 2)
    self.assertEqual(results[C], 5)

    self.assertEqual(results.num_nodes_run, 3)
    self.assertEqual(results.max_parallelism_seen, 1)

  def test_fanout(self):
    A = my_first_task
    B = my_later_task
    C = my_last_task

    graph = {C: {A}, B: {A}}
    results = run_graph(graph)

    self.assertEqual(results[A], 1)
    self.assertEqual(results[B], 2)
    self.assertEqual(results[C], 4)

    self.assertEqual(results.num_nodes_run, 3)
    self.assertEqual(results.max_parallelism_seen, 2)

  def test_list_of_tasks(self):
    graph = make_graph_from_list([my_first_task, my_later_task, my_last_task])

    results = run_graph(graph)
    self.assertEqual(results[my_last_task], 5)

    self.assertEqual(results.num_nodes_run, 6)
    self.assertEqual(results.max_parallelism_seen, 1)

  def test_with_exception(self):
    graph = {scatter_raises_exception: {}}

    results = run_graph(graph)
    self.assertTrue(isinstance(results[scatter_raises_exception], TypeError))

  def test_dynamic_subgraph(self):
    graph = {scatter: {}}

    results = run_graph(graph)

    self.assertEqual(results[scatter], 21)

    self.assertEqual(results.num_nodes_run, 9)
    self.assertEqual(results.max_parallelism_seen, 7)

  def test_dynamic_subgraph_completion(self):
    # scatter() produces a dynamic subgraph. my_last_task() should run only
    # after the entire subgraph is done, after the (implict) gather().
    #
    # Furthermore, my_last_task() should not "see" the subgraph at all; it
    # should be transparently handled by the runtime. my_last_task() should
    # see the result of gather().
    graph = {scatter: {}, my_last_task: {scatter}}
    results = run_graph(graph)

    self.assertEqual(results[scatter], 21)
    self.assertEqual(results[my_last_task], 24)

    self.assertEqual(results.num_nodes_run, 10)
    self.assertEqual(results.max_parallelism_seen, 7)

  def test_dynamic_subgraph_completion_with_peer(self):
    context.foo = 13

    graph = {my_last_task: [scatter, increment_foo], scatter: {}}
    results = run_graph(graph)

    self.assertEqual(context.foo, 14)
    self.assertEqual(results[scatter], 21)
    self.assertEqual(results[my_last_task], 24)

    self.assertEqual(results.num_nodes_run, 11)
    self.assertEqual(results.max_parallelism_seen, 7)

  def test_dynamic_subgraph_with_context_completion(self):
    # Same as `test_dynamic_subgraph_completion` but with the addition of
    # context. The context returned by the final node of the dynamic subgraph
    # should become the context for the successor node in the parent graph.
    graph = {
        scatter_with_context: {},
        my_last_task_with_context: {scatter_with_context}
    }

    results = run_graph(graph)

    self.assertEqual(results[scatter_with_context], 21)
    self.assertEqual(results[my_last_task_with_context], 24)

    self.assertEqual(results.num_nodes_run, 10)
    self.assertEqual(results.max_parallelism_seen, 7)

  def test_mutable_context(self):
    A = increment_foo
    B = increment_bar

    graph = {B: {A}}

    context.foo = 2
    context.bar = 3

    results = run_graph(graph)

    self.assertEqual(results[increment_foo], 3)
    self.assertEqual(results[increment_bar], 4)
    self.assertEqual(context.foo, 3)
    self.assertEqual(context.bar, 4)

    self.assertEqual(results.num_nodes_run, 2)
    self.assertEqual(results.max_parallelism_seen, 1)

  def test_dynamic_graph_with_context(self):
    graph = {scatter_with_context: {}}

    results = run_graph(graph)

    self.assertEqual(results[scatter_with_context], 21)
    self.assertEqual(sorted(context.processed), [0, 1, 2, 3, 4, 5, 6])
    self.assertEqual(context.sum, 21)

    self.assertEqual(results.num_nodes_run, 9)
    self.assertEqual(results.max_parallelism_seen, 7)

  def test_conditional_dynamic_graph(self):
    dynamic_graph_task = (maybe_run_dynamic_graph, True)
    graph = {dynamic_graph_task: {}}
    results = run_graph(graph)
    self.assertEqual(results[dynamic_graph_task], 21)

    self.assertEqual(results.num_nodes_run, 9)
    self.assertEqual(results.max_parallelism_seen, 7)

    dynamic_graph_task = (maybe_run_dynamic_graph, False)
    graph = {dynamic_graph_task: {}}
    results = run_graph(graph)
    self.assertEqual(results[dynamic_graph_task], {})

    self.assertEqual(results.num_nodes_run, 1)
    self.assertEqual(results.max_parallelism_seen, 1)

  def test_multiple_inputs(self):
    task = (fn_with_multiple_inputs, 1, 2, 3)
    graph = {task: {}}
    results = run_graph(graph)
    self.assertEqual(results[task], 6)

    self.assertEqual(results.num_nodes_run, 1)
    self.assertEqual(results.max_parallelism_seen, 1)

  def test_simple_graph_with_priority(self):
    # set_foo(20) with priority 1 will run first, followed by set_foo(10).
    # The final result will be 10 + 1.
    A = (set_foo, 1)
    B = Node(set_foo, 10, priority=2)
    C = Node(set_foo, 20, priority=1)
    D = increment_foo

    graph = {D: {B, C}, C: {A}, B: {A}}

    results = run_graph(graph)

    self.assertEqual(results[A], 1)
    self.assertEqual(results[B], 10)
    self.assertEqual(results[C], 20)
    self.assertEqual(results[D], 11)

    self.assertEqual(results.num_nodes_run, 4)
    self.assertEqual(results.max_parallelism_seen, 2)

  def test_duplicate_ancestors(self):
    # The increment_foo task should only run once, because A1 == A2.
    A1 = increment_foo
    A2 = increment_foo
    B = (process, 2)
    C = (process, 3)
    D = gather_with_context

    graph = {D: {B, C}, B: {A1}, C: {A2}}

    context.foo = 7
    context.processed = []

    results = run_graph(graph, debug=True)

    self.assertEqual(context.foo, 8)
    self.assertEqual(sorted(context.processed), [2, 3])
    self.assertEqual(context.sum, 5)
    self.assertEqual(results.num_nodes_run, 4)
    self.assertEqual(results.max_parallelism_seen, 2)

  def test_duplicate_nodes(self):
    A = Node(increment_foo, name='first_foo')
    B = Node(increment_foo, name='second_foo')

    graph = {A: {}, B: {}}

    context.foo = 7
    run_graph(graph)

    self.assertEqual(context.foo, 9)

  def test_scatter_in_subgraphs(self):
    graph = scatter_in_subgraphs()

    results = run_graph(graph)

    self.assertEqual(results[gather_from_subgraphs], 78)
    self.assertEqual(results.max_parallelism_seen, 13)

  def test_overhead(self):
    count = 1000
    graph = {(scatter_huge, count): {}}

    start = time.time()
    results = run_graph(graph)
    end = time.time()
    ms_per_fn = 1000.0 * (end - start) / count

    print(f'scatter_huge: each fn took {ms_per_fn} ms')
    self.assertLess(ms_per_fn, 2.0)

  def test_graph_with_status(self):
    graph = {'D': {'B', 'C'}, 'C': {'A'}, 'B': {'A'}}
    ts = TopologicalSorterWithStatus(graph)
    ts.prepare()

    ready_nodes = ts.get_ready()
    self.assertEqual(ready_nodes, ('A',))

    self.assertEqual(ts.ready_nodes, {'A'})
    self.assertEqual(ts.waiting_nodes, {'B', 'C', 'D'})
    self.assertEqual(ts.done_nodes, set())

    ts.done('A')
    self.assertEqual(ts.ready_nodes, set())
    self.assertEqual(ts.waiting_nodes, {'B', 'C', 'D'})
    self.assertEqual(ts.done_nodes, {'A'})

    ready_nodes = ts.get_ready()
    self.assertEqual(set(ready_nodes), {'B', 'C'})

    ts.done('B')
    self.assertEqual(ts.ready_nodes, {'C'})
    self.assertEqual(ts.waiting_nodes, {'D'})
    self.assertEqual(ts.done_nodes, {'A', 'B'})

    ts.done('C')
    self.assertEqual(ts.ready_nodes, set())
    self.assertEqual(ts.waiting_nodes, {'D'})
    self.assertEqual(ts.done_nodes, {'A', 'B', 'C'})

    ready_nodes = ts.get_ready()
    self.assertEqual(set(ready_nodes), {'D'})

    ts.done('D')
    self.assertEqual(ts.ready_nodes, set())
    self.assertEqual(ts.waiting_nodes, set())
    self.assertEqual(ts.done_nodes, {'A', 'B', 'C', 'D'})

  def test_graph_status_merging(self):
    status_1 = GraphStatus(waiting_nodes={'foo'}, done_nodes={'baz'})
    status_2 = GraphStatus(done_nodes={'bar'})

    status = status_1 | status_2
    self.assertEqual(
        status,
        GraphStatus(done_nodes={'bar', 'baz'},
                    ready_nodes=set(),
                    waiting_nodes={'foo'}))

  def test_status_fn(self):
    graph = {scatter: {}}

    final_status = None

    def function_name_from_node(node):
      return node.task._function.__name__

    def print_status(status):
      print('### status', status)

      print(f'### done {len(status.done_nodes)} '
            f'ready {len(status.ready_nodes)} '
            f'waiting {len(status.waiting_nodes)}')

      nonlocal final_status
      final_status = status

    results = run_graph(graph, status_fn=print_status)

    self.assertEqual(results[scatter], 21)

    self.assertEqual(results.num_nodes_run, 9)
    self.assertEqual(results.max_parallelism_seen, 7)

    self.assertEqual(len(final_status.done_nodes), 9)
    self.assertEqual(len(final_status.ready_nodes), 0)
    self.assertEqual(len(final_status.waiting_nodes), 0)


if __name__ == '__main__':
  unittest.main()
