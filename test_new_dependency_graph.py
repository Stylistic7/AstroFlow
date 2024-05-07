import ray_local

import os
import unittest
import ray
import tempfile
import functools

from cache import cache
from new_dependency_graph import Graph, Node, run_graph, make_graph_from_list


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
def my_later_task(ancestor_results):
  assert len(ancestor_results) == 1, f'ancestor_results are {ancestor_results}'
  return ancestor_results[0] + 1


@ray.remote
def my_last_task(ancestor_results):
  assert len(ancestor_results) == 1, f'ancestor_results are {ancestor_results}'
  return ancestor_results[0] + 3


@ray.remote
def my_last_task_with_context(context, ancestor_results):
  assert len(ancestor_results) == 1, f'ancestor_results are {ancestor_results}'
  return context, ancestor_results[0] + 3


@ray.remote
def increment_foo(context):
  context['foo'] += 1
  return context, context['foo']


@ray.remote
def increment_bar(context):
  context['bar'] += 1
  return context, context['bar']


@ray.remote
def set_foo(context, value):
  context['foo'] = value
  return context, value


@ray.remote
def scatter():
  return Graph({gather: [(process, i) for i in range(7)]})


@ray.remote
def process(item):
  return item


@ray.remote
def gather(results):
  return sum(results)


@ray.remote
def scatter_with_context(context):
  s = context.get('starting_value', 0)
  context['processed'] = []
  return context, Graph({
      gather_with_context: [(process_with_context, i) for i in range(s, s + 7)]
  })


@ray.remote
def process_with_context(context, item):
  return context, item


@ray.remote
def gather_with_context(context, results):
  context['processed'].extend(results)
  context['sum'] = sum(results)
  return context, context['sum']


@ray.remote
def produce_file(output_dir, filename):
  filename = os.path.join(output_dir, filename)

  # No work to do.
  if os.path.exists(filename):
    return None

  open(filename, 'w').write('foo')
  return filename


@ray.remote
def maybe_produce_file(output_dir, filename):
  filename = os.path.join(output_dir, filename)
  open(filename, 'w').write('foo')
  return filename


maybe_produce_file.artifact = lambda o, f: os.path.join(o, f)
maybe_produce_file.artifact_exists = lambda f: os.path.exists(f)


@ray.remote
def merge_files(output_dir, filenames):
  # No work to do.
  if all(f == None for f in filenames):
    return None

  merged_filename = os.path.join(output_dir, 'merged.txt')
  open(merged_filename, 'w').write(str(filenames))
  return merged_filename


@ray.remote
def maybe_merge_files(output_dir, filenames):
  merged_filename = os.path.join(output_dir, 'merged.txt')
  open(merged_filename, 'w').write(str(filenames))
  return merged_filename


maybe_merge_files.artifact = lambda o: os.path.join(o, 'merged.txt')
maybe_merge_files.artifact_exists = lambda f: os.path.exists(f)


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
def initial_task(context):
  return context, None


@ray.remote
def start_parallel_scatter(context, starting_value):
  context['starting_value'] = starting_value
  return context, Graph({scatter_with_context: {}})


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

  def test_list_of_tasks(self):
    graph = make_graph_from_list([my_first_task, my_later_task, my_last_task])

    results = run_graph(graph)
    self.assertEqual(results[my_last_task], 5)

    self.assertEqual(results.num_nodes_run, 3)
    self.assertEqual(results.max_parallelism_seen, 1)

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

  def test_dynamic_subgraph_with_context_completion(self):
    # Same as `test_dynamic_subgraph_completion` but with the addition of
    # context. The context returned by the final node of the dynamic subgraph
    # should become the context for the successor node in the parent graph.
    graph = {
        scatter_with_context: {},
        my_last_task_with_context: {scatter_with_context}
    }
    context = {}

    context, results = run_graph(graph, context)

    self.assertEqual(results[scatter_with_context], 21)
    self.assertEqual(results[my_last_task_with_context], 24)

    self.assertEqual(results.num_nodes_run, 10)
    self.assertEqual(results.max_parallelism_seen, 7)

  def test_mutable_context(self):
    A = increment_foo
    B = increment_bar

    graph = {B: {A}}
    context = {'foo': 2, 'bar': 3}

    context, results = run_graph(graph, context)

    self.assertEqual(results[increment_foo], 3)
    self.assertEqual(results[increment_bar], 4)
    self.assertEqual(context, {'foo': 3, 'bar': 4})

    self.assertEqual(results.num_nodes_run, 2)
    self.assertEqual(results.max_parallelism_seen, 1)

  def test_dynamic_graph_with_context(self):
    graph = {scatter_with_context: {}}
    context = {}

    context, results = run_graph(graph, context)

    self.assertEqual(results[scatter_with_context], 21)
    self.assertEqual(sorted(context['processed']), [0, 1, 2, 3, 4, 5, 6])
    self.assertEqual(context['sum'], 21)

    self.assertEqual(results.num_nodes_run, 9)
    self.assertEqual(results.max_parallelism_seen, 7)

  def test_parallel_dynamic_graphs_with_context(self):
    # This test starts two scatter operations with different values stored in
    # the context object in each operation.
    graph = {
        initial_task: {},
        (start_parallel_scatter, 100): {initial_task},
        (start_parallel_scatter, 200): {initial_task}
    }

    context = {'foo': 'bar'}
    context, results = run_graph(graph, context)

    self.assertEqual(results[(start_parallel_scatter, 100)], 721)
    self.assertEqual(results[(start_parallel_scatter, 200)], 1421)

    self.assertEqual(results.num_nodes_run, 21)
    self.assertEqual(results.max_parallelism_seen, 14)

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
    context = {'foo': None}
    context, results = run_graph(graph, context)

    self.assertEqual(results[A], 1)
    self.assertEqual(results[B], 10)
    self.assertEqual(results[C], 20)
    self.assertEqual(results[D], 11)

    self.assertEqual(results.num_nodes_run, 4)
    self.assertEqual(results.max_parallelism_seen, 2)

  def test_merge_files(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      graph = {
          (merge_files, tmp_dir): [
              (produce_file, tmp_dir, f'file_{i}.txt') for i in range(7)
          ]
      }

      # None of the files exist on the first run, so the `merge_files`
      # function should do work, and return the filename it wrote.
      results = run_graph(graph)
      self.assertEqual(len(os.listdir(tmp_dir)), 8)
      self.assertEqual(results[(merge_files, tmp_dir)],
                       os.path.join(tmp_dir, 'merged.txt'))

      self.assertEqual(results.num_nodes_run, 8)
      self.assertEqual(results.max_parallelism_seen, 7)

      # On the second run, all of the files exist, so the `merge_files`
      # function should have no work to do, and return None.
      results = run_graph(graph)
      self.assertEqual(len(os.listdir(tmp_dir)), 8)
      self.assertEqual(results[(merge_files, tmp_dir)], None)

      self.assertEqual(results.num_nodes_run, 8)
      self.assertEqual(results.max_parallelism_seen, 7)


if __name__ == '__main__':
  unittest.main()
