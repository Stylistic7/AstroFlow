import ray_local

import unittest
import ray

from cache import cache
from new_dependency_graph import Graph, run_graph


@ray.remote
def my_remote_task(arg, *args):
  print('running remote task', arg)
  return arg


@ray.remote(resources={'diskcache': 0.001})
def my_local_task(arg, *args):
  print('running local task', arg)
  return arg


@ray.remote(resources={'diskcache': 0.001})
def my_task_with_global(arg, *args):
  cache['foo'] = cache.get('foo', 0) + 1
  return arg


@ray.remote
def my_first_task(*args):
  return 1


@ray.remote
def my_later_task(ancestor_result):
  return ancestor_result + 1


@ray.remote
def scatter(*args):
  return Graph({gather: [(process, i) for i in range(7)]})


@ray.remote
def process(item, *args):
  return item


@ray.remote
def gather(results):
  return sum(results)


class TestDependencyGraph(unittest.TestCase):

  def test_simple_graph(self):
    A = (my_remote_task, 'A')
    B = (my_remote_task, 'B')
    C = (my_remote_task, 'C')
    D = (my_remote_task, 'D')

    graph = {D: {B, C}, C: {A}, B: {A}}
    results = run_graph(graph)

    expected_results = {
        (my_remote_task, 'A'): 'A',
        (my_remote_task, 'C'): 'C',
        (my_remote_task, 'B'): 'B',
        (my_remote_task, 'D'): 'D',
    }
    self.assertEqual(results, expected_results)

  def test_resources(self):
    A = (my_local_task, 'A')
    B = (my_remote_task, 'B')
    C = (my_local_task, 'C')
    D = (my_remote_task, 'D')

    graph = {D: {B, C}, C: {A}, B: {A}}
    results = run_graph(graph)

    expected_results = {
        (my_local_task, 'A'): 'A',
        (my_local_task, 'C'): 'C',
        (my_remote_task, 'B'): 'B',
        (my_remote_task, 'D'): 'D',
    }

    self.assertEqual(results, expected_results)

  def test_global_variable(self):
    cache['foo'] = 10

    A = (my_task_with_global, 'A')
    B = (my_task_with_global, 'B')
    C = (my_task_with_global, 'C')
    D = (my_task_with_global, 'D')

    graph = {D: {C}, C: {B}, B: {A}}
    results = run_graph(graph)

    self.assertEqual(cache['foo'], 14)

  def test_pass_down(self):
    A = my_first_task
    B = my_later_task

    graph = {B: {A}}
    results = run_graph(graph)

    self.assertEqual(results[A], 1)
    self.assertEqual(results[B], 2)

  def test_dynamic_subgraph(self):
    graph = {scatter: {}}
    results = run_graph(graph)

    self.assertEqual(results[scatter][gather], 21)


if __name__ == '__main__':
  unittest.main()
