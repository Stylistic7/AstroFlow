import unittest
import os
import ray
import graphlib

import new_dependency_graph as dg

# This Ray cluster will be able to run tasks that are explicitly local, as well
# as any task with no specific resources.
ray.init(resources={'diskcache': 1})


@ray.remote
def my_remote_task(arg):
  print('running remote task', arg)
  return dg.ResultDict(run_successors=True, result=arg)


@ray.remote(resources={'diskcache': 1})
def my_local_task(arg):
  print('running local task', arg)
  return dg.ResultDict(run_successors=True, result=arg)


class TestDependencyGraph(unittest.TestCase):
  def test_simple_graph(self):
    A = (my_remote_task, "A")
    B = (my_remote_task, "B")
    C = (my_remote_task, "C")
    D = (my_remote_task, "D")

    my_remote_task.should_run = lambda arg: True

    graph = {D: {B, C}, C: {A}, B: {A}}
    results = dg.run_graph(graph)

    print('### results', results)

    expected_results = {
        ('my_remote_task', 'A'): {
            'ran': True,
            'result': 'A',
            'run_successors': True,
        },
        ('my_remote_task', 'C'): {
            'ran': True,
            'result': 'C',
            'run_successors': True,
        },
        ('my_remote_task', 'B'): {
            'ran': True,
            'result': 'B',
            'run_successors': True,
        },
        ('my_remote_task', 'D'): {
            'ran': True,
            'result': 'D',
            'run_successors': True,
        }
    }

    self.assertEqual(results, expected_results)

  def test_should_run_fn(self):
    A = (my_remote_task, "A")
    B = (my_remote_task, "B")
    C = (my_remote_task, "C")
    D = (my_remote_task, "D")

    my_remote_task.should_run = lambda arg: arg != 'B'

    graph = {D: {B, C}, C: {A}, B: {A}}
    results = dg.run_graph(graph)

    expected_results = {
        ('my_remote_task', 'A'): {
            'ran': True,
            'result': 'A',
            'run_successors': True
        },
        ('my_remote_task', 'C'): {
            'ran': True,
            'result': 'C',
            'run_successors': True
        },
        ('my_remote_task', 'B'): {
            'ran': False,
            'run_successors': False
        },
        ('my_remote_task', 'D'): {
            'ran': True,
            'result': 'D',
            'run_successors': True
        }
    }

    self.assertEqual(results, expected_results)

  def test_no_predecessors_ran(self):
    A = (my_remote_task, "A")
    B = (my_remote_task, "B")
    C = (my_remote_task, "C")
    D = (my_remote_task, "D")

    my_remote_task.should_run = lambda arg: arg not in ['B', 'C']

    graph = {D: {B, C}, C: {A}, B: {A}}
    results = dg.run_graph(graph)

    expected_results = {
        ('my_remote_task', 'A'): {
            'ran': True,
            'result': 'A',
            'run_successors': True
        },
        ('my_remote_task', 'C'): {
            'ran': False,
            'run_successors': False
        },
        ('my_remote_task', 'B'): {
            'ran': False,
            'run_successors': False
        },
        ('my_remote_task', 'D'): {
            'ran': False,
            'run_successors': False
        }
    }

    self.assertEqual(results, expected_results)

  def test_resources(self):
    A = (my_local_task, "A")
    B = (my_remote_task, "B")
    C = (my_local_task, "C")
    D = (my_remote_task, "D")

    my_remote_task.should_run = lambda arg: arg not in ['B']
    my_local_task.should_run = lambda arg: arg not in ['C']

    graph = {D: {B, C}, C: {A}, B: {A}}
    results = dg.run_graph(graph)

    expected_results = {
        ('my_local_task', 'A'): {
            'ran': True,
            'result': 'A',
            'run_successors': True
        },
        ('my_local_task', 'C'): {
            'ran': False,
            'run_successors': False
        },
        ('my_remote_task', 'B'): {
            'ran': False,
            'run_successors': False
        },
        ('my_remote_task', 'D'): {
            'ran': False,
            'run_successors': False
        }
    }

    self.assertEqual(results, expected_results)


if __name__ == "__main__":
  unittest.main()