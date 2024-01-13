import unittest
import os
import ray
import graphlib

import new_dependency_graph as dg

ray.init()


@ray.remote
def my_task(arg):
  print('running task', arg)
  return dg.ResultDict(run_successors=True, result=arg)


class TestDependencyGraph(unittest.TestCase):
  def test_simple_graph(self):
    A = (my_task, "A")
    B = (my_task, "B")
    C = (my_task, "C")
    D = (my_task, "D")

    my_task.should_run = lambda arg: True

    graph = {D: {B, C}, C: {A}, B: {A}}
    results = dg.run_graph(graph)

    print('### results', results)

    expected_results = {
        ('my_task', 'A'): {
            'ran': True,
            'result': 'A',
            'run_successors': True,
        },
        ('my_task', 'C'): {
            'ran': True,
            'result': 'C',
            'run_successors': True,
        },
        ('my_task', 'B'): {
            'ran': True,
            'result': 'B',
            'run_successors': True,
        },
        ('my_task', 'D'): {
            'ran': True,
            'result': 'D',
            'run_successors': True,
        }
    }

    self.assertEqual(results, expected_results)

  def test_should_run_fn(self):
    A = (my_task, "A")
    B = (my_task, "B")
    C = (my_task, "C")
    D = (my_task, "D")

    my_task.should_run = lambda arg: arg != 'B'

    graph = {D: {B, C}, C: {A}, B: {A}}
    results = dg.run_graph(graph)

    expected_results = {
        ('my_task', 'A'): {
            'ran': True,
            'result': 'A',
            'run_successors': True
        },
        ('my_task', 'C'): {
            'ran': True,
            'result': 'C',
            'run_successors': True
        },
        ('my_task', 'B'): {
            'ran': False,
            'run_successors': False
        },
        ('my_task', 'D'): {
            'ran': True,
            'result': 'D',
            'run_successors': True
        }
    }

    self.assertEqual(results, expected_results)

  def test_no_predecessors_ran(self):
    A = (my_task, "A")
    B = (my_task, "B")
    C = (my_task, "C")
    D = (my_task, "D")

    my_task.should_run = lambda arg: arg not in ['B', 'C']

    graph = {D: {B, C}, C: {A}, B: {A}}
    results = dg.run_graph(graph)

    expected_results = {
        ('my_task', 'A'): {
            'ran': True,
            'result': 'A',
            'run_successors': True
        },
        ('my_task', 'C'): {
            'ran': False,
            'run_successors': False
        },
        ('my_task', 'B'): {
            'ran': False,
            'run_successors': False
        },
        ('my_task', 'D'): {
            'ran': False,
            'run_successors': False
        }
    }

    self.assertEqual(results, expected_results)


if __name__ == "__main__":
  unittest.main()