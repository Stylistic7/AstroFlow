import unittest
import os
import graphlib

import ray

ray.init(resources={'diskcache': 1}, ignore_reinit_error=True)

import globals
import new_dependency_graph as dg


@ray.remote
def my_remote_task(arg, *args):
  print('running remote task', arg)
  return dg.ResultDict(run_successors=True, result=arg)


@ray.remote(resources={'diskcache': 0.001})
def my_local_task(arg, *args):
  print('running local task', arg)
  return dg.ResultDict(run_successors=True, result=arg)


@ray.remote
def my_task_with_global(arg, *args):
  print('running task with global', arg)
  globals.plus_equals('foo', 1)
  return dg.ResultDict(run_successors=True, result=arg)


@ray.remote
def my_first_task(*args):
  return dg.ResultDict(run_successors=True, counter=1)


@ray.remote
def my_later_task(arg, ancestor_results):
  print('running later task', arg, 'ancestor_results', ancestor_results)
  return dg.ResultDict(
      run_successors=True,
      counter=next(iter(ancestor_results.values()))['counter'] + 1)


class TestDependencyGraph(unittest.TestCase):
  def test_simple_graph(self):
    A = (my_remote_task, "A")
    B = (my_remote_task, "B")
    C = (my_remote_task, "C")
    D = (my_remote_task, "D")

    my_remote_task.should_run = lambda arg: True

    graph = {D: {B, C}, C: {A}, B: {A}}
    results = dg.run_graph(graph)

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

  def test_global_variable(self):
    globals.set_global('foo', 0)

    A = (my_task_with_global, "A")
    B = (my_task_with_global, "B")
    C = (my_task_with_global, "C")
    D = (my_task_with_global, "D")

    graph = {D: {C}, C: {B}, B: {A}}
    results = dg.run_graph(graph)

    self.assertEqual(globals.get_global('foo'), 4)

  def test_pass_down(self):
    A = (my_first_task, None)
    B = (my_later_task, None)

    graph = {B: {A}}
    results = dg.run_graph(graph)

    self.assertEqual(results[(my_first_task._function.__name__, None)].counter,
                     1)
    self.assertEqual(results[(my_later_task._function.__name__, None)].counter,
                     2)


if __name__ == "__main__":
  unittest.main()