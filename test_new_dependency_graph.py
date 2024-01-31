# yapf: disable
import os
os.environ["ASTROFLOW_MOCK_CACHE"] = "true"

from cache import cache

import tempfile
import ray
ray.init(
  address='local',
  namespace='AstroFlow',
  num_cpus=24,
  resources={"diskcache": 1},
  ignore_reinit_error=True,
  runtime_env={"env_vars": {
    "ASTROFLOW_MOCK_CACHE": "true"
  }},
  storage=tempfile.TemporaryDirectory().name
)

import ray_diskcache
global_var_actor = ray_diskcache.GlobalVarActor.options(name='ray_diskcache',
                                          namespace='AstroFlow',
                                          get_if_exists=True).remote()
# yapf: enable

import unittest
import graphlib

import ray_diskcache
from new_dependency_graph import Graph, ResultDict, run_graph


@ray.remote
def my_remote_task(arg, *args):
  print('running remote task', arg)
  return ResultDict(ran=True, run_successors=True, result=arg)


@ray.remote(resources={'diskcache': 0.001})
def my_local_task(arg, *args):
  print('running local task', arg)
  return ResultDict(ran=True, run_successors=True, result=arg)


@ray.remote(resources={'diskcache': 0.001})
def my_task_with_global(arg, *args):
  cache['foo'] = cache.get('foo', 0) + 1
  return ResultDict(ran=True, run_successors=True, result=arg)


@ray.remote
def my_first_task(*args):
  return ResultDict(ran=True, run_successors=True, result=1)


@ray.remote
def my_later_task(ancestor_result):
  return ResultDict(ran=True, run_successors=True, result=ancestor_result + 1)


@ray.remote
def scatter(*args):
  return Graph({gather: [(process, i) for i in range(7)]})


@ray.remote
def process(item, *args):
  return ResultDict(ran=True, run_successors=True, result=item)


@ray.remote
def gather(results):
  return ResultDict(ran=True, run_successors=True, result=sum(results))


class TestDependencyGraph(unittest.TestCase):

  def test_simple_graph(self):
    A = (my_remote_task, "A")
    B = (my_remote_task, "B")
    C = (my_remote_task, "C")
    D = (my_remote_task, "D")

    my_remote_task.should_run = lambda arg: True

    graph = {D: {B, C}, C: {A}, B: {A}}
    results = run_graph(graph)

    expected_results = {
        (my_remote_task, 'A'): {
            'ran': True,
            'result': 'A',
            'run_successors': True,
        },
        (my_remote_task, 'C'): {
            'ran': True,
            'result': 'C',
            'run_successors': True,
        },
        (my_remote_task, 'B'): {
            'ran': True,
            'result': 'B',
            'run_successors': True,
        },
        (my_remote_task, 'D'): {
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
    results = run_graph(graph)

    expected_results = {
        (my_remote_task, 'A'): {
            'ran': True,
            'result': 'A',
            'run_successors': True
        },
        (my_remote_task, 'C'): {
            'ran': True,
            'result': 'C',
            'run_successors': True
        },
        (my_remote_task, 'B'): {
            'ran': False,
            'run_successors': False
        },
        (my_remote_task, 'D'): {
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
    results = run_graph(graph)

    expected_results = {
        (my_remote_task, 'A'): {
            'ran': True,
            'result': 'A',
            'run_successors': True
        },
        (my_remote_task, 'C'): {
            'ran': False,
            'run_successors': False
        },
        (my_remote_task, 'B'): {
            'ran': False,
            'run_successors': False
        },
        (my_remote_task, 'D'): {
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
    results = run_graph(graph)

    expected_results = {
        (my_local_task, 'A'): {
            'ran': True,
            'result': 'A',
            'run_successors': True
        },
        (my_local_task, 'C'): {
            'ran': False,
            'run_successors': False
        },
        (my_remote_task, 'B'): {
            'ran': False,
            'run_successors': False
        },
        (my_remote_task, 'D'): {
            'ran': False,
            'run_successors': False
        }
    }

    self.assertEqual(results, expected_results)

  def test_global_variable(self):
    cache['foo'] = 10

    A = (my_task_with_global, "A")
    B = (my_task_with_global, "B")
    C = (my_task_with_global, "C")
    D = (my_task_with_global, "D")

    graph = {D: {C}, C: {B}, B: {A}}
    results = run_graph(graph)

    self.assertEqual(cache['foo'], 14)

  def test_pass_down(self):
    A = my_first_task
    B = my_later_task

    graph = {B: {A}}
    results = run_graph(graph)

    # TODO: Make a better API for getting results.
    self.assertEqual(results[A].result, 1)
    self.assertEqual(results[B].result, 2)

  def test_dynamic_task_creation(self):
    graph = {scatter: {}}
    results = run_graph(graph)

    self.assertEqual(results[scatter][gather].result, 21)


if __name__ == "__main__":
  unittest.main()
