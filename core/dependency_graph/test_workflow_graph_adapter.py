from ray_infra import ray_local

from typing import TypeVar
import unittest
import ray
import structlog
import pprint

import compute_graph
from context import context, get_cache
from dependency_graph import Graph, run_graph
from testing import make_test_context
from dependency_graph.logging import capture_logs
from mosaics.orion_mosaic import add_orion_mosaic_images
from cache.cached_decorator import make_cached_decorator, Injected
from hashing import hash_call
from testing.sidecar_hasher import SidecarHasher

cache = get_cache()
log = structlog.get_logger()

T = TypeVar('T')

sidecar_hasher = SidecarHasher()
cached = make_cached_decorator(cache=cache,
                               content_hasher=sidecar_hasher,
                               call_hasher=hash_call,
                               debug=True)


@ray.remote
@capture_logs
def first_remote_task(arg: T) -> T:
  log.info(f'first_remote_task {arg}')
  return arg


@ray.remote
@capture_logs
def second_remote_task(arg: T) -> T:
  log.info(f'second_remote_task {arg}')
  return arg


@ray.remote
@capture_logs
def process(item: T) -> T:
  log.info(f'process {item}')
  return item


@ray.remote
@capture_logs
def gather(*ancestor_results: list[int]) -> int:
  result = sum(ancestor_results)
  log.info(f'gather result {result}')
  return result


@ray.remote
def build_dynamic_graph() -> Graph:
  return Graph({gather: [(process, i) for i in range(3)]})


@cached.remote
def task_raises_exception() -> None:
  context.num_task_raises_exception_runs = context.num_task_raises_exception_runs + 1
  raise ValueError("oh no!")


class TestWorkflowGraphAdapter(unittest.TestCase):

  def setUp(self) -> None:
    # Start with an empty context.
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def tearDown(self) -> None:
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_simple_graph(self) -> None:
    A = (first_remote_task, 'A')
    B = (first_remote_task, 'B')
    C = (second_remote_task, 'C')
    D = (second_remote_task, 'D')

    graph = Graph({D: {B, C}, C: {A}, B: {A}, A: {}})

    workflow_graph = None

    def workflow_graph_fn(wg):
      nonlocal workflow_graph
      workflow_graph = wg

    results, logs = run_graph(graph, workflow_graph_fn=workflow_graph_fn)

    pprint.pprint(workflow_graph, indent=2, width=200)

    expected_workflow_graph = {
        'skeleton': [{
            'id': 'Node(fn=first_remote_task, args=A, priority=1)',
            'next': [{
                'id': 'Node(fn=first_remote_task, args=B, priority=1)',
                'next': [{
                    'id': 'Node(fn=second_remote_task, args=D, priority=1)',
                    'type': 'execution'
                }],
                'type': 'execution'
            }, {
                'id': 'Node(fn=second_remote_task, args=C, priority=1)',
                'next': [{
                    'id': 'Node(fn=second_remote_task, args=D, priority=1)',
                    'type': 'execution'
                }],
                'type': 'execution'
            }],
            'type': 'execution'
        }],
        'state': {
            'Node(fn=first_remote_task, args=A, priority=1)': {
                'displayName': 'first_remote_task',
                'state': 'SUCCEEDED',
            },
            'Node(fn=first_remote_task, args=B, priority=1)': {
                'displayName': 'first_remote_task',
                'state': 'SUCCEEDED',
            },
            'Node(fn=second_remote_task, args=C, priority=1)': {
                'displayName': 'second_remote_task',
                'state': 'SUCCEEDED',
            },
            'Node(fn=second_remote_task, args=D, priority=1)': {
                'displayName': 'second_remote_task',
                'state': 'SUCCEEDED',
            }
        }
    }

    self.assertEqual(workflow_graph, expected_workflow_graph)

  def test_graph_with_subgraph(self) -> None:
    A = (first_remote_task, 'A')
    B = (first_remote_task, 'B')
    C = (second_remote_task, 'C')
    D = build_dynamic_graph
    E = (second_remote_task, 'D')

    graph = Graph({E: {D}, D: {B, C}, C: {A}, B: {A}})

    workflow_graph = None

    def workflow_graph_fn(wg):
      nonlocal workflow_graph
      workflow_graph = wg

    results, logs = run_graph(graph, workflow_graph_fn=workflow_graph_fn)

    pprint.pprint(workflow_graph, indent=2, width=200)

    expected_workflow_graph = {
        'skeleton': [{
            'id': 'Node(fn=first_remote_task, args=A, priority=1)',
            'next': [{
                'id': 'Node(fn=first_remote_task, args=B, priority=1)',
                'next': [{
                    'definition': [{
                        'id': 'Node(fn=process, args=0, priority=1)',
                        'next': [{
                            'id': 'Node(fn=gather, priority=1)',
                            'type': 'execution'
                        }],
                        'type': 'execution'
                    }, {
                        'id': 'Node(fn=process, args=1, priority=1)',
                        'next': [{
                            'id': 'Node(fn=gather, priority=1)',
                            'type': 'execution'
                        }],
                        'type': 'execution'
                    }, {
                        'id': 'Node(fn=process, args=2, priority=1)',
                        'next': [{
                            'id': 'Node(fn=gather, priority=1)',
                            'type': 'execution'
                        }],
                        'type': 'execution'
                    }],
                    'id': 'Node(fn=build_dynamic_graph, priority=1)',
                    'type': 'group'
                }],
                'type': 'execution'
            }, {
                'id': 'Node(fn=second_remote_task, args=C, priority=1)',
                'next': [{
                    'definition': [{
                        'id': 'Node(fn=process, args=0, priority=1)',
                        'next': [{
                            'id': 'Node(fn=gather, priority=1)',
                            'type': 'execution'
                        }],
                        'type': 'execution'
                    }, {
                        'id': 'Node(fn=process, args=1, priority=1)',
                        'next': [{
                            'id': 'Node(fn=gather, priority=1)',
                            'type': 'execution'
                        }],
                        'type': 'execution'
                    }, {
                        'id': 'Node(fn=process, args=2, priority=1)',
                        'next': [{
                            'id': 'Node(fn=gather, priority=1)',
                            'type': 'execution'
                        }],
                        'type': 'execution'
                    }],
                    'id': 'Node(fn=build_dynamic_graph, priority=1)',
                    'type': 'group'
                }],
                'type': 'execution'
            }],
            'type': 'execution'
        }],
        'state': {
            'Node(fn=build_dynamic_graph, priority=1)': {
                'displayName': 'build_dynamic_graph',
                'groupMeta': {
                    'Node(fn=gather, priority=1)': {
                        'description': '',
                        'displayName': 'gather',
                        'state': 'SUCCEEDED',
                    },
                    'Node(fn=process, args=0, priority=1)': {
                        'description': '0',
                        'displayName': 'process',
                        'state': 'SUCCEEDED',
                    },
                    'Node(fn=process, args=1, priority=1)': {
                        'description': '1',
                        'displayName': 'process',
                        'state': 'SUCCEEDED',
                    },
                    'Node(fn=process, args=2, priority=1)': {
                        'description': '2',
                        'displayName': 'process',
                        'state': 'SUCCEEDED',
                    }
                },
                'hasControlNode': True
            },
            'Node(fn=first_remote_task, args=A, priority=1)': {
                'displayName': 'first_remote_task',
                'state': 'SUCCEEDED',
            },
            'Node(fn=first_remote_task, args=B, priority=1)': {
                'displayName': 'first_remote_task',
                'state': 'SUCCEEDED',
            },
            'Node(fn=gather, priority=1)': {
                'displayName': 'gather',
                'state': 'SUCCEEDED',
            },
            'Node(fn=process, args=0, priority=1)': {
                'displayName': 'process',
                'state': 'SUCCEEDED',
            },
            'Node(fn=process, args=1, priority=1)': {
                'displayName': 'process',
                'state': 'SUCCEEDED',
            },
            'Node(fn=process, args=2, priority=1)': {
                'displayName': 'process',
                'state': 'SUCCEEDED',
            },
            'Node(fn=second_remote_task, args=C, priority=1)': {
                'displayName': 'second_remote_task',
                'state': 'SUCCEEDED',
            }
        }
    }

    self.assertEqual(workflow_graph, expected_workflow_graph)

  def test_dry_run(self):
    context.num_task_raises_exception_runs = 0

    workflow_graph = None

    def workflow_graph_fn(wg):
      nonlocal workflow_graph
      workflow_graph = wg

    A = (first_remote_task, 'A')
    B = (second_remote_task, 'B')

    graph = Graph({task_raises_exception: {A, B}})

    results, _ = run_graph(graph,
                           workflow_graph_fn=workflow_graph_fn,
                           dry_run=True)
    pprint.pprint(workflow_graph, indent=2, width=200)
    self.assertEqual(context.num_task_raises_exception_runs, 0)

    results, _ = run_graph(graph,
                           workflow_graph_fn=workflow_graph_fn,
                           dry_run=False)
    pprint.pprint(workflow_graph, indent=2, width=200)
    self.assertEqual(context.num_task_raises_exception_runs, 1)


if __name__ == '__main__':
  unittest.main()
