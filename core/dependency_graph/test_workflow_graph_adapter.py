from ray_infra import ray_local

from typing import TypeVar
import unittest
import ray
import structlog
import pprint

import compute_graph
from testing.test_context import make_test_context
from context import get_cache
from dependency_graph import Graph, run_graph
from dependency_graph.logging import capture_logs
from mosaics.orion_mosaic import add_orion_mosaic_images

cache = get_cache()
log = structlog.get_logger()

T = TypeVar('T')


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
    D = build_dynamic_graph
    E = (second_remote_task, 'D')

    graph = Graph({E: {D}, D: {B, C}, C: {A}, B: {A}})

    workflow_graph = None

    def workflow_graph_fn(wg):
      nonlocal workflow_graph
      workflow_graph = wg

    results, logs = run_graph(graph, workflow_graph_fn=workflow_graph_fn)

    pprint.pprint(workflow_graph, indent=2, width=80)

    expected_workflow_graph = {
        'skeleton': [{
            'id': 'build_dynamic_graph',
            'next': [{
                'id': '(second_remote_task, D)',
                'type': 'execution'
            }],
            'type': 'group'
        }, {
            'id': '(second_remote_task, C)',
            'next': [{
                'id': 'build_dynamic_graph',
                'type': 'group'
            }],
            'type': 'execution'
        }, {
            'id': '(first_remote_task, B)',
            'next': [{
                'id': 'build_dynamic_graph',
                'type': 'group'
            }],
            'type': 'execution'
        }, {
            'id': '(first_remote_task, A)',
            'next': [{
                'id': '(second_remote_task, C)',
                'type': 'execution'
            }],
            'type': 'execution'
        }, {
            'id': '(first_remote_task, A)',
            'next': [{
                'id': '(first_remote_task, B)',
                'type': 'execution'
            }],
            'type': 'execution'
        }, {
            'definition': [{
                'id': '(process, 0)',
                'next': [{
                    'id': 'gather',
                    'type': 'execution'
                }],
                'type': 'execution'
            }, {
                'id': '(process, 1)',
                'next': [{
                    'id': 'gather',
                    'type': 'execution'
                }],
                'type': 'execution'
            }, {
                'id': '(process, 2)',
                'next': [{
                    'id': 'gather',
                    'type': 'execution'
                }],
                'type': 'execution'
            }],
            'id': 'build_dynamic_graph',
            'type': 'group'
        }],
        'state': {
            'build_dynamic_graph': {
                'displayName': 'build_dynamic_graph',
                'hasControlNode': True
            }
        }
    }

    self.assertEqual(workflow_graph, expected_workflow_graph)

  def test_orion_mosaic(self) -> None:
    add_orion_mosaic_images()

    # Running the graph the first time, with an empty shopping cart, should
    # produce all the subframe previews for the stacks, targets, and mosaic.
    graph = compute_graph.build_compute_graph(regroup=True,
                                              make_previews=True,
                                              run_main_processing_graph=True)

    def workflow_graph_fn(workflow_graph):
      print('### workflow_graph_fn called with', workflow_graph)

    results, logs = run_graph(graph, workflow_graph_fn=workflow_graph_fn)


if __name__ == '__main__':
  unittest.main()
