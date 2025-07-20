from typing import TypeVar
import unittest
import ray
import structlog

from dependency_graph import Graph, run_graph
from dependency_graph.logging import capture_logs
from dependency_graph.debugging import node_to_string

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
@capture_logs
def maybe_run_dynamic_graph(should_run: bool) -> Graph:
  if should_run:
    log.info('running dynamic graph')
    return Graph({gather: [(process, i) for i in range(7)]})
  else:
    log.info('not running dynamic graph')
    return Graph({})


class TestLogging(unittest.TestCase):

  def test_simple_graph(self) -> None:
    A = (first_remote_task, 'A')
    B = (first_remote_task, 'B')
    C = (second_remote_task, 'C')
    D = (second_remote_task, 'D')

    graph = Graph({D: {B, C}, C: {A}, B: {A}})
    results, logs = run_graph(graph)

    context.last_run_results = {
        node_to_string(node): result
        for node, result in results.flatten().items()
    }

    context.last_run_logs = {
        node_to_string(node): log for node, log in logs.flatten().items()
    }

    self.assertEqual(results[A], 'A')
    self.assertEqual(results[B], 'B')
    self.assertEqual(results[C], 'C')
    self.assertEqual(results[D], 'D')

    self.assertEqual(logs[A][0]['event'], 'first_remote_task A')
    self.assertEqual(logs[B][0]['event'], 'first_remote_task B')
    self.assertEqual(logs[C][0]['event'], 'second_remote_task C')
    self.assertEqual(logs[D][0]['event'], 'second_remote_task D')

    self.assertEqual(results.num_nodes_run, 4)
    self.assertEqual(results.max_parallelism_seen, 2)

  def test_conditional_dynamic_graph(self) -> None:
    dynamic_graph_task = (maybe_run_dynamic_graph, True)
    graph = Graph({dynamic_graph_task: {}})
    results, logs = run_graph(graph)
    self.assertEqual(results[dynamic_graph_task], 21)

    self.assertEqual(results.num_nodes_run, 9)
    self.assertEqual(results.max_parallelism_seen, 7)

    print(f'### {len(logs)} nodes produced logs')

    for node, log in logs.flatten().items():
      print(f'### node {node_to_string(node)} has log of length {len(log)}')
      for entry in log:
        print(f'###   {entry}')

    context.last_run_results = {
        node_to_string(node): result
        for node, result in results.flatten().items()
    }

    context.last_run_logs = {
        node_to_string(node): log for node, log in logs.flatten().items()
    }


if __name__ == '__main__':
  unittest.main()
