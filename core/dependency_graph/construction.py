import ray
from ray.remote_function import RemoteFunction

from typing import Any, cast

from dependency_graph import Graph, Node, TaskDescriptor


@ray.remote
def list_task_done(ancestor_results: list[Any]) -> Any:
  return ancestor_results[0]


def make_graph_from_list(tasks: list[TaskDescriptor]) -> Graph:
  graph = Graph({})
  prev_done_node = None
  for i, task in enumerate(tasks):
    done_node = Node(cast(RemoteFunction, list_task_done), name=f'done_{i}')
    graph[done_node] = [task]

    if prev_done_node:
      graph[task] = [prev_done_node]

    prev_done_node = done_node

  return graph
