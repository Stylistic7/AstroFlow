import ray

from typing import Any

from dependency_graph import Graph, TaskDescriptor


@ray.remote
def list_task_done(x: Any) -> Any:
  return x


def make_graph_from_list(tasks: list[TaskDescriptor]) -> Graph:
  graph = Graph({})
  for i in range(len(tasks) - 1):
    graph[tasks[i + 1]] = {tasks[i]}

  return graph
