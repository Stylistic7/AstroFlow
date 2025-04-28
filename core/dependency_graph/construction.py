import ray

from typing import Any

from dependency_graph import Graph, TaskDescriptor


@ray.remote
def list_task_done(x: Any) -> Any:
  return x


def chain(*nodes):
  """Create a linear chain of graph dependencies."""
  return {nodes[i + 1]: {nodes[i]} for i in range(len(nodes) - 1)}
