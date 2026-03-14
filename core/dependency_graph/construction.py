import ray

from typing import Any

from dependency_graph import Graph, TaskDescriptor
from beartype import beartype


@ray.remote
def list_task_done(x: Any) -> Any:
    return x


def chain(*nodes: TaskDescriptor) -> Graph:
    """Create a linear chain of graph dependencies."""
    return Graph({nodes[i + 1]: {nodes[i]} for i in range(len(nodes) - 1)})
