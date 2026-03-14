from ray.remote_function import RemoteFunction
from copy import copy

from dependency_graph import Graph, Node, canonicalize_graph
from beartype import beartype


def replace_task(
    graph: Graph, from_task: RemoteFunction, to_task: RemoteFunction
) -> Graph:
    graph = canonicalize_graph(graph)

    def update_node(node: Node):
        new_node = copy(node)
        if new_node.task == from_task:
            new_node.task = to_task

        return new_node

    output_graph = Graph()
    for k, v in graph.items():
        new_k = update_node(k)

        if type(v) is list:
            output_graph[new_k] = [update_node(i) for i in v]
        elif type(v) is set:
            output_graph[new_k] = {update_node(i) for i in v}

    return output_graph
