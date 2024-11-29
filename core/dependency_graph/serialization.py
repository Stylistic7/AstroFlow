import pickle

from dependency_graph import Graph, Node, GraphStatus, TopologicalSorterWithStatus, canonicalize_graph, run_graph, print_graph, make_graph_from_list
from dependency_graph.debugging import node_to_string_tuple


# TODO: Include logs for each task in this.
def pickle_graph(graph: Graph) -> str:
  converted_graph = {}
  for key, value in graph.items():
    converted_graph[node_to_string_tuple(key)] = [
        node_to_string_tuple(v) for v in value
    ]

  return pickle.dumps(converted_graph)
