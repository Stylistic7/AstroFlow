import ray

from dependency_graph import Graph, Node


@ray.remote
def list_task_done(ancestor_results):
  return ancestor_results[0]


def make_graph_from_list(tasks):
  graph = Graph({})
  prev_done_node = None
  for i, task in enumerate(tasks):
    done_node = Node(list_task_done, name=f'done_{i}')
    graph[done_node] = [task]

    if prev_done_node:
      graph[task] = [prev_done_node]

    prev_done_node = done_node

  return graph
