from typing import Any
from collections import Counter

from ray.remote_function import RemoteFunction

from dependency_graph import Node, GraphStatus
from dependency_graph.debugging import node_to_string
from profiling.estimate_time import estimate_time


# This function takes lists of ready, waiting, and done nodes from the
# dependency graph, and collates them into a dict of processes, described by
# their total number, remaining number, and estimated remaining time.
#
# TODO: Strongly-type the output.
def graph_status_to_progress(status: GraphStatus) -> dict[str, Any]:
  print(f'### status has {len(status.ready_nodes)} ready nodes, '
        f'{len(status.waiting_nodes)} waiting nodes, '
        f'{len(status.done_nodes)} done nodes')

  task_counts = Counter(
      node.task._function.__name__ for node in status.ready_nodes)
  print('### ready nodes: ' +
        ', '.join(f"'{task}': {count}" for task, count in task_counts.items()))

  def function_name_from_node(node: Node) -> str:
    assert isinstance(node.task, RemoteFunction)
    return node.task._function.__name__

  status_dict = {}
  for node in status.ready_nodes | status.waiting_nodes:
    name = function_name_from_node(node)
    estimated_time = estimate_time(name)
    if estimated_time is None:
      continue

    if name not in status_dict:
      status_dict[name] = {
          'total_number': 1,
          'remaining_number': 1,
          'estimated_time': estimated_time
      }
    else:
      status_dict[name]['total_number'] += 1
      status_dict[name]['remaining_number'] += 1
      status_dict[name]['estimated_time'] = status_dict[name][
          'remaining_number'] * estimated_time

  for node in status.done_nodes:
    name = function_name_from_node(node)
    estimated_time = estimate_time(name)
    if estimated_time is None:
      continue

    if name not in status_dict:
      status_dict[name] = {
          'total_number': 1,
          'remaining_number': 0,
          'estimated_time': 0
      }
    else:
      status_dict[name]['total_number'] += 1

  for name in status_dict.keys():
    if status_dict[name]['remaining_number'] == 0:
      status_dict[name]['estimated_time'] = 0

  return status_dict
