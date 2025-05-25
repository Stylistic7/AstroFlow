from collections import defaultdict

from dependency_graph import Node, TopologicalSorterWithStatus
from cache.types import NeverRun


def node_to_id(node: Node) -> str:
  return str(hash(node))


class WorkflowGraph(dict):
  pass


def _state_string(ts, node):
  if ts.results.get(node, NeverRun) == NeverRun:
    return 'NO_STATE_RUNTIME'

  if isinstance(ts.results[node], Exception):
    return 'FAILED'

  return 'SUCCEEDED'


def to_workflow_graph(subgraphs: list[TopologicalSorterWithStatus]):
  subgraphs_by_parent = {
      sg.parent_node: sg for sg in subgraphs if sg.parent_node is not None
  }

  state: dict[str, dict] = {}
  skeleton: dict[str, dict] = {}

  def flatten_graph(ts: TopologicalSorterWithStatus):
    graph = ts.graph

    reverse = defaultdict(set)
    for node, deps in graph.items():
      for dep in deps:
        reverse[dep].add(node)

    for node in graph:
      key = node_to_id(node)
      if key in skeleton:
        continue

      if node in subgraphs_by_parent:
        sub_ts = subgraphs_by_parent[node]
        group_meta = {
            node_to_id(child): {
                "displayName": child.task._function.__name__,
                "description": str(child.args[0]) if child.args else "",
                "state": _state_string(sub_ts, child),
            } for child in sub_ts.graph
        }
        state[key] = {
            "displayName": node.task._function.__name__,
            "hasControlNode": True,
            "groupMeta": group_meta,
        }
        skeleton[key] = {
            "id":
                key,
            "type":
                "group",
            "definition": [
                node_to_id(n) for n in sorted(child for child in sub_ts.graph
                                              if not sub_ts.graph[child])
            ]
        }
        flatten_graph(sub_ts)
      else:
        state[key] = {
            "displayName": node.task._function.__name__,
            "state": _state_string(ts, node)
        }
        children = sorted(reverse[node], key=node_to_id)
        skeleton[key] = {
            "id": key,
            "type": "execution",
            **({
                "next": [{
                    "id": node_to_id(children[0])
                }]
            } if children else {})
        }

  for ts in subgraphs:
    flatten_graph(ts)

  top_level_ts = next(ts for ts in subgraphs if ts.parent_node is None)
  roots = sorted(n for n, deps in top_level_ts.graph.items() if not deps)
  root_ids = [node_to_id(n) for n in roots]

  return {"skeleton": list(skeleton.values()), "state": state}
