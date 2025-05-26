from collections import defaultdict

from dependency_graph import Node, TopologicalSorterWithStatus
from dependency_graph.debugging import node_to_string
from cache.types import NeverRun


class WorkflowGraph(dict):
  pass


def node_to_id(node: Node) -> str:
  return node_to_string(node)


def _state_string(ts, node):
  result = ts.results.get(node, NeverRun)
  return 'NO_STATE_RUNTIME' if result == NeverRun else 'FAILED' if isinstance(
      result, Exception) else 'SUCCEEDED'


def gather_group_meta(
    ts: TopologicalSorterWithStatus,
    subgraphs_by_parent: dict[Node, TopologicalSorterWithStatus]
) -> dict[str, dict]:

  def recurse(ts: TopologicalSorterWithStatus, node: Node):
    key = node_to_id(node)
    meta = {
        "displayName": node.task._function.__name__,
        "state": _state_string(ts, node),
    }
    if node.args:
      meta["description"] = str(node.args[0])
    if node.name:
      meta["description"] = node.name
    if node in subgraphs_by_parent:
      sub_ts = subgraphs_by_parent[node]
      meta["hasControlNode"] = True
      meta["groupMeta"] = {
          node_to_id(n): recurse(sub_ts, n) for n in sub_ts.graph
      }
    return meta

  return {node_to_id(n): recurse(ts, n) for n in ts.graph}


def build_skeleton(
    ts: TopologicalSorterWithStatus,
    subgraphs_by_parent: dict[Node, TopologicalSorterWithStatus]) -> list[dict]:
  graph = ts.graph
  reverse = defaultdict(set)
  for node, deps in graph.items():
    for dep in deps:
      reverse[dep].add(node)
  roots = sorted(n for n, deps in graph.items() if not deps)

  def build(node):
    key = node_to_id(node)

    if node in subgraphs_by_parent:
      sub_ts = subgraphs_by_parent[node]
      result = {
          "id": key,
          "type": "group",
          "definition": build_skeleton(sub_ts, subgraphs_by_parent)
      }
    else:
      result = {"id": key, "type": "execution"}

    children = [build(n) for n in sorted(reverse[node])]
    if children:
      result["next"] = children

    return result

  return [build(n) for n in roots]


def to_workflow_graph(subgraphs: list[TopologicalSorterWithStatus]):
  subgraphs_by_parent = {
      sg.parent_node: sg for sg in subgraphs if sg.parent_node is not None
  }

  top_level_ts = next(ts for ts in subgraphs if ts.parent_node is None)

  return {
      "skeleton": build_skeleton(top_level_ts, subgraphs_by_parent),
      "state": gather_group_meta(top_level_ts, subgraphs_by_parent),
  }
