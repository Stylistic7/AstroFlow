from dependency_graph import Node, TopologicalSorterWithStatus, canonicalize_node
from dependency_graph.debugging import node_to_string


class WorkflowGraph(dict):
  pass


from collections import defaultdict


def to_workflow_graph(subgraphs: list[TopologicalSorterWithStatus]):
  subgraphs_by_parent = {
      sg.parent_node: sg for sg in subgraphs if sg.parent_node is not None
  }

  state: dict[str, dict] = {}

  def build_graph(ts: TopologicalSorterWithStatus):
    graph = ts.graph

    reverse = defaultdict(set)
    for node, deps in graph.items():
      for dep in deps:
        reverse[dep].add(node)

    roots = sorted([n for n, deps in graph.items() if not deps])

    def build(node):
      key = node_to_string(node)

      if node in subgraphs_by_parent:
        sub_ts = subgraphs_by_parent[node]
        group_meta = {
            node_to_string(child): {
                "displayName": child.task._function.__name__,
                "description": str(child.args[0]) if child.args else "",
            } for child in sub_ts.graph
        }
        state[key] = {
            "displayName": node.task._function.__name__,
            "hasControlNode": True,
            "groupMeta": group_meta,
        }

        return {
            "id": node_to_string(node),
            "type": "group",
            "definition": build_graph(sub_ts)
        }

      state[key] = {"displayName": node.task._function.__name__}

      children = [build(n) for n in sorted(reverse[node])]
      return {
          "id": node_to_string(node),
          "type": "execution",
          **({
              "next": children
          } if children else {})
      }

    return [build(n) for n in roots]

  top_level_ts = next(ts for ts in subgraphs if ts.parent_node is None)
  return {"skeleton": build_graph(top_level_ts), "state": state}
