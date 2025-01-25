from dependency_graph import Node, TopologicalSorterWithStatus, canonicalize_node
from dependency_graph.debugging import node_to_string


class WorkflowGraph(dict):
  pass


def to_workflow_graph(
    subgraphs: list[TopologicalSorterWithStatus]) -> WorkflowGraph:

  print(f'### to_workflow_graph called with {len(subgraphs)} subgraphs')

  def to_entry(node: Node):
    print(f'### to_entry {node}')
    node = canonicalize_node(node)
    id = node_to_string(node)

    type = 'execution'
    for s in subgraphs:
      if s.parent_node and s.parent_node == node:
        type = 'group'

    return {'id': id, 'type': type}

  def add_subgraph(subgraph: TopologicalSorterWithStatus):
    skeleton = []
    for k, values in subgraph.graph.items():
      for v in values:
        entry = to_entry(v)
        entry['next'] = [to_entry(k)]
        skeleton.append(entry)

    return skeleton

  state = {}
  skeleton = []
  skeleton.extend(add_subgraph(subgraphs[0]))

  for subgraph in subgraphs[1:]:
    node = subgraph.parent_node
    node = canonicalize_node(node)
    id = node_to_string(node)
    skeleton.append({
        'id': id,
        'type': 'group',
        'definition': add_subgraph(subgraph)
    })
    state[id] = {'displayName': id, 'hasControlNode': True}

  return {'skeleton': skeleton, 'state': state}
