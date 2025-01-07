from typing import Any

from ray.remote_function import RemoteFunction

from dependency_graph import Graph, CanonicalGraph, Node, ResultDict, canonicalize_node


def _tuple_values(tup: tuple) -> str:
  return ', '.join([str(t) for t in tup])


def node_to_string(node: Node) -> str:
  node = canonicalize_node(node)
  assert isinstance(node.task, RemoteFunction)

  if node.name:
    string = f'Node({node.task._function.__name__}'
    if node.args:
      string += f', {_tuple_values(node.args)}'
    string += f", name='{node.name}')"
    return string

  if node.args:
    return f'({node.task._function.__name__}, {_tuple_values(node.args)})'
  else:
    return f'{node.task._function.__name__}'


def node_to_string_tuple(node: Node) -> str:
  node = canonicalize_node(node)
  assert isinstance(node.task, RemoteFunction)
  return (node.task._function.__name__, *[str(x) for x in node.args])


def print_result_dict(results: ResultDict, indent: int = 0) -> None:
  for node, result in results.items():
    if type(node.task) is str:
      print(f'{" " * indent}{node.task} = {result}')
      continue

    assert isinstance(node.task, RemoteFunction)
    if type(result) is not ResultDict:
      print(
          f'{" " * indent}{node.task._function.__name__} {node.args}: {result}')
    else:
      print(f'{" " * indent}{node.task._function.__name__} {node.args}:')
      print_result_dict(result, indent + 2)


def print_graph(graph: CanonicalGraph) -> None:
  if not graph:
    print('Compute graph is empty')
    return

  print(f'Printing compute graph with {len(graph)} edges:')
  for k, v in graph.items():

    if type(v) in [list, set]:
      strings = [node_to_string(i) for i in v]
      print(f'{node_to_string(k)}: [{", ".join(strings)}]')
      continue

    assert False, f'Node dependencies have unsupported type {type(v)}: {v}'

  print('Done printing compute graph')
