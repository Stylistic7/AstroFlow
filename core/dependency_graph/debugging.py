from dependency_graph import Graph, GraphStatus, Node, ResultDict, TopologicalSorterWithStatus, _unpack_node
from artifacts import FileArtifact, CacheArtifact


def print_result_dict(results, indent=0):
  for node, result in results.items():
    node = _unpack_node(node)

    if type(node.task) is str:
      print(f'{" " * indent}{node.task} = {result}')
      continue

    if type(result) is not ResultDict:
      print(
          f'{" " * indent}{node.task._function.__name__} {node.args}: {result}')
    else:
      print(f'{" " * indent}{node.task._function.__name__} {node.args}:')
      print_result_dict(result, indent + 2)


def print_graph(graph):
  if not graph:
    print('Compute graph is empty')
    return

  print(f'Printing compute graph with {len(graph)} edges:')
  for k, v in graph.items():

    def type_or_value(x):
      if type(x) in [int, float, str, FileArtifact, CacheArtifact]:
        return f"'{str(x)}'"

      return str(type(x))

    def tuple_values(tup):
      return ', '.join([type_or_value(t) for t in tup])

    def to_string(node):
      node = _unpack_node(node)
      return f'({node.task._function.__name__}, {tuple_values(node.args)})'

    if type(v) in [list, set]:
      strings = [to_string(i) for i in v]
      print(f'{to_string(k)}: [{", ".join(strings)}]')
      continue

    assert False, f'Node dependencies have unsupported type {type(v)}: {v}'

  print('Done printing compute graph')
