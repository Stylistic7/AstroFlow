from typing import Any, Callable, Sequence, Type, TypeAlias, Tuple
from dataclasses import dataclass, field
import graphlib

from ray.remote_function import RemoteFunction


class Node():

  task: RemoteFunction
  name: str | None
  priority: int
  args: Any

  def __init__(self,
               task: RemoteFunction,
               *args: Any,
               name: str | None = None,
               priority: int = 1) -> None:
    self.task = task
    self.name = name
    self.priority = priority
    self.args = args

  def __eq__(self, other: Any) -> bool:
    assert isinstance(other, Node)
    return (self.task._function.__module__ == other.task._function.__module__
            and
            self.task._function.__name__ == other.task._function.__name__ and
            self.name == other.name and self.priority == other.priority and
            self.args == other.args)

  def __hash__(self) -> int:
    return hash((self.task, self.name, self.priority, self.args))


RemoteFunctionTuple: TypeAlias = Tuple[RemoteFunction, ...]
TaskDescriptor: TypeAlias = Node | RemoteFunction | RemoteFunctionTuple


def canonicalize_node(task: TaskDescriptor) -> Node:

  if isinstance(task, Node):
    return task
  elif isinstance(task, tuple):
    fn, *args = task
    return Node(fn, *args)
  else:
    return Node(task)


class Graph(dict[TaskDescriptor, Sequence[TaskDescriptor]]):
  pass


class CanonicalGraph(dict[Node, set[Node]]):
  pass


@dataclass
class GraphStatus:
  done_nodes: set[Node] = field(default_factory=set)
  ready_nodes: set[Node] = field(default_factory=set)
  waiting_nodes: set[Node] = field(default_factory=set)

  def __or__(self, rhs: 'GraphStatus') -> 'GraphStatus':
    self.done_nodes |= rhs.done_nodes
    self.ready_nodes |= rhs.ready_nodes
    self.waiting_nodes |= rhs.waiting_nodes

    return self


class ResultDict(dict[Node, Any]):

  num_nodes_run: int = 0
  max_parallelism_seen: int = 0

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    super(ResultDict, self).__init__(*args, **kwargs)

  def __getattr__(self, node: Any) -> Any:
    return self[node]

  def __getitem__(self, node: Node) -> Any | None:
    if type(node) is str:
      return self.get(node)

    node = canonicalize_node(node)

    for result_node, result in self.items():
      if result_node == node:
        return result

    raise KeyError(
        f'Key {node.task._function.__module__}.{node.task._function.__name__} '
        f'args {node.args} not found')


DoneCallback = Callable[[], None]


class TopologicalSorterWithStatus(graphlib.TopologicalSorter):
  done_nodes: set[Node]
  ready_nodes: set[Node]
  waiting_nodes: set[Node]

  graph: CanonicalGraph
  done_callback: DoneCallback | None
  results: ResultDict
  final_result: Any

  def __init__(self,
               graph: CanonicalGraph,
               done_callback: DoneCallback | None = None) -> None:
    super().__init__(graph)

    self.graph = graph
    self.done_callback = done_callback
    self.results = ResultDict()
    self.final_result = None

    self.done_nodes = set()
    self.ready_nodes = set()
    self.waiting_nodes = set(graphlib.TopologicalSorter(graph).static_order())

  def get_status(self) -> GraphStatus:
    return GraphStatus(self.done_nodes, self.ready_nodes, self.waiting_nodes)

  def get_ready(self) -> Tuple[Any, ...]:
    ready_nodes = super().get_ready()

    self.ready_nodes = set(ready_nodes)
    self.waiting_nodes -= self.ready_nodes

    return ready_nodes

  def done(self, *nodes: Any) -> None:
    for node in nodes:
      super().done(node)
      self.done_nodes.add(node)
      if node in self.ready_nodes:
        self.ready_nodes.remove(node)
