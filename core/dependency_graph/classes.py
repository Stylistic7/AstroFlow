from typing import Dict
from dataclasses import dataclass, field
import graphlib


def canonicalize_node(node):

  if isinstance(node, Node):
    return node
  elif isinstance(node, tuple):
    return Node(*node)
  else:
    return Node(node)


class Node():

  def __init__(self, task, *args, name=None, priority=1):
    self.task = task
    self.name = name
    self.priority = priority
    self.args = args

  def __eq__(self, other):
    return (self.task._function.__module__ == other.task._function.__module__
            and
            self.task._function.__name__ == other.task._function.__name__ and
            self.name == other.name and self.priority == other.priority and
            self.args == other.args)

  def __hash__(self):
    return hash((self.task, self.name, self.priority, self.args))


class Graph(dict):
  pass


@dataclass
class GraphStatus:
  done_nodes: set[Node] = field(default_factory=set)
  ready_nodes: set[Node] = field(default_factory=set)
  waiting_nodes: set[Node] = field(default_factory=set)

  def __or__(self, rhs):
    self.done_nodes |= rhs.done_nodes
    self.ready_nodes |= rhs.ready_nodes
    self.waiting_nodes |= rhs.waiting_nodes

    return self


class TopologicalSorterWithStatus(graphlib.TopologicalSorter):

  def __init__(self, graph):
    super().__init__(graph)

    self.done_nodes = set()
    self.ready_nodes = set()
    self.waiting_nodes = set(graphlib.TopologicalSorter(graph).static_order())

  def get_status(self):
    return GraphStatus(self.done_nodes, self.ready_nodes, self.waiting_nodes)

  def get_ready(self):
    ready_nodes = super().get_ready()

    self.ready_nodes = set(ready_nodes)
    self.waiting_nodes -= self.ready_nodes

    return ready_nodes

  def done(self, node):
    super().done(node)
    self.done_nodes.add(node)
    if node in self.ready_nodes:
      self.ready_nodes.remove(node)


class ResultDict(dict):

  def __init__(self, *args, **kwargs):
    super(ResultDict, self).__init__(*args, **kwargs)

  def __getattr__(self, k):
    return self[k]

  def __getitem__(self, key):
    if type(key) is str:
      return self.get(key)

    key = canonicalize_node(key)

    for node, result in self.items():
      # node = canonicalize_node(node)
      if node == key:
        return result

    raise KeyError(
        f'Key {key.task._function.__module__}.{key.task._function.__name__} '
        f'args {key.args} not found')
