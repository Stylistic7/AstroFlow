from __future__ import annotations

from hashing import hash_list_int
from typing import Any, Callable, Union, TypeAlias, Iterable
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from collections import defaultdict
from cache.types import NeverRunType

from ray.remote_function import RemoteFunction
from ray._private.worker import (
    RemoteFunctionNoArgs,
    RemoteFunction0,
    RemoteFunction1,
    RemoteFunction2,
    RemoteFunction3,
    RemoteFunction4,
    RemoteFunction5,
    RemoteFunction6,
)

GenericRemoteFunction: TypeAlias = (
    RemoteFunction
    | RemoteFunctionNoArgs
    | RemoteFunction0
    | RemoteFunction1
    | RemoteFunction2
    | RemoteFunction3
    | RemoteFunction4
    | RemoteFunction5
    | RemoteFunction6
)


def _tuple_values(tup: tuple) -> str:
    return ", ".join([str(t) for t in tup])


class SkippedType:
    def __eq__(self, other):
        return isinstance(other, SkippedType)

    def __hash__(self):
        return hash("SkippedType")

    def __repr__(self):
        return "Skipped"

    def __str__(self):
        return "Skipped"


Skipped = SkippedType()


class Node:
    task: GenericRemoteFunction
    name: str | None
    priority: int
    args: Any

    def __init__(
        self,
        task: GenericRemoteFunction,
        *args: Any,
        name: str | None = None,
        priority: int = 1,
    ) -> None:
        self.task = task
        self.name = name
        self.priority = priority
        self.args = args

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, Node)
        return (
            self.task._function.__module__ == other.task._function.__module__
            and self.task._function.__name__ == other.task._function.__name__
            and self.name == other.name
            and self.priority == other.priority
            and self.args == other.args
        )

    def compatible_with(self, other) -> bool:
        other = canonicalize_node(other)

        return (
            self.task._function.__module__ == other.task._function.__module__
            and self.task._function.__name__ == other.task._function.__name__
            and self.name == other.name
            and self.priority == other.priority
            and all(a == b for a, b in zip(self.args, other.args))
        )

    def __hash__(self) -> int:
        return hash_list_int(
            [
                self.task._function.__module__,
                self.task._function.__name__,
                self.name,
                self.priority,
                self.args,
            ]
        )

    def __str__(self):
        return (
            f"Node({self.task._function.__module__}.{self.task._function.__name__}"
            f"({_tuple_values(self.args)}) name='{self.name}' "
            f"priority={self.priority})"
        )

    def __lt__(self, other: "Node") -> bool:
        assert isinstance(other, Node)
        return (
            self.name or "",
            self.task._function.__name__,
            self.priority,
            self.args,
        ) < (
            other.name or "",
            other.task._function.__name__,
            other.priority,
            other.args,
        )


RemoteFunctionTuple: TypeAlias = Union[
    tuple[GenericRemoteFunction, Any],
    tuple[GenericRemoteFunction, Any, Any],
    tuple[GenericRemoteFunction, Any, Any, Any],
    tuple[GenericRemoteFunction, Any, Any, Any, Any],
    tuple[GenericRemoteFunction, Any, Any, Any, Any, Any],
    tuple[GenericRemoteFunction, Any, Any, Any, Any, Any, Any],
    tuple[GenericRemoteFunction, Any, Any, Any, Any, Any, Any, Any],
]

TaskDescriptor: TypeAlias = Node | GenericRemoteFunction | RemoteFunctionTuple


def canonicalize_node(task: TaskDescriptor) -> Node:
    if isinstance(task, Node):
        return task
    elif isinstance(task, tuple):
        fn, *args = task
        return Node(fn, *args)
    else:
        return Node(task)


class TopologicalSorterWithStatus:
    def __init__(
        self,
        graph: CanonicalGraph,
        graph_update_fn: GraphUpdateFn | None = None,
        parent_node: Node | None = None,
        done_callback: DoneCallback | None = None,
    ) -> None:
        self.parent_node = parent_node
        self.done_callback = done_callback
        self.results = ResultDict()
        self.logs = ResultDict()
        self.complete_args = ResultDict()
        self.final_result = None

        self.graph = defaultdict(list)
        self.graph_update_fn = graph_update_fn
        self.done_nodes = list()
        self._ts = None

        self._ready_cache = None
        self._waiting_cache = None
        self._cache_valid = False

        if graph:
            for node, dependencies in graph.items():
                self.add(node, *dependencies)

    def add(self, node: Any, *dependencies: Iterable[Any]):
        seen = set(self.graph[node])
        for dep in dependencies:
            if dep not in seen:
                self.graph[node].append(dep)
                seen.add(dep)
            if dep not in self.graph:
                self.graph[dep] = []

    def prepare(self):
        self._ts = TopologicalSorter(self.graph)
        self._ts.prepare()

    def is_active(self) -> bool:
        return self._ts.is_active()

    def get_ready(self):
        return self._ts.get_ready()

    def done(self, node: Any):
        self._ts.done(node)
        self.done_nodes.append(node)
        self._cache_valid = False  # Invalidate cache

    def _update_caches(self):
        ready = []
        waiting = []
        done_set = set(self.done_nodes)
        for node, deps in self.graph.items():
            if node in done_set:
                continue
            if all(dep in done_set for dep in deps):
                ready.append(node)
            else:
                waiting.append(node)
        self._ready_cache = ready
        self._waiting_cache = waiting
        self._cache_valid = True

    def peek_ready(self) -> list:
        if not self._cache_valid:
            self._update_caches()
        return self._ready_cache

    def peek_waiting(self) -> list:
        if not self._cache_valid:
            self._update_caches()
        return self._waiting_cache

    def peek_done(self) -> list:
        return list(self.done_nodes)

    def get_status(self) -> GraphStatus:
        return GraphStatus(
            self.peek_done(), self.peek_ready(), self.peek_waiting()
        )


GraphUpdateFn = Callable[list[TopologicalSorterWithStatus], Any]


class Graph(dict[TaskDescriptor, Iterable[TaskDescriptor]]):
    __slots__ = ("graph_update_fn",)
    graph_update_fn: GraphUpdateFn | None

    def __init__(
        self,
        *args,
        graph_update_fn: GraphUpdateFn | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.graph_update_fn = graph_update_fn


class CanonicalGraph(dict[Node, list[Node]]):
    def __eq__(self, other):
        if len(self) != len(other):
            return False

        for k in self.keys():
            if k not in other:
                return False
            if type(self[k]) != type(other[k]):
                return False
            for vs, vo in zip(self[k], other[k]):
                if vs != vo:
                    return False

        return True


# TODO: Maybe support nodes with one ancestor, like this {done: process},
# instead of requiring {done: {process}}.
def canonicalize_graph(graph: Graph) -> CanonicalGraph:
    output = CanonicalGraph({})
    for node, dependents in graph.items():
        output[canonicalize_node(node)] = [
            canonicalize_node(x) for x in dependents
        ]
    return output


@dataclass
class GraphStatus:
    done_nodes: list[Node] = field(default_factory=list)
    ready_nodes: list[Node] = field(default_factory=list)
    waiting_nodes: list[Node] = field(default_factory=list)

    def __add__(self, rhs: "GraphStatus") -> "GraphStatus":
        self.done_nodes += rhs.done_nodes
        self.ready_nodes += rhs.ready_nodes
        self.waiting_nodes += rhs.waiting_nodes

        return self


class ResultDict(dict[Node, Any]):
    num_nodes_run: int = 0
    max_parallelism_seen: int = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(ResultDict, self).__init__(*args, **kwargs)

    def __getattr__(self, node: Any) -> Any:
        return self[node]

    def __getitem__(self, node: Any) -> Any | None:
        node = canonicalize_node(node)

        for result_node, result in self.items():
            if result_node == node:
                return result

        raise KeyError(
            f"Key {node.task._function.__module__}.{node.task._function.__name__} "
            f"args {node.args} not found"
        )

    def flatten(self) -> dict[Node, Any]:
        output = {}

        def process(r: ResultDict) -> None:
            for node, value in r.items():
                if isinstance(value, ResultDict):
                    process(value)
                else:
                    output[node] = value

        process(self)
        return output

    def contains_instance(self, cls):
        found = False

        def process(r: ResultDict) -> None:
            nonlocal found
            for value in r.values():
                if isinstance(value, ResultDict):
                    process(value)
                elif isinstance(value, cls):
                    found = True
                    return

        process(self)
        return found

    def contains_exception(self):
        return self.contains_instance(Exception)

    def is_complete(self):
        return not self.contains_instance(
            Exception
        ) and not self.contains_instance(NeverRunType)


DoneCallback = Callable[[], None]
