from __future__ import annotations

from collections import Counter
from hashing import hash_list_int
from typing import Any, Union, TypeAlias
from collections.abc import Iterable, Callable
from dataclasses import dataclass, field
from cache.types import NeverRunType
from beartype import beartype

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


Skipped = SkippedType()


class Node:
    task: GenericRemoteFunction
    name: str | None
    args: Any

    def __init__(
        self,
        task: GenericRemoteFunction,
        *args: Any,
        name: str | None = None,
    ) -> None:
        self.task = task
        self.name = name
        self.args = args

    def __eq__(self, other: Any) -> bool:
        assert isinstance(other, Node)
        if not hasattr(self.task, "_function"):
            raise AttributeError(
                f"Node.task missing `_function` attr — task is a bare "
                f"{type(self.task).__name__} ({self.task!r}); expected a "
                f"@ray.remote RemoteFunction or @node wrapper. Node name="
                f"{self.name!r}, args={self.args!r}. A pipeline or graph "
                f"builder created a Node with an undecorated callable."
            )
        if not hasattr(other.task, "_function"):
            raise AttributeError(
                f"Other Node.task missing `_function` attr — task is a bare "
                f"{type(other.task).__name__} ({other.task!r}); expected a "
                f"@ray.remote RemoteFunction or @node wrapper. Other Node "
                f"name={other.name!r}, args={other.args!r}."
            )
        return (
            self.task._function.__module__ == other.task._function.__module__
            and self.task._function.__name__ == other.task._function.__name__
            and self.name == other.name
            and self.args == other.args
        )

    def compatible_with(self, other) -> bool:
        other = canonicalize_node(other)

        return (
            self.task._function.__module__ == other.task._function.__module__
            and self.task._function.__name__ == other.task._function.__name__
            and self.name == other.name
            and all(a == b for a, b in zip(self.args, other.args))
        )

    def __hash__(self) -> int:
        if not hasattr(self, '_hash_cache'):
            if not hasattr(self.task, "_function"):
                raise AttributeError(
                    f"Node.task missing `_function` attr — task is a bare "
                    f"{type(self.task).__name__} ({self.task!r}); expected a "
                    f"@ray.remote RemoteFunction or @node wrapper. Node "
                    f"name={self.name!r}, args={self.args!r}. A pipeline or "
                    f"graph builder created a Node with an undecorated "
                    f"callable."
                )
            object.__setattr__(self, '_hash_cache', hash_list_int(
                [
                    self.task._function.__module__,
                    self.task._function.__name__,
                    self.name,
                    self.args,
                ]
            ))
        return self._hash_cache

    def __str__(self):
        return (
            f"Node(fn={self.task._function.__name__}"
            f", name='{self.name}')" if self.name else
            f"Node(fn={self.task._function.__name__})"
        )

    def __lt__(self, other: "Node") -> bool:
        assert isinstance(other, Node)
        return (
            self.name or "",
            self.task._function.__name__,
            self.args,
        ) < (
            other.name or "",
            other.task._function.__name__,
            other.args,
        )


def reconstruct_call_args(literals, dep_values, dep_positions):
    """Rebuild a node's positional call args from its literals and its
    resolved data-dep values, placing each dep at the ORIGINAL position
    it occupied in the traced call.

    `dep_positions` is the list of arg indices that were data-dep
    handles at trace time (ascending; `dep_values` is in the same
    order). With it, an op whose first arg is a dep — e.g.
    `stack_files(downscale_stack(8, stack), config, *cal_masters)` in the
    downscaled pipeline — reconstructs as `[downscaled, config, cal…]`
    instead of collapsing the literal `config` into slot 0.

    `dep_positions is None` is the legacy path for Nodes built directly
    (not via `record()`): fall back to literals-then-deps, the historical
    "deps trailing" layout.
    """
    if dep_positions is None:
        return list(literals) + list(dep_values)
    pos = set(dep_positions)
    total = len(literals) + len(dep_values)
    lit_it = iter(literals)
    dep_it = iter(dep_values)
    return [next(dep_it) if i in pos else next(lit_it) for i in range(total)]


NodeStatusFn = Callable[[Node, Any, Any], None]

AnyTask: TypeAlias = GenericRemoteFunction | Callable

RemoteFunctionTuple: TypeAlias = Union[
    tuple[AnyTask],
    tuple[AnyTask, Any],
    tuple[AnyTask, Any, Any],
    tuple[AnyTask, Any, Any, Any],
    tuple[AnyTask, Any, Any, Any, Any],
    tuple[AnyTask, Any, Any, Any, Any, Any],
    tuple[AnyTask, Any, Any, Any, Any, Any, Any],
    tuple[AnyTask, Any, Any, Any, Any, Any, Any, Any],
    tuple[AnyTask, Any, Any, Any, Any, Any, Any, Any, Any],
]

TaskDescriptor: TypeAlias = Node | AnyTask | RemoteFunctionTuple


def canonicalize_node(task: TaskDescriptor) -> Node:
    if isinstance(task, Node):
        return task
    elif isinstance(task, tuple):
        fn, *args = task
        return Node(fn, *args)
    else:
        return Node(task)


GraphUpdateFn = Callable[..., Any]


class Graph(dict[TaskDescriptor, Iterable[TaskDescriptor]]):
    __slots__ = ("graph_update_fn", "node_status_fn", "ephemeral_files")
    graph_update_fn: GraphUpdateFn | None
    node_status_fn: NodeStatusFn | None
    # Optional `EphemeralFiles` registry built at eager-trace time in a
    # graph-builder @node (build_main_processing_graph / build_preview_graph).
    # GraphRuntime.__init__ folds it into the root runtime's shared
    # registry when this Graph becomes a child runtime.
    ephemeral_files: "Any | None"

    def __init__(
        self,
        *args,
        graph_update_fn: GraphUpdateFn | None = None,
        node_status_fn: NodeStatusFn | None = None,
        ephemeral_files: "Any | None" = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.graph_update_fn = graph_update_fn
        self.node_status_fn = node_status_fn
        self.ephemeral_files = ephemeral_files


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


@beartype
@dataclass
class GraphStatus:
    done_count: int = 0
    ready_nodes: list[Node] = field(default_factory=list)
    in_flight_nodes: list[Node] = field(default_factory=list)
    waiting_count: int = 0
    # Per-function-name counts. The UI uses these to render
    # "place_image 12/34" — total is every node of that type known to
    # the graph (including child runtimes), done is how many of those
    # have finished. Keyed by `node.task._function.__name__`.
    done_per_fn: Counter = field(default_factory=Counter)
    total_per_fn: Counter = field(default_factory=Counter)

    def __add__(self, rhs: "GraphStatus") -> "GraphStatus":
        return GraphStatus(
            done_count=self.done_count + rhs.done_count,
            ready_nodes=self.ready_nodes + rhs.ready_nodes,
            in_flight_nodes=self.in_flight_nodes + rhs.in_flight_nodes,
            waiting_count=self.waiting_count + rhs.waiting_count,
            done_per_fn=self.done_per_fn + rhs.done_per_fn,
            total_per_fn=self.total_per_fn + rhs.total_per_fn,
        )


@dataclass
class RunStats:
    num_nodes_run: int = 0
    trace: list[Node] = field(default_factory=list)


class ResultDict(dict[Node, Any]):

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


