from typing import Any

from dependency_graph import (
    CanonicalGraph,
    Graph,
    Node,
    ResultDict,
    canonicalize_node,
    canonicalize_graph,
)
from beartype import beartype


@beartype
def _tuple_values(tup: tuple) -> str:
    return ", ".join([str(t) for t in tup])


@beartype
def node_to_string(node: Node, exclude_args: bool = True) -> str:
    node = canonicalize_node(node)

    string = f"Node(fn={node.task._function.__name__}"
    if node.args and not exclude_args:
        string += f", args={_tuple_values(node.args)}"
    if node.name:
        string += f", name='{node.name}'"
    if node.priority:
        string += f", priority={node.priority}"
    string += ")"

    return string


@beartype
def node_to_string_tuple(node: Node) -> tuple:
    node = canonicalize_node(node)
    return (node.task._function.__name__, *[str(x) for x in node.args])


@beartype
def print_result_dict(results: ResultDict, indent: int = 0) -> None:
    for node, result in results.items():
        if type(node.task) is str:
            print(f"{' ' * indent}{node.task} = {result}")
            continue

        if type(result) is not ResultDict:
            print(
                f"{' ' * indent}{node.task._function.__name__} {node.args}: {result}"
            )
        else:
            print(f"{' ' * indent}{node.task._function.__name__} {node.args}:")
            print_result_dict(result, indent + 2)


@beartype
def print_graph(graph: CanonicalGraph | Graph) -> None:
    if not graph:
        print("Compute graph is empty")
        return

    graph = canonicalize_graph(graph)

    print(f"Printing compute graph with {len(graph)} edges:")
    for k, v in graph.items():
        if type(v) in [list, set]:
            strings = ",\n  ".join([node_to_string(i) for i in v])
            print(f"{node_to_string(k)}:\n  {strings}")
            continue

        assert False, f"Node dependencies have unsupported type {type(v)}: {v}"

    print("Done printing compute graph")
