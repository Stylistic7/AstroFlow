from collections import defaultdict
import os
from typing import Any
import traceback
import ray
from pathlib import Path

from dependency_graph import Node, TopologicalSorterWithStatus, Skipped
from cache.types import NeverRun
from context import context
from file_io.types import WithSidecar

_DEBUG = False


def debug_print(message):
    if _DEBUG:
        print(message)


class WorkflowGraph(dict):
    pass


def node_to_id(node: Node) -> str:
    return str(node.task._function.__name__) + "_" + str(hash(node))


def gather_group_meta(
    ts: TopologicalSorterWithStatus,
    subgraphs_by_parent: dict[Node, TopologicalSorterWithStatus],
) -> dict[str, dict]:
    def state_string(ts, node):
        result = ts.results.get(node, NeverRun)
        return (
            "NO_STATE_RUNTIME"
            if result == NeverRun
            else "SKIPPED"
            if result == Skipped
            else "FAILED"
            if isinstance(result, Exception)
            else "SUCCEEDED"
        )

    def subgraph_contains_exception(ts, node):
        if node in subgraphs_by_parent:
            sub_ts = subgraphs_by_parent[node]
            if any(
                subgraph_contains_exception(sub_ts, n) for n in sub_ts.graph
            ):
                return True

        return isinstance(ts.results.get(node, NeverRun), Exception)

    def recurse(ts: TopologicalSorterWithStatus, node: Node):
        # TODO: Split this out as an independent function -- it may get complex.
        filename = None
        try:
            if hasattr(node.task, "filename_fn"):
                args = ts.complete_args[node]
                filename = node.task.filename_fn(*args).resolve_at(
                    context.output_dir
                )
                if os.path.exists(filename):
                    filename = os.fspath(filename)
        except:
            pass

        meta = {
            "displayName": node.task._function.__name__,
            "state": state_string(ts, node),
            "options": {
                "imageUrl": f"/api/open_image/{filename}",
            },
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
            if subgraph_contains_exception(sub_ts, node):
                meta["state"] = "FAILED"
        return meta

    return {node_to_id(n): recurse(ts, n) for n in ts.graph}


def build_skeleton(
    ts: TopologicalSorterWithStatus,
    subgraphs_by_parent: dict[Node, TopologicalSorterWithStatus],
    indent: int = 0,
) -> list[dict]:
    existing_nodes = {}

    debug_print(
        f"{' ' * indent} build_skeleton() called for ts with {len(ts.graph)} entries"
    )

    graph = ts.graph
    reverse = defaultdict(set)
    for node, deps in graph.items():
        for dep in deps:
            reverse[dep].add(node)
    roots = sorted(n for n, deps in graph.items() if not deps)

    def build(node, indent):
        key = node_to_id(node)
        debug_print(f"{' ' * indent} build() called for node {node}")

        if (ts, key) in existing_nodes:
            debug_print(f"{' ' * indent} key {key} is in existing_nodes")
            return {
                "id": existing_nodes[ts, key]["id"],
                "type": existing_nodes[ts, key]["type"],
            }

        if node in subgraphs_by_parent:
            sub_ts = subgraphs_by_parent[node]
            debug_print(
                f"{' ' * indent} calling build_skeleton() for subgraph node {node}"
            )
            result = {
                "id": key,
                "type": "group",
                "definition": build_skeleton(
                    sub_ts, subgraphs_by_parent, indent=indent + 2
                ),
            }
        else:
            result = {"id": key, "type": "execution"}

        debug_print(
            f"{' ' * indent} calling build() for {len(reverse[node])} "
            f"children of {node}"
        )
        children = [build(n, indent=indent + 2) for n in sorted(reverse[node])]
        if children:
            result["next"] = children

        existing_nodes[ts, key] = result
        debug_print(f"{' ' * indent} putting key {key} into existing_nodes")

        return result

    return [build(n, indent=indent) for n in roots]


def to_workflow_graph(
    top_level_ts: TopologicalSorterWithStatus,
    subgraphs: list[TopologicalSorterWithStatus],
):
    ts_by_parent = {
        sg.parent_node: sg for sg in subgraphs if sg.parent_node is not None
    }

    return {
        "skeleton": build_skeleton(top_level_ts, ts_by_parent),
        "state": gather_group_meta(top_level_ts, ts_by_parent),
    }


def store_workflow_graph(
    key: Any,
    ts: TopologicalSorterWithStatus,
    subgraphs: list[TopologicalSorterWithStatus],
):
    context.or_equals(
        "workflow_graphs", {key: to_workflow_graph(ts, subgraphs)}
    )


def store_node_status(node: Node, result: Any, logs: list[Any]):
    if isinstance(result, ray.ObjectRef):
        result = ray.get(result)

    status_dict = dict(result_type=str(type(result).__name__), logs=logs)

    if isinstance(result, Exception):
        status_dict["exception"] = result
        status_dict["stack_trace"] = traceback.format_exc()
    elif isinstance(result, (Path, WithSidecar)):
        status_dict["path"] = str(result)

    context[("node_status", node_to_id(node))] = status_dict
