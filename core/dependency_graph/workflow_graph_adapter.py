from collections import defaultdict
import os
from typing import Any
import traceback
import ray
from pathlib import Path

from dependency_graph import Node, GraphRuntime, Skipped
from cache.types import NeverRun
from context import context
from file_io.types import WithSidecar

_DEBUG = False


def debug_print(message):
    if _DEBUG:
        print(message)


class WorkflowGraph(dict):
    pass


class WorkflowGraphsDict(dict):
    """Dict mapping workflow graph keys (tuples) to ContextRef objects."""
    def __hash__(self):
        from hashing import hash_list_int
        from hashing.hashing import hash_list
        return hash_list_int([hash_list(sorted(str(k) for k in self.keys()))])

    def __str__(self):
        return f"WorkflowGraphsDict({len(self)})"


def node_to_id(node: Node) -> str:
    return str(node.task._function.__name__) + "_" + str(hash(node))


def gather_group_meta(
    rt: GraphRuntime,
    subgraphs_by_parent: dict[Node, GraphRuntime],
) -> dict[str, dict]:
    def state_string(rt, node):
        result = rt.results.get(node, NeverRun)
        return (
            "NO_STATE_RUNTIME"
            if result == NeverRun
            else "SKIPPED"
            if result == Skipped
            else "FAILED"
            if isinstance(result, Exception)
            else "SUCCEEDED"
        )

    def subgraph_contains_exception(rt, node):
        if node in subgraphs_by_parent:
            sub_rt = subgraphs_by_parent[node]
            if any(
                subgraph_contains_exception(sub_rt, n) for n in sub_rt.graph
            ):
                return True

        return isinstance(rt.results.get(node, NeverRun), Exception)

    def recurse(rt: GraphRuntime, node: Node):
        filename = None
        try:
            fragment = rt.filenames.get(node)
            if fragment is not None:
                resolved = fragment.resolve_at(context.output_dir)
                if os.path.exists(resolved):
                    filename = os.fspath(resolved)
        except (AttributeError, KeyError, TypeError):
            pass

        meta = {
            "displayName": node.task._function.__name__,
            "state": state_string(rt, node),
            "options": {
                "imageUrl": f"/api/open_image/{filename}",
            },
        }
        if node.args:
            meta["description"] = str(node.args[0])
        if node.name:
            meta["description"] = node.name
        if node in subgraphs_by_parent:
            sub_rt = subgraphs_by_parent[node]
            meta["hasControlNode"] = True
            meta["groupMeta"] = {
                node_to_id(n): recurse(sub_rt, n) for n in sub_rt.graph
            }
            if subgraph_contains_exception(sub_rt, node):
                meta["state"] = "FAILED"
        return meta

    return {node_to_id(n): recurse(rt, n) for n in rt.graph}


def build_skeleton(
    rt: GraphRuntime,
    subgraphs_by_parent: dict[Node, GraphRuntime],
    indent: int = 0,
) -> list[dict]:
    existing_nodes = {}

    debug_print(
        f"{' ' * indent} build_skeleton() called for rt with {len(rt.graph)} entries"
    )

    graph = rt.graph
    reverse = defaultdict(set)
    for node, deps in graph.items():
        for dep in deps:
            reverse[dep].add(node)
    roots = sorted(n for n, deps in graph.items() if not deps)

    def build(node, indent):
        key = node_to_id(node)
        debug_print(f"{' ' * indent} build() called for node {node}")

        if (rt, key) in existing_nodes:
            debug_print(f"{' ' * indent} key {key} is in existing_nodes")
            return {
                "id": existing_nodes[rt, key]["id"],
                "type": existing_nodes[rt, key]["type"],
            }

        if node in subgraphs_by_parent:
            sub_rt = subgraphs_by_parent[node]
            debug_print(
                f"{' ' * indent} calling build_skeleton() for subgraph node {node}"
            )
            result = {
                "id": key,
                "type": "group",
                "definition": build_skeleton(
                    sub_rt, subgraphs_by_parent, indent=indent + 2
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

        existing_nodes[rt, key] = result
        debug_print(f"{' ' * indent} putting key {key} into existing_nodes")

        return result

    return [build(n, indent=indent) for n in roots]


def to_workflow_graph(root: GraphRuntime):
    by_parent = root.subgraphs_by_parent

    return {
        "skeleton": build_skeleton(root, by_parent),
        "state": gather_group_meta(root, by_parent),
    }


def store_workflow_graph(key: Any, rt: GraphRuntime):
    from cache.context_ref import ContextRef

    wf_graph = to_workflow_graph(rt)
    cache_key = ("workflow_graphs", str(key))
    new_ref = ContextRef.store(cache_key, wf_graph)
    context.or_equals("workflow_graphs", {key: new_ref})


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
