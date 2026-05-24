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

    meta = {node_to_id(n): recurse(rt, n) for n in rt.graph}

    # Synthetic lexical-group entries: every distinct prefix of
    # every Node's `_group_path` gets its own meta entry, keyed by
    # the synthetic id `build_skeleton` uses. Aggregates state
    # from descendants (FAILED beats RUNNING beats SUCCEEDED beats
    # NO_STATE_RUNTIME).
    def aggregate_state(states):
        if not states:
            return "NO_STATE_RUNTIME"
        if "FAILED" in states:
            return "FAILED"
        if any(s not in ("SUCCEEDED", "SKIPPED", "NO_STATE_RUNTIME") for s in states):
            return "RUNNING"
        if all(s in ("SUCCEEDED", "SKIPPED") for s in states):
            return "SUCCEEDED"
        return "NO_STATE_RUNTIME"

    by_prefix: dict[tuple, list] = {}  # prefix tuple → nodes inside
    for n in rt.graph:
        path = _group_path_of(n)
        for k in range(1, len(path) + 1):
            by_prefix.setdefault(path[:k], []).append(n)

    for prefix, members in by_prefix.items():
        gid = _synthetic_group_id(prefix[-1])
        if gid in meta:
            continue
        member_states = [state_string(rt, m) for m in members]
        meta[gid] = {
            "displayName": prefix[-1][0],  # the group name
            "state": aggregate_state(member_states),
            "hasControlNode": True,
            "groupMeta": {node_to_id(m): meta[node_to_id(m)] for m in members
                          if node_to_id(m) in meta},
        }

    return meta


def _group_path_of(node: Node) -> tuple:
    """Lexical workflow-group path attached to a Node by
    `tracing.record()` when the node was created inside one or
    more `@workflow_group`-decorated helpers. Empty tuple when no
    grouping was active (top-level pipeline-body nodes)."""
    return getattr(node, "_group_path", ())


def _synthetic_group_id(segment: tuple) -> str:
    """Stable id for a synthetic lexical group entry. `segment`
    is a `(name, group_id)` pair from a Node's `_group_path`."""
    name, gid = segment
    return f"group_{name}_{gid}"


def _build_subskeleton(
    rt: GraphRuntime,
    subgraphs_by_parent: dict[Node, GraphRuntime],
    group_prefix: tuple,
    indent: int,
) -> list[dict]:
    """Build the skeleton entries that live inside the lexical
    group at `group_prefix`. Walks every Node whose `_group_path`
    starts with `group_prefix`; nodes whose path matches exactly
    appear as direct entries, nodes whose path is deeper get
    grouped under synthetic `type: "group"` entries at this
    level."""
    L = len(group_prefix)
    graph = rt.graph

    # Nodes within this group's scope (prefix matches), partitioned
    # into "direct" (path is exactly group_prefix) and deeper
    # (grouped by the next-segment of their path).
    in_scope = [n for n in graph if _group_path_of(n)[:L] == group_prefix]
    direct = [n for n in in_scope if len(_group_path_of(n)) == L]
    subgroup_members: dict[tuple, list] = {}
    for n in in_scope:
        if len(_group_path_of(n)) > L:
            seg = _group_path_of(n)[L]
            subgroup_members.setdefault(seg, []).append(n)

    # entry_id → entry dict, in insertion order.
    entries: dict[str, dict] = {}

    # Direct nodes — same as the legacy flat path, plus the
    # existing sub-runtime "group" treatment for nodes whose
    # task returned a Graph at runtime.
    for n in direct:
        nid = node_to_id(n)
        if n in subgraphs_by_parent:
            sub_rt = subgraphs_by_parent[n]
            entries[nid] = {
                "id": nid,
                "type": "group",
                "definition": _build_subskeleton(
                    sub_rt, subgraphs_by_parent, group_prefix=(), indent=indent + 2,
                ),
            }
        else:
            entries[nid] = {"id": nid, "type": "execution"}

    # Synthetic lexical-group entries — one per distinct next
    # segment. Recurse into each with a deeper group_prefix; the
    # returned definition is the group's contents (which may itself
    # contain further nested groups).
    for seg in sorted(subgroup_members.keys()):
        gid = _synthetic_group_id(seg)
        entries[gid] = {
            "id": gid,
            "type": "group",
            "definition": _build_subskeleton(
                rt, subgraphs_by_parent,
                group_prefix=group_prefix + (seg,),
                indent=indent + 2,
            ),
        }

    # Project every dep edge onto THIS level. For an edge
    # `from_node → to_node`, both endpoints must live at or below
    # `group_prefix`; the edge belongs at this level iff the
    # nearest common group containing both is exactly
    # `group_prefix`. Each endpoint maps either to its own entry
    # (if its group path matches the prefix exactly) or to the
    # synthetic-group entry one level deeper (if its path is
    # deeper). The frontend then renders inter-group edges
    # between collapsed group nodes and intra-group edges inside
    # each group's `definition`.
    def project(n):
        path = _group_path_of(n)
        if len(path) == L:
            return node_to_id(n)
        return _synthetic_group_id(path[L])

    def common_prefix_len(a, b):
        i = 0
        while i < len(a) and i < len(b) and a[i] == b[i]:
            i += 1
        return i

    seen_edges: set[tuple] = set()
    for from_node, deps in graph.items():
        fp = _group_path_of(from_node)
        if fp[:L] != group_prefix:
            continue
        for to_node in deps:
            tp = _group_path_of(to_node)
            if tp[:L] != group_prefix:
                continue
            if common_prefix_len(fp, tp) != L:
                continue  # belongs in a deeper level's skeleton
            from_id = project(from_node)
            to_id = project(to_node)
            if from_id == to_id:
                continue
            if from_id not in entries or to_id not in entries:
                continue
            edge_key = (from_id, to_id)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            entries[from_id].setdefault("next", []).append({
                "id": to_id,
                "type": entries[to_id]["type"],
            })

    return list(entries.values())


def build_skeleton(
    rt: GraphRuntime,
    subgraphs_by_parent: dict[Node, GraphRuntime],
    indent: int = 0,
) -> list[dict]:
    """Build the workflow-graph skeleton: a hierarchy of entries
    (each `{"id", "type", "next"?, "definition"?}`) the frontend
    renders.

    Two grouping mechanisms produce `type: "group"` entries:

    1. **Runtime sub-runtime** — a Node whose task returned a
       `Graph` at runtime got a child `GraphRuntime` in
       `subgraphs_by_parent`; its definition is the recursive
       skeleton of that sub-runtime.
    2. **Lexical workflow_group** — Nodes created inside a
       `@workflow_group("Name")`-decorated helper share a
       `_group_path` prefix; a synthetic group entry collects
       them.

    Both produce the same `{type: "group", definition: [...]}`
    shape, so the frontend doesn't distinguish them."""
    debug_print(
        f"{' ' * indent} build_skeleton() called for rt with {len(rt.graph)} entries"
    )
    return _build_subskeleton(rt, subgraphs_by_parent, group_prefix=(), indent=indent)


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


# node_status is read only by the UI (/api/node_status/<id>), which
# json-dumps it for display. Captured structlog event dicts can carry
# arbitrarily large objects (a logged Graph/Stack/MetadataDict), which
# made single entries 100+ MB. Stringify and bound everything before it
# reaches the cache — the UI only ever shows this as text anyway.
_LOG_FIELD_MAX = 4000
_LOG_MAX_ENTRIES = 1000


def _stringify(v: Any) -> str:
    if isinstance(v, str):
        return v
    try:
        return repr(v)
    except Exception:
        return f"<unreprable {type(v).__name__}>"


def _truncate(s: str) -> str:
    if len(s) <= _LOG_FIELD_MAX:
        return s
    return s[:_LOG_FIELD_MAX] + f"…[+{len(s) - _LOG_FIELD_MAX} chars]"


def _bounded_logs(logs: Any) -> list:
    if not isinstance(logs, list):
        return [{"event": _truncate(_stringify(logs))}]
    out: list = []
    if len(logs) > _LOG_MAX_ENTRIES:
        out.append(
            {"event": f"[{len(logs) - _LOG_MAX_ENTRIES} earlier entries elided]"}
        )
    for e in logs[-_LOG_MAX_ENTRIES:]:
        if isinstance(e, dict):
            out.append({k: _truncate(_stringify(v)) for k, v in e.items()})
        else:
            out.append({"event": _truncate(_stringify(e))})
    return out


def store_node_status(node: Node, result: Any, logs: list[Any]):
    if isinstance(result, ray.ObjectRef):
        result = ray.get(result)

    status_dict = dict(
        result_type=str(type(result).__name__), logs=_bounded_logs(logs)
    )

    if isinstance(result, Exception):
        # repr, not the raw exception: a pickled exception drags its
        # traceback frames/locals (more 100 MB blobs).
        status_dict["exception"] = _truncate(_stringify(result))
        status_dict["stack_trace"] = traceback.format_exc()
    elif isinstance(result, (Path, WithSidecar)):
        status_dict["path"] = str(result)

    context[("node_status", node_to_id(node))] = status_dict
