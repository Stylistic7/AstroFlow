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


def _filter_of(node: Node) -> str | None:
    """Value bound to the `filter` parameter in `node.args`, if any.
    Pipeline ops carry `filter` as a literal arg on sinks like
    `merge_target_channel`, `place_image`, `merge_images`,
    `store_mosaic_color_channel_xisf`."""
    import inspect
    fn = getattr(node.task, "_function", node.task)
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    params = list(sig.parameters.keys())
    arg_map = {params[i]: node.args[i]
               for i in range(min(len(node.args), len(node.args), len(params)))}
    f = arg_map.get("filter")
    return None if f is None else str(f)


def _filter_membership(graph) -> dict[Node, set[str]]:
    """Per-node set of filters whose ancestor closure reaches the
    node. A node with empty set is "global" (no filter-bearing
    descendant) and should remain visible in every per-filter
    slice."""
    sinks_by_filter: dict[str, list] = {}
    for n in graph:
        f = _filter_of(n)
        if f is not None:
            sinks_by_filter.setdefault(f, []).append(n)
    membership: dict[Node, set[str]] = {n: set() for n in graph}
    for filter_key, sinks in sinks_by_filter.items():
        seen: set = set()
        stack = list(sinks)
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            membership.setdefault(n, set()).add(filter_key)
            for dep in graph.get(n, []):
                stack.append(dep)
    return membership


def gather_group_meta(
    rt: GraphRuntime,
    subgraphs_by_parent: dict[Node, GraphRuntime],
) -> dict[str, dict]:
    filters_by_node = _filter_membership(rt.graph)

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
            "filters": sorted(filters_by_node.get(node, ())),
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

    meta = {
        node_to_id(n): recurse(rt, n)
        for n in rt.graph if not _is_transparent(n)
    }

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
        if _is_transparent(n):
            continue
        path = _group_path_of(n)
        for k in range(1, len(path) + 1):
            by_prefix.setdefault(path[:k], []).append(n)

    for prefix, members in by_prefix.items():
        gid = _synthetic_group_id(prefix[-1])
        if gid in meta:
            continue
        member_states = [state_string(rt, m) for m in members]
        group_filters: set = set()
        for m in members:
            group_filters |= filters_by_node.get(m, set())
        meta[gid] = {
            "displayName": prefix[-1][0],  # the group name
            "state": aggregate_state(member_states),
            "filters": sorted(group_filters),
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


# Ops the workflow-graph viewer hides — uninteresting glue Nodes
# the trace machinery adds automatically (single-input, single-
# output, no semantic content the user cares about). Their incoming
# edges are stitched directly to their outgoing edges, so an A →
# compute_image_metadata → B chain renders as A → B.
_TRANSPARENT_OP_NAMES = frozenset({
    "compute_image_metadata",   # @makes_image trace hook
    "_register_tap_path_op",    # tap-path recorder
})


def _is_transparent(node: Node) -> bool:
    try:
        return node.task._function.__name__ in _TRANSPARENT_OP_NAMES
    except AttributeError:
        return False


def _resolve_through_transparent(dep, graph, _seen=None) -> set:
    """Walk back through transparent nodes; return the set of
    non-transparent upstreams. A transparent node's contribution
    to the rendered graph is its closest non-transparent
    ancestor(s) — so an edge X → transparent → Y becomes an edge
    X → Y."""
    if _seen is None:
        _seen = set()
    if dep in _seen:
        return set()
    _seen.add(dep)
    if not _is_transparent(dep):
        return {dep}
    upstreams = set()
    for d in graph.get(dep, []):
        upstreams |= _resolve_through_transparent(d, graph, _seen)
    return upstreams


def _build_subskeleton(
    rt: GraphRuntime,
    subgraphs_by_parent: dict[Node, GraphRuntime],
    group_prefix: tuple,
    indent: int,
) -> list[dict]:
    """Build the skeleton entries that live inside the lexical
    group at `group_prefix` — returned as ROOTS at this level
    (each root's `next` array nests its dependents recursively).

    Walks every Node whose `_group_path` starts with
    `group_prefix`. Nodes whose path matches exactly appear as
    direct entries; nodes whose path is deeper get grouped under
    synthetic `type: "group"` entries at this level."""
    L = len(group_prefix)
    graph = rt.graph

    def gp(n):
        return _group_path_of(n)

    def common_prefix_len(a, b):
        i = 0
        while i < len(a) and i < len(b) and a[i] == b[i]:
            i += 1
        return i

    in_scope = [
        n for n in graph
        if gp(n)[:L] == group_prefix and not _is_transparent(n)
    ]
    direct = sorted(n for n in in_scope if len(gp(n)) == L)
    subgroup_members: dict[tuple, list] = {}
    for n in in_scope:
        if len(gp(n)) > L:
            seg = gp(n)[L]
            subgroup_members.setdefault(seg, []).append(n)

    # entry_id → ("node", Node) or ("group_segment", segment)
    entry_kind: dict[str, tuple] = {}
    ordered_ids: list[str] = []  # stable insertion order
    for n in direct:
        nid = node_to_id(n)
        entry_kind[nid] = ("node", n)
        ordered_ids.append(nid)
    for seg in sorted(subgroup_members.keys()):
        gid = _synthetic_group_id(seg)
        entry_kind[gid] = ("group_segment", seg)
        ordered_ids.append(gid)

    # Project a Node onto its representative entry at THIS level
    # (its own id if its group_path matches `group_prefix`
    # exactly, else the synthetic group id one segment deeper).
    def project(n):
        path = gp(n)
        if len(path) == L:
            return node_to_id(n)
        return _synthetic_group_id(path[L])

    # Build forward adjacency at this level: dep_id → set of
    # dependent_ids (downstream). `rt.graph` is upstream-keyed
    # (node → deps), but the viewer's `next` array is downstream,
    # so we flip here. Transparent deps are resolved through to
    # their non-transparent ancestors, so an edge X →
    # compute_image_metadata → Y renders as X → Y.
    forward: dict[str, set[str]] = defaultdict(set)
    for node, deps in graph.items():
        if _is_transparent(node):
            continue
        if gp(node)[:L] != group_prefix:
            continue
        resolved_deps = set()
        for dep in deps:
            resolved_deps |= _resolve_through_transparent(dep, graph)
        for dep in resolved_deps:
            if gp(dep)[:L] != group_prefix:
                continue
            # The edge belongs at this level iff the nearest
            # common group containing both endpoints is exactly
            # `group_prefix`. Deeper-shared edges show up in the
            # deeper level's recursion.
            if common_prefix_len(gp(node), gp(dep)) != L:
                continue
            node_id = project(node)
            dep_id = project(dep)
            if node_id == dep_id:
                continue
            if node_id not in entry_kind or dep_id not in entry_kind:
                continue
            forward[dep_id].add(node_id)

    # Roots at this level: entries with no incoming downstream
    # edge. The skeleton starts here; dependents nest under
    # `next` recursively.
    has_incoming = {dependent for s in forward.values() for dependent in s}
    roots = [eid for eid in ordered_ids if eid not in has_incoming]

    # Recursive tree-flat build with dedupe — same shape the
    # pre-grouping `build_skeleton` produced for the legacy flat
    # path, just one level deep into either direct nodes or
    # synthetic group containers.
    existing: dict[str, dict] = {}

    def build_entry(entry_id: str) -> dict:
        if entry_id in existing:
            return {
                "id": entry_id,
                "type": existing[entry_id]["type"],
            }
        kind, payload = entry_kind[entry_id]
        if kind == "node":
            node = payload
            if node in subgraphs_by_parent:
                sub_rt = subgraphs_by_parent[node]
                result = {
                    "id": entry_id,
                    "type": "group",
                    "definition": _build_subskeleton(
                        sub_rt, subgraphs_by_parent,
                        group_prefix=(), indent=indent + 2,
                    ),
                }
            else:
                result = {"id": entry_id, "type": "execution"}
        else:  # synthetic lexical group
            seg = payload
            result = {
                "id": entry_id,
                "type": "group",
                "definition": _build_subskeleton(
                    rt, subgraphs_by_parent,
                    group_prefix=group_prefix + (seg,),
                    indent=indent + 2,
                ),
            }
        children = [build_entry(c) for c in sorted(forward.get(entry_id, set()))]
        if children:
            result["next"] = children
        existing[entry_id] = result
        return result

    return [build_entry(r) for r in roots]


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
