"""Visual layer between the executed `Graph` and the rendered
workflow-graph dict the frontend consumes.

A `VisualGraph` is shaped like a `Graph` (`dict[node, list[deps]]`,
upstream-keyed) but its nodes are `VisualNode` instances — each
wraps a canonical `Node` plus enough disambiguation to allow MORE
than one VisualNode per canonical Node. That's how render-time
transforms (see `graph_transforms.py`) can express things like
"duplicate this shared cal-master Node into each consumer's
`@workflow_group` scope" without touching execution semantics.

Build a VisualGraph from a `Graph` via `to_visual_graph`. Serialize
a VisualGraph to the `{skeleton, state}` dict the frontend reads
via `build_skeleton_from_visual` and `gather_state_from_visual`.
"""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from cache.types import NeverRun
from context import context
from dependency_graph.classes import Graph, Node, Skipped
from file_io.types import WithSidecar
from pathlib import Path


@dataclass(frozen=True)
class VisualNode:
    """One visual rendering of a canonical Node.

    Multiple VisualNodes may share the same `canonical` (visual
    duplication of a deduped computation). Equality / hash use
    (`canonical`, `scope`, `copy_tag`) so that two VisualNodes
    rendering the same Node in the same scope are deduped.

    - `canonical`: the underlying execution-time Node. Used for
      `/api/node_status/<id>` lookup (one Node = one status entry).
    - `scope`: the `@workflow_group` lexical scope this rendering
      lives in, expressed as a tuple of `(group_name, group_id)`
      segments. Empty tuple = top-level (no group).
    - `copy_tag`: extra disambiguator for two visual copies in the
      same scope (rare; reserved for future transforms).
    """
    canonical: Node
    scope: tuple = ()
    copy_tag: str = ""

    @property
    def visual_id(self) -> str:
        suffix = _scope_suffix(self.scope)
        tag = f":{self.copy_tag}" if self.copy_tag else ""
        return f"{_canonical_id(self.canonical)}@{suffix}{tag}"

    @property
    def op_name(self) -> str:
        return self.canonical.task._function.__name__


VisualGraph = dict[VisualNode, list[VisualNode]]


def _canonical_id(node: Node) -> str:
    """Stable string id of a canonical Node — what
    `/api/node_status/<id>` keys on."""
    return f"{node.task._function.__name__}_{hash(node)}"


def _scope_suffix(scope: tuple) -> str:
    """String form of a scope tuple, used in `visual_id`. `()` →
    `"root"`; nested scopes concat the per-segment `group_id`."""
    if not scope:
        return "root"
    return "_".join(gid for (_name, gid) in scope)


def _synthetic_group_id(segment: tuple) -> str:
    """Skeleton entry id for a synthetic lexical-group container.
    `segment` is a `(name, group_id)` pair."""
    name, gid = segment
    return f"group_{name}_{gid}"


def _group_path_of(node: Node) -> tuple:
    """The lexical scope a canonical Node was traced into.
    `tracing.record()` sets `_group_path` to the active
    `@workflow_group` stack; Nodes that bypass `record()` (e.g.
    the `run_pipeline_done` sentinel) default to `()`."""
    return getattr(node, "_group_path", ())


def to_visual_graph(graph: Graph) -> VisualGraph:
    """Identity-mapped: one VisualNode per canonical Node, with
    scope taken from the Node's `_group_path`. The transforms
    in `graph_transforms.py` then rewrite this into the sanitized
    shape the viewer displays."""
    nodes: dict[Node, VisualNode] = {}
    for k, deps in graph.items():
        if not isinstance(k, Node):
            continue
        nodes.setdefault(k, VisualNode(canonical=k, scope=_group_path_of(k)))
        for d in deps:
            if isinstance(d, Node):
                nodes.setdefault(d, VisualNode(canonical=d, scope=_group_path_of(d)))

    out: VisualGraph = {}
    for k, deps in graph.items():
        if not isinstance(k, Node):
            continue
        out[nodes[k]] = [nodes[d] for d in deps if isinstance(d, Node)]
    # Include dep-only nodes (referenced from a key's deps but not
    # themselves a key in `graph`).
    for vn in nodes.values():
        out.setdefault(vn, [])
    return out


# ---------------------------------------------------------------------------
# Serialization: VisualGraph → `{skeleton, state}` for the frontend.
# ---------------------------------------------------------------------------


def _state_string(rt, node: Node) -> str:
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


def _aggregate_state(states) -> str:
    if not states:
        return "NO_STATE_RUNTIME"
    if "FAILED" in states:
        return "FAILED"
    if any(s not in ("SUCCEEDED", "SKIPPED", "NO_STATE_RUNTIME") for s in states):
        return "RUNNING"
    succeeded = any(s in ("SUCCEEDED", "SKIPPED") for s in states)
    not_run = any(s == "NO_STATE_RUNTIME" for s in states)
    if succeeded and not_run:
        # Partial: some members done, some not started yet — treat
        # as in-progress so the group doesn't flash back to "never
        # run" while the rest catches up.
        return "RUNNING"
    if succeeded:
        return "SUCCEEDED"
    return "NO_STATE_RUNTIME"


def _filter_of(node: Node) -> str | None:
    """Value bound to the `filter` parameter in `node.args`, if any.
    Pipeline ops carry `filter` as a literal arg on sinks like
    `merge_target_channel`, `place_image`, `merge_images`."""
    import inspect
    fn = getattr(node.task, "_function", node.task)
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    params = list(sig.parameters.keys())
    arg_map = {params[i]: node.args[i]
               for i in range(min(len(node.args), len(params)))}
    f = arg_map.get("filter")
    return None if f is None else str(f)


def _filter_membership(vg: VisualGraph) -> dict[VisualNode, set[str]]:
    """Per-VisualNode set of filters whose ancestor closure reaches
    that node (walking deps upstream from filter-bearing sinks).
    Empty set = "global", visible in every per-filter slice."""
    sinks_by_filter: dict[str, list[VisualNode]] = {}
    for vn in vg:
        f = _filter_of(vn.canonical)
        if f is not None:
            sinks_by_filter.setdefault(f, []).append(vn)
    membership: dict[VisualNode, set[str]] = {vn: set() for vn in vg}
    for filter_key, sinks in sinks_by_filter.items():
        seen: set = set()
        stack = list(sinks)
        while stack:
            vn = stack.pop()
            if vn in seen:
                continue
            seen.add(vn)
            membership.setdefault(vn, set()).add(filter_key)
            for dep in vg.get(vn, []):
                stack.append(dep)
    return membership


def _resolved_filename(rt, node: Node) -> str | None:
    """Resolve the on-disk output filename for a Node, if known by
    the runtime. Used by the frontend's `imageUrl` field."""
    try:
        fragment = rt.filenames.get(node)
        if fragment is None:
            return None
        resolved = fragment.resolve_at(context.output_dir)
        if os.path.exists(resolved):
            return os.fspath(resolved)
    except (AttributeError, KeyError, TypeError):
        pass
    return None


def gather_state_from_visual(
    vg: VisualGraph,
    rt,
    subgraphs_by_parent: dict[Node, Any] | None = None,
) -> dict[str, dict]:
    """One meta entry per VisualNode (keyed by `visual_id`) plus
    one synthetic-group entry per distinct scope prefix. The
    state field comes from the canonical Node's runtime result."""
    if subgraphs_by_parent is None:
        subgraphs_by_parent = {}
    filters_by_vn = _filter_membership(vg)
    meta: dict[str, dict] = {}

    for vn in vg:
        node = vn.canonical
        entry: dict[str, Any] = {
            "displayName": vn.op_name,
            "state": _state_string(rt, node),
            "filters": sorted(filters_by_vn.get(vn, ())),
            "options": {
                "imageUrl": f"/api/open_image/{_resolved_filename(rt, node)}",
            },
        }
        if node.args:
            entry["description"] = str(node.args[0])
        if node.name:
            entry["description"] = node.name
        if node in subgraphs_by_parent:
            sub_rt = subgraphs_by_parent[node]
            sub_vg = to_visual_graph(sub_rt.graph)
            entry["hasControlNode"] = True
            entry["groupMeta"] = gather_state_from_visual(
                sub_vg, sub_rt, sub_rt.subgraphs_by_parent,
            )
            if _subgraph_has_exception(sub_rt, subgraphs_by_parent):
                entry["state"] = "FAILED"
        meta[vn.visual_id] = entry

    # Synthetic group entries: for every distinct prefix of every
    # VisualNode's scope, one collapsible group entry. State and
    # filters aggregate across all members reachable through this
    # prefix.
    by_prefix: dict[tuple, list[VisualNode]] = defaultdict(list)
    for vn in vg:
        for k in range(1, len(vn.scope) + 1):
            by_prefix[vn.scope[:k]].append(vn)

    # Map each prefix to the set of one-deeper child prefixes that
    # exist — these are nested synthetic groups that should appear
    # in this group's `groupMeta` (so the frontend can look up the
    # nested group's displayName when rendering it as a child).
    children_of: dict[tuple, set[tuple]] = defaultdict(set)
    for prefix in by_prefix:
        if len(prefix) >= 2:
            children_of[prefix[:-1]].add(prefix)

    # Process deepest prefixes first so a parent's `groupMeta` can
    # reference its already-built nested-group entries.
    for prefix in sorted(by_prefix, key=lambda p: -len(p)):
        members = by_prefix[prefix]
        gid = _synthetic_group_id(prefix[-1])
        if gid in meta:
            continue
        states = [_state_string(rt, m.canonical) for m in members]
        group_filters: set = set()
        for m in members:
            group_filters |= filters_by_vn.get(m, set())
        # groupMeta contains DIRECT children: VisualNode entries at
        # exactly this prefix, plus nested synthetic-group entries
        # one segment deeper.
        direct_members = [m for m in members if m.scope == prefix]
        group_meta = {
            m.visual_id: meta[m.visual_id]
            for m in direct_members if m.visual_id in meta
        }
        for child_prefix in children_of.get(prefix, ()):
            child_gid = _synthetic_group_id(child_prefix[-1])
            if child_gid in meta:
                group_meta[child_gid] = meta[child_gid]
        meta[gid] = {
            "displayName": prefix[-1][0],
            "state": _aggregate_state(states),
            "filters": sorted(group_filters),
            "hasControlNode": True,
            "groupMeta": group_meta,
        }
    return meta


def _subgraph_has_exception(rt, subgraphs_by_parent) -> bool:
    if any(isinstance(rt.results.get(n, NeverRun), Exception) for n in rt.graph):
        return True
    for n in rt.graph:
        if n in subgraphs_by_parent:
            if _subgraph_has_exception(subgraphs_by_parent[n], subgraphs_by_parent):
                return True
    return False


def build_skeleton_from_visual(
    vg: VisualGraph,
    rt,
    subgraphs_by_parent: dict[Node, Any] | None = None,
) -> list[dict]:
    """Build the recursive skeleton tree the frontend renders.

    Each lexical scope prefix becomes a synthetic `type: "group"`
    entry; VisualNodes at the exact scope appear as `type:
    "execution"` (or `type: "group"` if their canonical Node has
    a runtime sub-graph)."""
    if subgraphs_by_parent is None:
        subgraphs_by_parent = {}
    return _build_subskeleton_visual(vg, rt, subgraphs_by_parent, group_prefix=())


def _build_subskeleton_visual(
    vg: VisualGraph,
    rt,
    subgraphs_by_parent: dict[Node, Any],
    group_prefix: tuple,
) -> list[dict]:
    L = len(group_prefix)

    direct: list[VisualNode] = []
    subgroup_members: dict[tuple, list[VisualNode]] = {}
    for vn in vg:
        if vn.scope[:L] != group_prefix:
            continue
        if len(vn.scope) == L:
            direct.append(vn)
        else:
            seg = vn.scope[L]
            subgroup_members.setdefault(seg, []).append(vn)
    direct.sort(key=lambda vn: vn.visual_id)

    entry_kind: dict[str, tuple] = {}
    ordered_ids: list[str] = []
    for vn in direct:
        entry_kind[vn.visual_id] = ("node", vn)
        ordered_ids.append(vn.visual_id)
    for seg in sorted(subgroup_members):
        gid = _synthetic_group_id(seg)
        entry_kind[gid] = ("group_segment", seg)
        ordered_ids.append(gid)

    def project(vn: VisualNode) -> str:
        if len(vn.scope) == L:
            return vn.visual_id
        return _synthetic_group_id(vn.scope[L])

    # Forward adjacency at this level (dep_id → downstream_ids).
    # The VisualGraph is already de-transparented and duplicated by
    # the transforms, so no scope-matching heuristics here — just a
    # straight projection.
    forward: dict[str, set[str]] = defaultdict(set)
    for downstream, deps in vg.items():
        if downstream.scope[:L] != group_prefix:
            continue
        d_id = project(downstream)
        for upstream in deps:
            if upstream.scope[:L] != group_prefix:
                continue
            u_id = project(upstream)
            if u_id == d_id:
                continue
            if u_id not in entry_kind or d_id not in entry_kind:
                continue
            forward[u_id].add(d_id)

    has_incoming = {x for s in forward.values() for x in s}
    roots = [eid for eid in ordered_ids if eid not in has_incoming]

    existing: dict[str, dict] = {}

    def build_entry(entry_id: str) -> dict:
        if entry_id in existing:
            return {"id": entry_id, "type": existing[entry_id]["type"]}
        kind, payload = entry_kind[entry_id]
        if kind == "node":
            vn: VisualNode = payload
            if vn.canonical in subgraphs_by_parent:
                sub_rt = subgraphs_by_parent[vn.canonical]
                sub_vg = to_visual_graph(sub_rt.graph)
                result = {
                    "id": entry_id,
                    "type": "group",
                    "definition": build_skeleton_from_visual(
                        sub_vg, sub_rt, sub_rt.subgraphs_by_parent,
                    ),
                }
            else:
                result = {"id": entry_id, "type": "execution"}
        else:
            seg = payload
            result = {
                "id": entry_id,
                "type": "group",
                "definition": _build_subskeleton_visual(
                    vg, rt, subgraphs_by_parent,
                    group_prefix=group_prefix + (seg,),
                ),
            }
        children = [build_entry(c) for c in sorted(forward.get(entry_id, set()))]
        if children:
            result["next"] = children
        existing[entry_id] = result
        return result

    return [build_entry(r) for r in roots]
