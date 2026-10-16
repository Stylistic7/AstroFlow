from flask import Blueprint, Response, request
import json
import os
import subprocess

from context import local_context as context
from third_party_tools import pixinsight

compute_graphs_bp = Blueprint("compute_graphs", __name__)

_EMPTY_WORKFLOW_GRAPH = {"skeleton": [], "state": {}}


@compute_graphs_bp.route("/api/compute_graph/status")
def compute_graph_status():
    return context.compute_graph_status or {}


# Workflow-graph routes. Today user pipelines don't call
# `set_workflow_key(...)` from their bodies, so the `workflow_graphs`
# context dict is empty and these routes return the empty-graph
# sentinel. The pipeline / filter segments are accepted but unused;
# they reserve the URL shape for when pipelines start tagging their
# trace with workflow keys.
def _workflow_graph(object_type: str, object_id: str, pipeline_name) -> dict:
    """One object's workflow-graph snapshot, prepared for the frontend the
    same way at every hierarchy level: durable state overlaid, server-only
    fields stripped. The stack and target routes used to return the raw
    snapshot, and the DagNode constructor rejects its `filters` key — every
    target page in the cart errored once targets were cart items of their
    own (2026-09-20)."""
    if not pipeline_name:
        return _EMPTY_WORKFLOW_GRAPH
    wf = context["workflow_graphs"] or {}
    snapshot = _resolve_ref(wf.get((object_type, object_id, pipeline_name)))
    if snapshot is None:
        return _EMPTY_WORKFLOW_GRAPH
    return _strip_server_internal_fields(_overlay_durable_state(snapshot))


@compute_graphs_bp.route("/api/workflow_graph/stack/<stack_id>")
@compute_graphs_bp.route("/api/workflow_graph/stack/<stack_id>/<pipeline_name>")
def workflow_graph_for_stack(stack_id, pipeline_name=None):
    return _workflow_graph("stack", stack_id, pipeline_name)


@compute_graphs_bp.route("/api/workflow_graph/target/<target_id>")
@compute_graphs_bp.route("/api/workflow_graph/target/<target_id>/<pipeline_name>")
def workflow_graph_for_target(target_id, pipeline_name=None):
    return _workflow_graph("target", target_id, pipeline_name)


@compute_graphs_bp.route("/api/workflow_graph/mosaic/<mosaic_id>")
@compute_graphs_bp.route("/api/workflow_graph/mosaic/<mosaic_id>/<pipeline_name>")
@compute_graphs_bp.route("/api/workflow_graph/mosaic/<mosaic_id>/<pipeline_name>/<filter>")
def workflow_graph_for_mosaic(mosaic_id, pipeline_name=None, filter=None):
    """One workflow graph per `(mosaic, id, pipeline)`, snapshotted
    at pipeline-trace time and refreshed with live runtime state
    by the runtime's `graph_update_fn`. The optional `<filter>`
    segment slices the snapshot to nodes whose ancestor closure
    reaches a `filter`-bearing sink — sliced on read here rather
    than written as a separate snapshot, so `graph_update_fn`
    stays filter-unaware.

    Query param `?raw=1` skips the `sanitize_for_display` transform
    pipeline and serializes the trace's raw graph (transparent ops
    visible, shared Nodes not duplicated). Re-traces the pipeline
    on the spot, so it's slower but only used for debugging when
    the sanitized view looks wrong."""
    if not pipeline_name:
        return _EMPTY_WORKFLOW_GRAPH
    if request.args.get("raw") == "1":
        snapshot = _build_raw_workflow_graph(mosaic_id, pipeline_name)
    else:
        wf = context["workflow_graphs"] or {}
        snapshot = _resolve_ref(wf.get(("mosaic", mosaic_id, pipeline_name)))
    if snapshot is None:
        return _EMPTY_WORKFLOW_GRAPH
    snapshot = _overlay_durable_state(snapshot)
    if filter:
        snapshot = _slice_workflow_graph_by_filter(snapshot, filter)
    return _strip_server_internal_fields(snapshot)


def _build_raw_workflow_graph(mosaic_id: str, pipeline_name: str) -> dict | None:
    """Re-trace `pipeline_name` for the mosaic and serialize WITHOUT
    the sanitize-for-display transforms. Returns `None` if the
    mosaic or pipeline isn't found.

    Local imports keep the heavy module deps off the import-time
    fast path."""
    from dependency_graph.graph_runtime import GraphRuntime
    from dependency_graph.workflow_graph_adapter import to_workflow_graph
    from decisions.settings import ProcessingConfig
    from mosaics.types import Mosaic
    from pipelines.registry import get_pipeline

    mosaic_ref = (context["mosaics"] or {}).get(mosaic_id)
    if mosaic_ref is None:
        return None
    mosaic = mosaic_ref.get()
    settings = context["settings"]
    config = ProcessingConfig(
        calibration_stacks=context["calibration_stacks"] or {},
        development_settings=settings.development_settings,
        min_frames_in_stack=settings.clustering_settings.min_frames_in_stack,
    )
    try:
        runner = get_pipeline(pipeline_name, processes=Mosaic)
    except Exception:
        return None
    graph = runner(mosaic, config)
    if graph is None:
        return None
    rt = GraphRuntime(graph)
    return to_workflow_graph(rt, raw=True)


def _overlay_durable_state(snapshot: dict) -> dict:
    """Rewrite each node's `state` from DURABLE signals so the graph shows
    the most progress EVER made, not just this session. Mutates & returns
    `snapshot` (a fresh, deserialized dict).

    Per leaf: its baked `options.expectedPath` on disk ⇒ SUCCEEDED (a file
    implies the node ran — fully durable, survives restart/eviction); else
    consult the lean `("node_outcome", id)` record to split ran-then-
    reclaimed / failed / skipped / never-ran. The heavy `node_status`/logs
    are never read; `node_outcome` is read ONLY for file-missing nodes and
    memoized per request. Groups re-aggregate from their children. The
    internal RECLAIMED state is translated here to the frontend-accepted
    SUCCEEDED + `deleted` modifier (the DAG component throws on unknown
    state strings), so no internal state leaks to the client."""
    from dependency_graph.visual_graph import (
        _RECLAIMED, aggregate_states, derive_display_state,
    )
    state = snapshot.get("state", {})
    outcome_cache: dict = {}

    def outcome(nid):
        cid = nid.partition("@")[0]
        if cid not in outcome_cache:
            outcome_cache[cid] = context.get(("node_outcome", cid))
        return outcome_cache[cid]

    def walk(nid, meta):
        gm = meta.get("groupMeta")
        if gm:
            meta["state"] = aggregate_states([walk(c, m) for c, m in gm.items()])
            return meta["state"]
        path = (meta.get("options") or {}).get("expectedPath")
        has_output = path is not None
        present = has_output and os.path.exists(path)
        oc = None if present else outcome(nid)
        internal = derive_display_state(
            has_output, present, oc, meta.get("state", "NO_STATE_RUNTIME"),
        )
        if internal == _RECLAIMED:
            meta["state"] = "SUCCEEDED"
            meta["modifiers"] = ["deleted"]
            meta["callout"] = "reclaimed"
            meta["stateTooltip"] = (
                "ran successfully — output reclaimed by ephemeral cleanup"
            )
        else:
            meta["state"] = internal
            # Clear any stale reclaimed decoration — defensive against a
            # snapshot dict being reused across serves (e.g. the file
            # reappeared on a re-run since the last render). These keys are
            # only ever set here, so dropping them is safe.
            for k in ("modifiers", "callout", "stateTooltip"):
                meta.pop(k, None)
        return internal

    for nid, meta in state.items():
        walk(nid, meta)
    return snapshot


def _slice_workflow_graph_by_filter(snapshot: dict, filter: str) -> dict:
    """Prune the snapshot's `state` and `skeleton` to entries whose
    `filters` list contains `filter` (or is empty — empty means
    "shared / global", visible in every slice).

    Group entries' `state` is RE-AGGREGATED from their surviving
    children, so a group that contained a FAILED descendant only
    reachable through a hidden filter shows the state of what's
    still visible (instead of the pre-computed cross-filter
    aggregate the snapshot baked in)."""
    state = snapshot.get("state", {})

    def visible(meta):
        filters = (meta or {}).get("filters", [])
        return not filters or filter in filters

    # Group re-aggregation after a filter slice — shared with the build side
    # and the durable overlay via `aggregate_states` (the overlay already ran,
    # so leaf states here are already frontend-valid; RECLAIMED never appears).
    from dependency_graph.visual_graph import aggregate_states as aggregate

    def slice_meta(m):
        """Return a copy of `m` with its `groupMeta` filtered to
        visible children, and its `state` recomputed from those
        surviving children's states. Returns None if `m` itself
        isn't visible."""
        if not visible(m):
            return None
        out = dict(m)
        gm = m.get("groupMeta")
        if gm:
            new_gm = {}
            for k, child in gm.items():
                sliced = slice_meta(child)
                if sliced is not None:
                    new_gm[k] = sliced
            out["groupMeta"] = new_gm
            if new_gm:
                out["state"] = aggregate([c.get("state") for c in new_gm.values()])
        return out

    new_state: dict[str, dict] = {}
    for nid, m in state.items():
        sliced = slice_meta(m)
        if sliced is not None:
            new_state[nid] = sliced

    def visible_id(node_id):
        return visible(state.get(node_id, {}))

    # A shared group (fan-in from >1 parent — e.g. `color_post_process`,
    # consumed by every filter's merge) has its `definition` baked onto only the
    # FIRST skeleton reference; later references are bare `{id, type:"group"}`
    # stubs (build_skeleton_from_visual's dedup). If the slice prunes the one
    # definition-bearing copy — its parent lives under a hidden filter — the
    # surviving stub renders as an un-expandable group. Index every group's full
    # definition so `prune` can graft it onto whichever copy survives, then
    # slice that too (mirrors what `slice_meta` already does for `groupMeta`).
    skeleton = snapshot.get("skeleton", [])
    full_defs: dict[str, list] = {}

    def _collect_defs(entries):
        for e in entries:
            if not isinstance(e, dict):
                continue
            eid = e.get("id")
            if eid is not None and isinstance(e.get("definition"), list):
                full_defs.setdefault(eid, e["definition"])
            for key in ("next", "definition"):
                if isinstance(e.get(key), list):
                    _collect_defs(e[key])

    _collect_defs(skeleton)

    def prune(entries):
        out = []
        for e in entries:
            nid = e.get("id")
            if nid is not None and not visible_id(nid):
                continue
            new_e = dict(e)
            if "next" in new_e:
                new_e["next"] = prune(new_e["next"])
            # Reconstruct the definition for any surviving group: a bare stub
            # grafts the id's full definition; every copy is then sliced to its
            # visible children (so `/H` expands to H's ops + the shared
            # deliverables, not all six filters' worth).
            defn = new_e.get("definition")
            if defn is None and new_e.get("type") == "group" and nid in full_defs:
                defn = full_defs[nid]
            if isinstance(defn, list):
                new_e["definition"] = prune(defn)
            out.append(new_e)
        return out

    return {
        "skeleton": prune(skeleton),
        "state": new_state,
    }


def _strip_server_internal_fields(snapshot: dict) -> dict:
    """Drop server-only fields from every meta entry before returning to
    the frontend — the DagNode constructor rejects unknown top-level keys.
    Removes `filters` (used by `_slice_workflow_graph_by_filter`) and the
    baked `options.expectedPath` (used by `_overlay_durable_state`; an
    absolute on-disk path that must not leak). Recurses into `groupMeta`,
    which nests meta dicts for synthetic-group / subgraph entries."""
    def strip(meta):
        out = {k: v for k, v in meta.items() if k != "filters"}
        opts = out.get("options")
        if isinstance(opts, dict) and "expectedPath" in opts:
            out["options"] = {k: v for k, v in opts.items() if k != "expectedPath"}
        if "groupMeta" in out:
            out["groupMeta"] = {
                nid: strip(m) for nid, m in out["groupMeta"].items()
            }
        return out

    state = {nid: strip(m) for nid, m in snapshot.get("state", {}).items()}
    return {"skeleton": snapshot.get("skeleton", []), "state": state}


def _resolve_ref(ref):
    """Pull a workflow_graphs ContextRef's payload. Returns None
    if the ref itself is None or its underlying entry has been
    evicted from the cache — the index dict can outlive a wipe
    of the per-graph data."""
    if ref is None:
        return None
    try:
        return ref.get()
    except Exception:
        return None


def _json_default(obj):
    try:
        return str(obj)
    except Exception:
        return repr(obj)


@compute_graphs_bp.route("/api/node_status/<node_id>")
def node_status(node_id):
    # Skeleton ids are mangled `<node_to_id>@<scope_suffix>` so the
    # same underlying Node can render in multiple consumer groups
    # (visual duplication). Status entries are keyed by the
    # canonical `node_to_id` alone — one Node has one status — so
    # strip the scope suffix before lookup.
    node_id = node_id.partition("@")[0]
    return json.dumps(context[("node_status", node_id)], default=_json_default)


@compute_graphs_bp.route("/api/open_node_result_path/<node_id>")
def open_node_result_path(node_id):
    node_id = node_id.partition("@")[0]
    status = context[("node_status", node_id)]
    if "path" in status:
        from file_io.planar import reconstruct_planar_tiff
        filename = str(reconstruct_planar_tiff(status["path"]))
        print(f"### opening filename {filename}")
        subprocess.Popen(
            [
                pixinsight("PixInsight.app", "Contents", "MacOS", "PixInsight"),
                filename,
            ]
        )

        return Response(status=200)
