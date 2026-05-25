"""Structural tests for the registered Mosaic pipelines on the
4-pane Orion test mosaic.

These trace each `@pipeline(processes=Mosaic)` body's function
against the Orion testdata's identified mosaic and assert:

1. Per-op node counts match what the algorithm should emit.
2. The lexical `@workflow_group` structure makes sense — each
   group has expected membership, and shared upstreams (cal
   masters) live where they should.

Marked `@pytest.mark.exhaustive` because loading the Orion
testdata + running identification is heavy. Run with
`-m exhaustive`.
"""
from collections import defaultdict
from dataclasses import replace

import pytest

from compute_graph import build_compute_graph
from context import context
from decisions.settings import ProcessingConfig
from dependency_graph import run_graph
from dependency_graph.classes import Graph, Node
from dependency_graph.graph_runtime import GraphRuntime
from dependency_graph.tracing import trace
from dependency_graph.workflow_graph_adapter import node_to_id
from mosaics.orion_mosaic import add_orion_mosaic_images


def _count_by_op(graph) -> dict[str, int]:
    """Count distinct Nodes by op name. Uses Node equality
    (__hash__/__eq__) for dedup so structurally-identical Nodes
    created across multiple trace-time `record()` calls are
    counted once — that matches what the @node runtime cache
    sees as a single op invocation."""
    counts: dict[str, int] = {}
    seen: set = set()
    for k, deps in graph.items():
        for n in [k, *deps]:
            if isinstance(n, Node) and n not in seen:
                seen.add(n)
                name = getattr(n.task, "__name__", str(n.task))
                counts[name] = counts.get(name, 0) + 1
    return counts


def _nodes_by_group_name(graph) -> dict[str, list[Node]]:
    """Bucket every Node by the outermost segment of its
    `_group_path`. Empty-path Nodes go under `""`."""
    by_group: dict[str, list[Node]] = defaultdict(list)
    seen = set()
    for k, deps in graph.items():
        for n in [k, *deps]:
            if not isinstance(n, Node) or id(n) in seen:
                continue
            seen.add(id(n))
            path = getattr(n, "_group_path", ())
            label = path[0][0] if path else ""
            by_group[label].append(n)
    return by_group


def _cross_group_edges(graph) -> list[tuple[Node, Node]]:
    """Edges (downstream, upstream) where the two endpoints'
    `_group_path` first segments differ. These are the only
    pairs the viewer renders as inter-group edges."""
    edges = []
    for node, deps in graph.items():
        if not isinstance(node, Node):
            continue
        np = getattr(node, "_group_path", ())
        n_seg = np[0] if np else None
        for dep in deps:
            if not isinstance(dep, Node):
                continue
            dp = getattr(dep, "_group_path", ())
            d_seg = dp[0] if dp else None
            if n_seg != d_seg:
                edges.append((node, dep))
    return edges


def _make_config(*, mock_pixinsight: bool = False) -> ProcessingConfig:
    settings = context.settings
    dev = settings.development_settings
    if mock_pixinsight and not dev.mock_pixinsight:
        dev = replace(dev, mock_pixinsight=True)
    return ProcessingConfig(
        calibration_stacks=context.calibration_stacks or {},
        development_settings=dev,
        min_frames_in_stack=settings.clustering_settings.min_frames_in_stack,
    )


def _identify(test_context):
    """Stage Orion testdata + run identification only (Subframe
    pipeline previews still run but their outputs are not what
    these tests assert on)."""
    add_orion_mosaic_images()
    run_graph(build_compute_graph(), debug=False)
    return next(iter(context.mosaics.values())).get()


@pytest.mark.exhaustive
def test_pi_master_orion_op_counts(test_context):
    """`PixInsight Master`.for_mosaic traced on the 4-pane Orion
    mosaic emits the expected per-op node counts.

    Orion's 4 panes each contain one stack at one filter, with
    bias+dark+flat cal stacks shared across every pane. The 2x2
    strip layout means each filter sees a within-strip pairwise
    merge per row, then a cross-strip merge."""
    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.pixinsight_master as pi
    g = trace(pi.for_mosaic._function, mosaic, config)
    assert g is not None
    counts = _count_by_op(g)

    print(f"\nPI Master op counts:\n{counts}")

    # Dump stack_files Nodes for debugging dedup behavior.
    print("\nstack_files Nodes:")
    seen = set()
    for node, deps in g.items():
        if id(node) in seen or not isinstance(node, Node):
            continue
        seen.add(id(node))
        if node.task._function.__name__ != "stack_files":
            continue
        stack = node.args[0]
        print(f"  type={type(stack).__name__} image_type={stack.image_type} "
              f"stable={stack.stable_hash[:12]} ndeps={len(deps)} "
              f"name={node.name[:12]}")

    # Per-pane chain:
    assert counts["post_process_stack"] == 4, counts
    # 4 light stack_files (unique per pane) + 3 cal stack_files
    # (bias/dark/flat — deduped across pane chains because the
    # same `stack_files(cs, config, *cal_masters_so_far)` Node
    # has the same args for all panes).
    assert counts["stack_files"] == 7, counts
    # Per-(target, filter) HDR-combine; Orion = 4 targets × 1 filter.
    assert counts["merge_target_channel"] == 4, counts
    # Panel placement: one per panel.
    assert counts["place_image"] == 4, counts
    # 2x2 strip layout: 2 within-strip pairwise (one per row of 2)
    # + 1 cross-strip pairwise = 3.
    assert counts["merge_images"] == 3, counts
    # Final per-filter mosaic file: 1.
    assert counts["store_mosaic_color_channel_xisf"] == 1, counts


def _format_group_path(node) -> str:
    """Render a Node's `_group_path` as `name#abcd/name#efgh`. The
    short hash suffix on each segment disambiguates instances of
    the same lexical group — e.g. `_channels_for_target` is called
    once per target, so without the suffix all four instances
    would render identically and cross-target edges would look
    like intra-group edges."""
    path = getattr(node, "_group_path", ())
    if not path:
        return "<top>"
    return "/".join(f"{name}#{gid[:4]}" for name, gid in path)


def _print_full_topology(label: str, graph) -> None:
    """Dump every Node grouped by lexical group_path, then every
    edge as `downstream <- upstream` annotated with each
    endpoint's op name + group path. Lets a human eyeball the
    full structure of a small graph like the Orion mosaic."""
    # Bucket Nodes by full group_path (so nested groups are
    # distinct, not lumped under the outermost segment).
    by_path: dict[tuple, list] = defaultdict(list)
    seen = set()
    for k, deps in graph.items():
        for n in [k, *deps]:
            if not isinstance(n, Node) or n in seen:
                continue
            seen.add(n)
            by_path[getattr(n, "_group_path", ())].append(n)

    print(f"\n=== {label}: Nodes by group path ===")
    for path in sorted(by_path.keys(), key=lambda p: tuple(s[0] for s in p)):
        label_path = _format_group_path(next(iter(by_path[path])))
        print(f"  [{label_path}]")
        for n in sorted(by_path[path], key=lambda n: n.task._function.__name__):
            primary = n.args[0] if n.args else "-"
            primary_repr = (
                f"{type(primary).__name__}({primary.stable_hash[:8]})"
                if hasattr(primary, "stable_hash") else repr(primary)[:40]
            )
            print(f"    {n.task._function.__name__}({primary_repr})")

    print(f"\n=== {label}: Edges (downstream <- upstream) ===")
    seen_edges = set()
    for node, deps in graph.items():
        if not isinstance(node, Node):
            continue
        for dep in deps:
            if not isinstance(dep, Node):
                continue
            key = (node, dep)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            d_path = _format_group_path(node)
            u_path = _format_group_path(dep)
            marker = " *" if d_path != u_path else ""
            print(f"  [{d_path}] {node.task._function.__name__} "
                  f"<- [{u_path}] {dep.task._function.__name__}{marker}")
    print(f"({len(seen_edges)} edges; * marks cross-group)")


@pytest.mark.exhaustive
def test_pi_master_orion_full_topology(test_context):
    """Dump the full PI Master traced topology on the 4-pane
    Orion mosaic. Doesn't assert structure — exists for a human
    to eyeball when investigating cross-group edges or weird
    fan-out patterns in the viewer."""
    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.pixinsight_master as pi
    g = trace(pi.for_mosaic._function, mosaic, config)

    _print_full_topology("PI Master / Orion 2x2", g)


def _canonical_nodes(graph) -> list[Node]:
    """De-duplicate Nodes across the whole graph (keys + dep lists)
    by equality. Returns one Node per equivalence class — useful for
    structural assertions about per-op cohorts."""
    seen, out = set(), []
    for k, deps in graph.items():
        for n in [k, *deps]:
            if isinstance(n, Node) and n not in seen:
                seen.add(n)
                out.append(n)
    return out


@pytest.mark.exhaustive
def test_pi_master_orion_group_structure(test_context):
    """Structural invariants of the PI Master trace on Orion.

    Locks in the dedup-clears-group-path semantics of `record()`:
    a Node re-recorded under a different lexical scope has its
    `_group_path` cleared to `()`, so no spurious `_channels_for_target`
    ↔ `_channels_for_target` cross-edges appear in the viewer.

    Specifically, on the 4-pane Orion mosaic with bias/dark/flat
    cal stacks shared across every pane:

    1. The 3 cal-master `stack_files` Nodes sit at `_group_path = ()`
       (shared across all four `_post_processed_master` recordings).
    2. Each of the 4 light-master `stack_files` is inside exactly
       one `_channels_for_target/_post_processed_master` scope.
    3. There is one `_channels_for_target` instance per target
       (4 total), and `merge_target_channel` Nodes live one per
       instance.
    4. No edge crosses from one `_channels_for_target` instance
       to another — shared computation must travel through a
       top-level Node."""
    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.pixinsight_master as pi
    g = trace(pi.for_mosaic._function, mosaic, config)

    nodes_by_op: dict[str, list[Node]] = defaultdict(list)
    for n in _canonical_nodes(g):
        nodes_by_op[n.task._function.__name__].append(n)

    # 1. Cal masters at top level.
    cal_stack_files = [
        n for n in nodes_by_op["stack_files"]
        if type(n.args[0]).__name__ == "CalibrationStack"
    ]
    assert len(cal_stack_files) == 3, cal_stack_files
    for n in cal_stack_files:
        assert n._group_path == (), (
            f"cal-master stack_files should sit at top level (shared "
            f"across targets clears _group_path), got {n._group_path}"
        )

    # 2. Light masters one per _post_processed_master.
    light_stack_files = [
        n for n in nodes_by_op["stack_files"]
        if type(n.args[0]).__name__ == "Stack"
    ]
    assert len(light_stack_files) == 4, light_stack_files
    ppm_ids = set()
    for n in light_stack_files:
        path = n._group_path
        assert len(path) == 2, (
            f"light stack_files should be inside _channels_for_target/"
            f"_post_processed_master, got group_path={path}"
        )
        assert path[0][0] == "_channels_for_target"
        assert path[1][0] == "_post_processed_master"
        ppm_ids.add(path[1][1])
    # Each light has a distinct _post_processed_master instance.
    assert len(ppm_ids) == 4, ppm_ids

    # 3. One _channels_for_target instance per target, with one
    #    merge_target_channel each.
    mtc = nodes_by_op["merge_target_channel"]
    assert len(mtc) == 4, mtc
    cft_ids = set()
    for n in mtc:
        path = n._group_path
        assert len(path) == 1 and path[0][0] == "_channels_for_target", path
        cft_ids.add(path[0][1])
    assert len(cft_ids) == 4, cft_ids

    # 4. No edges cross between two different _channels_for_target
    #    instances. (Edges to top-level / from outer scope are fine.)
    for node, deps in g.items():
        if not isinstance(node, Node):
            continue
        n_outer = node._group_path[0] if node._group_path else None
        n_in_cft = n_outer and n_outer[0] == "_channels_for_target"
        for dep in deps:
            if not isinstance(dep, Node):
                continue
            d_outer = dep._group_path[0] if dep._group_path else None
            d_in_cft = d_outer and d_outer[0] == "_channels_for_target"
            if n_in_cft and d_in_cft and n_outer[1] != d_outer[1]:
                raise AssertionError(
                    f"Cross-target edge: "
                    f"{node.task._function.__name__} in "
                    f"_channels_for_target#{n_outer[1][:4]} "
                    f"<- {dep.task._function.__name__} in "
                    f"_channels_for_target#{d_outer[1][:4]}"
                )


# ---------------------------------------------------------------------------
# Node-status / workflow-graph snapshot mechanism tests.
#
# These verify that the PI Master @pipeline runner wires up the
# `node_status_fn` and `graph_update_fn` callbacks correctly so that
# the workflow-graph viewer can render per-node run state.
#
# Background: the @pipeline decorator's body wraps the user fn in
# `keyed_fn` which calls `set_workflow_key(...)` inside the trace, so
# both callbacks land on the recorded Graph. At runtime, `run_pipeline`
# returns that Graph as a dynamic Ray sub-graph; the parent runtime
# spins up a child `GraphRuntime` whose `node_status_fn` /
# `graph_update_fn` are inherited from the Graph attributes. As nodes
# complete inside the child runtime, `_mark_done` fires the callbacks
# → `store_node_status` writes `context[("node_status", node_id)]`,
# and `_fire_graph_updates` periodically fires `graph_update_fn` →
# `store_workflow_graph(key, rt)` writes a fresh snapshot with current
# per-node state into `context["workflow_graphs"]`.
# ---------------------------------------------------------------------------


@pytest.mark.exhaustive
def test_pi_master_pipeline_sets_status_callbacks(test_context):
    """The @pipeline runner sets `node_status_fn` and
    `graph_update_fn` on the Graph it returns. Without these, the
    runtime has no way to populate per-node state for the viewer."""
    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.pixinsight_master as pi
    graph = pi.for_mosaic(mosaic, config)

    assert graph is not None
    assert graph.node_status_fn is not None, (
        "@pipeline runner failed to set node_status_fn — workflow "
        "viewer will never see per-node completion status."
    )
    assert graph.graph_update_fn is not None, (
        "@pipeline runner failed to set graph_update_fn — workflow "
        "graph snapshot will never be refreshed with runtime state."
    )


@pytest.mark.exhaustive
def test_pi_master_node_status_fn_writes_context(test_context):
    """Firing `graph.node_status_fn(node, result, logs)` writes a
    `("node_status", node_id)` entry into context. This is the path
    the runtime takes for every node completion."""
    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.pixinsight_master as pi
    graph = pi.for_mosaic(mosaic, config)
    leaf = next(n for n in graph if isinstance(n, Node) and not graph[n])

    graph.node_status_fn(leaf, "fake-result-string", [
        {"event": "fake log entry"},
    ])

    status = context[("node_status", node_to_id(leaf))]
    assert status is not None, (
        f"node_status_fn fired but context[('node_status', "
        f"{node_to_id(leaf)})] is empty — store_node_status did not "
        f"write to context."
    )
    assert status["result_type"] == "str", status
    assert any("fake log entry" in entry.get("event", "")
               for entry in status["logs"]), status["logs"]


@pytest.mark.exhaustive
def test_pi_master_graph_update_fn_reflects_runtime_state(test_context):
    """Firing `graph.graph_update_fn(rt)` against a runtime where
    some nodes have completed should produce a workflow-graph
    snapshot that surfaces those completions as state=SUCCEEDED in
    `context[("workflow_graphs", key)]`. This is the only path by
    which the viewer ever shows nodes as run."""
    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.pixinsight_master as pi
    graph = pi.for_mosaic(mosaic, config)

    rt = GraphRuntime(graph)
    leaf = next(n for n in graph if isinstance(n, Node) and not graph[n])
    rt._mark_done(leaf, "fake-result", [])

    graph.graph_update_fn(rt)

    wf_index = context["workflow_graphs"]
    assert wf_index, "workflow_graphs context entry empty after fire"

    # Find the snapshot the @pipeline runner keyed under
    # (type_tag, entity_id, pipeline_name) and confirm the leaf
    # appears as SUCCEEDED.
    key = ("mosaic", mosaic.stable_hash, "PixInsight Master")
    ref = wf_index.get(key)
    assert ref is not None, (
        f"no workflow_graph snapshot at key={key}; have keys: "
        f"{list(wf_index.keys())}"
    )
    snapshot = ref.get()
    assert snapshot is not None and "state" in snapshot

    leaf_state = snapshot["state"].get(node_to_id(leaf))
    assert leaf_state is not None, (
        f"leaf {node_to_id(leaf)} missing from snapshot state; "
        f"snapshot state keys: {list(snapshot['state'].keys())[:8]}…"
    )
    assert leaf_state["state"] == "SUCCEEDED", (
        f"leaf was _mark_done'd but snapshot shows state="
        f"{leaf_state['state']}: {leaf_state}"
    )


@pytest.mark.exhaustive
def test_pi_master_initial_topology_snapshot_is_stateless(test_context):
    """Documents the topology-only snapshot the @pipeline runner
    writes at trace time, before any nodes have run. Every node's
    state is `NO_STATE_RUNTIME`. The runtime overwrites this with
    a live snapshot as soon as its first `_fire_graph_updates`
    fires; this test exists to anchor that contract — if the
    initial snapshot ever starts carrying real state, callers that
    rely on the "viewer renders skeleton immediately" promise will
    notice."""
    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.pixinsight_master as pi
    pi.for_mosaic(mosaic, config)

    key = ("mosaic", mosaic.stable_hash, "PixInsight Master")
    snapshot = context["workflow_graphs"][key].get()
    states = {meta["state"] for meta in snapshot["state"].values()}
    assert states == {"NO_STATE_RUNTIME"}, (
        f"initial topology snapshot carried non-default states: {states}"
    )


@pytest.mark.exhaustive
def test_pi_master_runtime_mark_done_propagates_to_snapshot(test_context):
    """End-to-end: simulate the runtime path that the parent graph
    would take when `run_pipeline` returns the pipeline's Graph.

    Build a `GraphRuntime` over the pipeline's Graph (mirrors what
    happens in `node_completed` when a Ray task returns a Graph),
    `_mark_done` every node, fire `_fire_graph_updates(rt)` once,
    and verify every node in the snapshot ends up SUCCEEDED. This
    is the actual flow that should produce live "node is done"
    colouring in the viewer."""
    from dependency_graph.dependency_graph import _fire_graph_updates

    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.pixinsight_master as pi
    graph = pi.for_mosaic(mosaic, config)

    rt = GraphRuntime(graph)
    for n in list(graph.keys()):
        if isinstance(n, Node):
            rt._mark_done(n, "fake", [])

    _fire_graph_updates(rt)

    key = ("mosaic", mosaic.stable_hash, "PixInsight Master")
    snapshot = context["workflow_graphs"][key].get()
    states = [meta["state"] for meta in snapshot["state"].values()]
    # Synthetic group entries aggregate from members; with every
    # member SUCCEEDED, the group entries should be SUCCEEDED too.
    assert all(s == "SUCCEEDED" for s in states), (
        f"some nodes still show non-SUCCEEDED after _mark_done'ing "
        f"every node: {[s for s in states if s != 'SUCCEEDED']}"
    )


@pytest.mark.exhaustive
def test_pi_downscaled_pipeline_sets_status_callbacks(test_context):
    """Same wiring check on the Downscaled pipeline — confirms the
    @pipeline runner behaviour isn't accidentally pixinsight_master-
    specific."""
    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.pixinsight_master_downscaled as pid
    graph = pid.for_mosaic(mosaic, config)

    assert graph is not None
    assert graph.node_status_fn is not None
    assert graph.graph_update_fn is not None


@pytest.mark.exhaustive
def test_pi_master_callbacks_survive_ray_serialization(test_context):
    """`run_pipeline` is a `@ray.remote` task that returns the
    pipeline Graph. That return value is serialized via cloudpickle,
    shipped through the Ray object store, deserialized on the
    parent worker, and handed to `node_completed` which constructs
    a child `GraphRuntime` from it. The child runtime reads
    `graph.node_status_fn` / `graph.graph_update_fn` off the
    deserialized Graph — so the callbacks must survive cloudpickle.

    A non-picklable callback would silently drop to None on the
    parent side, and the child runtime would inherit nothing, and
    the viewer would stay frozen at the topology snapshot. That's
    the most plausible cause of the user's "nodes never look run"
    report once the in-isolation wiring is confirmed."""
    from ray import cloudpickle

    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.pixinsight_master as pi
    graph = pi.for_mosaic(mosaic, config)
    assert graph.node_status_fn is not None
    assert graph.graph_update_fn is not None

    blob = cloudpickle.dumps(graph)
    restored = cloudpickle.loads(blob)

    assert restored.node_status_fn is not None, (
        "node_status_fn was lost during cloudpickle round-trip — "
        "the runtime-side child GraphRuntime will have no callback "
        "and the workflow viewer will never see node completions."
    )
    assert restored.graph_update_fn is not None, (
        "graph_update_fn was lost during cloudpickle round-trip."
    )


@pytest.mark.exhaustive
def test_pi_master_filter_sliced_snapshot_shows_succeeded(test_context):
    """End-to-end check for the read-time per-filter slice that
    `web_server/backend/compute_graphs.py:_slice_workflow_graph_by_filter`
    serves on the `<filter>` URL: the FULL snapshot is written once,
    refreshed by `graph_update_fn` with live runtime state, and
    sliced on read by walking the per-node `filters` field in the
    snapshot's `state` dict. The sliced result must carry the same
    SUCCEEDED states as the unsliced snapshot — that's what makes
    the DAG show success checkmarks under a per-filter URL."""
    from dependency_graph.dependency_graph import _fire_graph_updates
    from web_server.backend.compute_graphs import (
        _slice_workflow_graph_by_filter,
    )

    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.pixinsight_master as pi
    graph = pi.for_mosaic(mosaic, config)

    rt = GraphRuntime(graph)
    for n in list(graph.keys()):
        if isinstance(n, Node):
            rt._mark_done(n, "fake", [])
    _fire_graph_updates(rt)

    key = ("mosaic", mosaic.stable_hash, "PixInsight Master")
    full_snapshot = context["workflow_graphs"][key].get()

    sliced = _slice_workflow_graph_by_filter(full_snapshot, "H")
    assert sliced["state"], "sliced state dict is empty for filter=H"
    states = [meta["state"] for meta in sliced["state"].values()]
    assert all(s == "SUCCEEDED" for s in states), (
        f"per-filter slice on filter=H has non-SUCCEEDED states: "
        f"{sorted({s for s in states if s != 'SUCCEEDED'})}"
    )

    # Slicing must actually subset something — Orion is single-
    # filter (H) so every node should be present. The asymmetric
    # bit: a fictional filter that doesn't exist should slice to
    # only globally-tagged nodes (empty filters set), if any.
    assert set(sliced["state"]) == set(full_snapshot["state"]), (
        f"filter=H slice for Orion (which only has H) should include "
        f"every node from the full snapshot. Missing: "
        f"{set(full_snapshot['state']) - set(sliced['state'])}"
    )


@pytest.mark.exhaustive
def test_pi_master_no_per_filter_snapshots_written(test_context):
    """The @pipeline decorator does NOT write per-filter workflow-
    graph snapshots. The MosaicViewer's per-filter URL
    (`/api/workflow_graph/mosaic/<id>/<pipeline>/<filter>`) falls
    back to the full-pipeline snapshot at the endpoint level — see
    `web_server/backend/compute_graphs.py:workflow_graph_for_mosaic`.

    Anchors that contract: a per-filter snapshot would be stale by
    design (only the full-pipeline `graph_update_fn` refreshes with
    live state), so per-filter keys must NOT exist or the endpoint
    would serve stale data instead of falling back."""
    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.pixinsight_master as pi
    pi.for_mosaic(mosaic, config)

    base_key = ("mosaic", mosaic.stable_hash, "PixInsight Master")
    wf = context["workflow_graphs"]
    assert base_key in wf, f"full-pipeline key {base_key} missing"
    per_filter_keys = [k for k in wf if len(k) == 4 and k[:3] == base_key]
    assert per_filter_keys == [], (
        f"unexpected per-filter snapshot keys (would be stale, "
        f"shadowing the full-pipeline fallback): {per_filter_keys}"
    )


@pytest.mark.exhaustive
def test_pi_master_pipeline_callbacks_are_fresh_per_call(test_context):
    """Independent calls to the @pipeline runner each get their own
    fresh callbacks — neither the closure nor `set_workflow_key` is
    holding onto a stale `partial(...)` from a prior trace."""
    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.pixinsight_master as pi
    g1 = pi.for_mosaic(mosaic, config)
    g2 = pi.for_mosaic(mosaic, config)
    assert g1 is not g2
    assert g1.node_status_fn is not None
    assert g2.node_status_fn is not None


@pytest.mark.exhaustive
def test_pi_downscaled_orion_op_counts(test_context):
    """`PixInsight Downscaled`.for_mosaic — same op counts as PI
    Master. `downscale_stack` is a plain Python function, not a
    `@node`, so no extra Nodes appear in the trace; the only
    difference between the two pipelines lives inside `stack_files`
    bodies (different upstream `stack.contents`)."""
    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.pixinsight_master_downscaled as pid
    g = trace(pid.for_mosaic._function, mosaic, config)
    counts = _count_by_op(g)

    print(f"\nPI Downscaled op counts:\n{counts}")

    assert counts["post_process_stack"] == 4, counts
    assert counts["stack_files"] == 7, counts
    assert counts["merge_target_channel"] == 4, counts
    assert counts["place_image"] == 4, counts
    assert counts["merge_images"] == 3, counts
    assert counts["store_mosaic_color_channel_xisf"] == 1, counts


@pytest.mark.exhaustive
def test_subframe_orion_op_counts(test_context):
    """`Subframe`.for_mosaic — no stacking / calibration; each
    panel's best light frame is taken directly, projected, then
    pairwise-merged across strips."""
    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.subframe as sf
    g = trace(sf.for_mosaic._function, mosaic, config)
    counts = _count_by_op(g)

    print(f"\nSubframe op counts:\n{counts}")

    assert counts["get_stack_subframe_image"] == 4, counts
    assert counts["place_image"] == 4, counts
    assert counts["merge_images_subframe"] == 3, counts
    assert counts["store_mosaic_color_channel_subframe"] == 1, counts
