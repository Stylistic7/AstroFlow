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

pytestmark = pytest.mark.testdata  # reads core/testdata

from compute_graph import build_compute_graph
from context import context
from decisions.settings import ProcessingConfig
from dependency_graph import run_graph
from dependency_graph.classes import Node
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
                fn = getattr(n.task, "_function", n.task)
                name = getattr(fn, "__name__", str(fn))
                counts[name] = counts.get(name, 0) + 1
    return counts


def _path_of(node) -> tuple:
    """A Node's lexical scope. `record()` sets `_group_path` to
    the active `@workflow_group` stack; on shared re-record the
    canonical's `_group_path` is cleared to `()`."""
    return getattr(node, "_group_path", ())


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


def _format_scope(scope: tuple) -> str:
    """Render one scope as `name#abcd/name#efgh`. The short hash
    suffix on each segment disambiguates instances of the same
    lexical group (e.g. `channels_for_target` is called once per
    target — without the suffix all four instances would render
    identically)."""
    if not scope:
        return "<top>"
    return "/".join(f"{name}#{gid[:4]}" for name, gid in scope)


def _print_full_topology(label: str, graph) -> None:
    """Dump every Node grouped by its lexical scope, then every
    underlying edge `downstream <- upstream` annotated with each
    endpoint's op name + scope. Lets a human eyeball the
    structure of a small traced graph (e.g. the Orion mosaic)
    before any rendering transforms are applied."""
    by_scope: dict[tuple, list] = defaultdict(list)
    seen = set()
    for k, deps in graph.items():
        for n in [k, *deps]:
            if not isinstance(n, Node) or n in seen:
                continue
            seen.add(n)
            by_scope[_path_of(n)].append(n)

    print(f"\n=== {label}: Nodes by scope ===")
    for scope in sorted(by_scope.keys(), key=lambda p: tuple(s[0] for s in p)):
        print(f"  [{_format_scope(scope)}]")
        for n in sorted(by_scope[scope], key=lambda n: n.task._function.__name__):
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
            d_scope = _format_scope(_path_of(node))
            u_scope = _format_scope(_path_of(dep))
            print(f"  [{d_scope}] {node.task._function.__name__} "
                  f"<- [{u_scope}] {dep.task._function.__name__}")
    print(f"({len(seen_edges)} edges)")


@pytest.mark.exhaustive
def test_pi_oss_orion_full_topology(test_context):
    """Dump the full PI Master + OSS Mosaic traced topology on the 4-pane
    Orion mosaic. Doesn't assert structure — exists for a human
    to eyeball when investigating cross-group edges or weird
    fan-out patterns in the viewer."""
    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.hybrid_pi_oss_mosaic as pi
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
def test_pi_oss_orion_traced_group_structure(test_context):
    """Structural invariants of the RAW traced graph (pre any
    `sanitize_for_display` transforms). With `record()` clearing
    `_group_path` to `()` on shared re-record:

    1. The 3 cal-master `stack_files` Nodes end up with
       `_group_path = ()` (top-level / no canonical home — they
       were re-recorded inside multiple `post_processed_master`
       calls). The render-time
       `duplicate_shared_into_consumer_scopes` transform decides
       where to actually display them.
    2. Each of the 4 light-master `stack_files` has scope
       `(channels_for_target, post_processed_master)` — recorded
       once, never re-recorded.
    3. One `channels_for_target` instance per target (4 total),
       with `merge_target_channel` at `(channels_for_target,)`."""
    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.hybrid_pi_oss_mosaic as pi
    g = trace(pi.for_mosaic._function, mosaic, config)

    nodes_by_op: dict[str, list[Node]] = defaultdict(list)
    for n in _canonical_nodes(g):
        nodes_by_op[n.task._function.__name__].append(n)

    # 1. Cal masters cleared to top-level scope.
    cal_stack_files = [
        n for n in nodes_by_op["stack_files"]
        if type(n.args[0]).__name__ == "CalibrationStack"
    ]
    assert len(cal_stack_files) == 3, cal_stack_files
    for n in cal_stack_files:
        assert _path_of(n) == (), (
            f"cal-master stack_files re-recorded under multiple "
            f"scopes should have _group_path cleared to (); got "
            f"{_path_of(n)}"
        )

    # 2. Light masters: one (cft, ppm) scope each.
    light_stack_files = [
        n for n in nodes_by_op["stack_files"]
        if type(n.args[0]).__name__ == "Stack"
    ]
    assert len(light_stack_files) == 4, light_stack_files
    ppm_ids = set()
    for n in light_stack_files:
        path = _path_of(n)
        assert len(path) == 2, path
        assert path[0][0] == "channels_for_target"
        assert path[1][0] == "post_processed_master"
        ppm_ids.add(path[1][1])
    assert len(ppm_ids) == 4, ppm_ids

    # 3. One `channels_for_target` instance per target, one
    #    merge_target_channel per instance.
    mtc = nodes_by_op["merge_target_channel"]
    assert len(mtc) == 4, mtc
    cft_ids = set()
    for n in mtc:
        path = _path_of(n)
        assert len(path) == 1 and path[0][0] == "channels_for_target", path
        cft_ids.add(path[0][1])
    assert len(cft_ids) == 4, cft_ids


@pytest.mark.exhaustive
def test_pi_oss_orion_sanitized_duplicates_cal_masters(test_context):
    """After `sanitize_for_display`, the VisualGraph has cal-master
    `stack_files` duplicated into each `post_processed_master`
    scope — that's the visual-duplication the user wants in the
    DAG viewer. Each canonical cal-master Node maps to 4
    VisualNodes (one per Orion target's stack)."""
    from dependency_graph.visual_graph import to_visual_graph
    from dependency_graph.graph_transforms import sanitize_for_display

    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.hybrid_pi_oss_mosaic as pi
    g = trace(pi.for_mosaic._function, mosaic, config)

    vg = sanitize_for_display(to_visual_graph(g))

    # Group VisualNodes by canonical for cal-master ops.
    by_canonical: dict[Node, list] = defaultdict(list)
    for vn in vg:
        if vn.op_name == "stack_files" and type(vn.canonical.args[0]).__name__ == "CalibrationStack":
            by_canonical[vn.canonical].append(vn)
    assert len(by_canonical) == 3, list(by_canonical.keys())
    for canonical, copies in by_canonical.items():
        scopes = {c.scope for c in copies}
        assert len(scopes) == 4, (
            f"cal-master {canonical.task._function.__name__} should be "
            f"duplicated into 4 post_processed_master scopes; got "
            f"{len(scopes)}: {scopes}"
        )
        for s in scopes:
            assert len(s) == 2
            assert s[0][0] == "channels_for_target"
            assert s[1][0] == "post_processed_master"


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
def test_pi_oss_pipeline_sets_status_callbacks(test_context):
    """The @pipeline runner sets `node_status_fn` and
    `graph_update_fn` on the Graph it returns. Without these, the
    runtime has no way to populate per-node state for the viewer."""
    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.hybrid_pi_oss_mosaic as pi
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
def test_pi_oss_node_status_fn_writes_context(test_context):
    """Firing `graph.node_status_fn(node, result, logs)` writes a
    `("node_status", node_id)` entry into context. This is the path
    the runtime takes for every node completion."""
    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.hybrid_pi_oss_mosaic as pi
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
def test_pi_oss_graph_update_fn_reflects_runtime_state(test_context):
    """Firing `graph.graph_update_fn(rt)` against a runtime where
    some nodes have completed should produce a workflow-graph
    snapshot that surfaces those completions as state=SUCCEEDED in
    `context[("workflow_graphs", key)]`. This is the only path by
    which the viewer ever shows nodes as run."""
    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.hybrid_pi_oss_mosaic as pi
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
    key = ("mosaic", mosaic.stable_hash, "PI Master + OSS Mosaic")
    ref = wf_index.get(key)
    assert ref is not None, (
        f"no workflow_graph snapshot at key={key}; have keys: "
        f"{list(wf_index.keys())}"
    )
    snapshot = ref.get()
    assert snapshot is not None and "state" in snapshot

    # State entries are keyed by `_entry_id(node, scope) =
    # <node_to_id>@<scope_suffix>`. Find the rendering(s) of the
    # leaf in the snapshot.
    leaf_renderings = [
        m for k, m in snapshot["state"].items()
        if k.partition("@")[0] == node_to_id(leaf)
    ]
    assert leaf_renderings, (
        f"leaf {node_to_id(leaf)} missing from snapshot state; "
        f"snapshot state keys: {list(snapshot['state'].keys())[:8]}…"
    )
    for r in leaf_renderings:
        assert r["state"] == "SUCCEEDED", (
            f"leaf was _mark_done'd but snapshot shows state="
            f"{r['state']}: {r}"
        )


@pytest.mark.exhaustive
def test_pi_oss_initial_topology_snapshot_is_stateless(test_context):
    """Documents the topology-only snapshot written by
    `_eager_trace_pipeline` at cart-dispatch time, before any nodes
    have run. Every node's state is `NO_STATE_RUNTIME`. The runtime
    overwrites this with a live snapshot as soon as its first
    `_fire_graph_updates` fires; this test anchors the contract —
    if the initial snapshot ever starts carrying real state, callers
    that rely on the "viewer renders skeleton immediately" promise
    will notice. The @pipeline runner itself NO LONGER snapshots:
    that lives at the execution-site `_eager_trace_pipeline` so
    dashboard tap-extraction traces stay cheap."""
    from compute_graph.compute_graph import _eager_trace_pipeline
    mosaic = _identify(test_context)
    config = _make_config()

    _eager_trace_pipeline("PI Master + OSS Mosaic", "mosaic", mosaic, config)

    key = ("mosaic", mosaic.stable_hash, "PI Master + OSS Mosaic")
    snapshot = context["workflow_graphs"][key].get()
    states = {meta["state"] for meta in snapshot["state"].values()}
    assert states == {"NO_STATE_RUNTIME"}, (
        f"initial topology snapshot carried non-default states: {states}"
    )


@pytest.mark.exhaustive
def test_pi_oss_runtime_mark_done_propagates_to_snapshot(test_context):
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

    import pipelines.hybrid_pi_oss_mosaic as pi
    graph = pi.for_mosaic(mosaic, config)

    rt = GraphRuntime(graph)
    for n in list(graph.keys()):
        if isinstance(n, Node):
            rt._mark_done(n, "fake", [])

    _fire_graph_updates(rt)

    key = ("mosaic", mosaic.stable_hash, "PI Master + OSS Mosaic")
    snapshot = context["workflow_graphs"][key].get()
    states = [meta["state"] for meta in snapshot["state"].values()]
    # Synthetic group entries aggregate from members; with every
    # member SUCCEEDED, the group entries should be SUCCEEDED too.
    assert all(s == "SUCCEEDED" for s in states), (
        f"some nodes still show non-SUCCEEDED after _mark_done'ing "
        f"every node: {[s for s in states if s != 'SUCCEEDED']}"
    )


@pytest.mark.exhaustive
def test_pi_oss_callbacks_survive_ray_serialization(test_context):
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

    import pipelines.hybrid_pi_oss_mosaic as pi
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
def test_pi_oss_filter_sliced_snapshot_shows_succeeded(test_context):
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

    import pipelines.hybrid_pi_oss_mosaic as pi
    graph = pi.for_mosaic(mosaic, config)

    rt = GraphRuntime(graph)
    for n in list(graph.keys()):
        if isinstance(n, Node):
            rt._mark_done(n, "fake", [])
    _fire_graph_updates(rt)

    key = ("mosaic", mosaic.stable_hash, "PI Master + OSS Mosaic")
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
def test_pi_oss_cal_masters_visually_duplicated_per_stack(test_context):
    """The viewer renders each cal-master `stack_files` Node ONCE
    per consuming `post_processed_master` scope — visual
    duplication eliminates the criss-cross fan-out from a single
    top-level cal-master to all four `channels_for_target`
    groups. Underlying execution is unchanged; one Node, multiple
    rendered entries."""
    mosaic = _identify(test_context)
    config = _make_config()

    # The @pipeline runner no longer snapshots on its own; the topology
    # snapshot is written at dispatch time by `_eager_trace_pipeline`.
    from compute_graph.compute_graph import _eager_trace_pipeline
    _eager_trace_pipeline("PI Master + OSS Mosaic", "mosaic", mosaic, config)

    key = ("mosaic", mosaic.stable_hash, "PI Master + OSS Mosaic")
    snapshot = context["workflow_graphs"][key].get()

    # Group state entries by the canonical node_to_id (the prefix
    # before `@`). Each cal-master Node should have 4 renderings.
    by_canonical: dict[str, list[str]] = defaultdict(list)
    for entry_id, meta in snapshot["state"].items():
        if meta.get("hasControlNode"):
            continue  # synthetic group entry, not a Node rendering
        canonical = entry_id.partition("@")[0]
        by_canonical[canonical].append(entry_id)

    cal_renderings = [
        eids for canonical, eids in by_canonical.items()
        if canonical.startswith("stack_files_") and len(eids) > 1
    ]
    assert len(cal_renderings) == 3, (
        f"expected 3 cal-master stack_files Nodes each rendered "
        f"multiply; got {len(cal_renderings)}. by_canonical: "
        f"{ {k: len(v) for k, v in by_canonical.items() if k.startswith('stack_files_')} }"
    )
    for eids in cal_renderings:
        assert len(eids) == 4, (
            f"each cal-master should render 4 times (one per stack); "
            f"got {len(eids)} entries: {eids}"
        )


@pytest.mark.exhaustive
def test_pi_oss_run_pipeline_done_hidden_from_skeleton(test_context):
    """`run_pipeline_done` is a sub-runtime terminator added by the
    `run_pipeline` Ray task — not part of the user's pipeline code.
    The `remove_transparent_nodes` transform strips it; the
    sanitized snapshot has no trace of it in skeleton or state."""
    from dependency_graph.workflow_graph_adapter import to_workflow_graph
    from compute_graph.compute_graph import run_pipeline_done

    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.hybrid_pi_oss_mosaic as pi
    pipeline_graph = pi.for_mosaic(mosaic, config)

    # Mimic what `run_pipeline` does: tack the terminator onto the
    # pipeline's Graph and wrap it in a runtime.
    depended_on = {d for ds in pipeline_graph.values() for d in ds}
    terminals = [n for n in pipeline_graph if n not in depended_on]
    pipeline_graph[run_pipeline_done] = terminals
    rt = GraphRuntime(pipeline_graph)

    wf = to_workflow_graph(rt)

    def walk(entries):
        for e in entries:
            yield e["id"]
            yield from walk(e.get("definition", []))
            yield from walk(e.get("next", []))

    for eid in walk(wf["skeleton"]):
        assert "run_pipeline_done" not in eid, (
            f"run_pipeline_done leaked into the rendered skeleton: {eid}"
        )
    for eid in wf["state"]:
        assert "run_pipeline_done" not in eid, (
            f"run_pipeline_done leaked into the rendered state: {eid}"
        )


@pytest.mark.exhaustive
def test_pi_oss_no_per_filter_snapshots_written(test_context):
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

    # The @pipeline runner no longer snapshots on its own; the topology
    # snapshot is written at dispatch time by `_eager_trace_pipeline`.
    from compute_graph.compute_graph import _eager_trace_pipeline
    _eager_trace_pipeline("PI Master + OSS Mosaic", "mosaic", mosaic, config)

    base_key = ("mosaic", mosaic.stable_hash, "PI Master + OSS Mosaic")
    wf = context["workflow_graphs"]
    assert base_key in wf, f"full-pipeline key {base_key} missing"
    per_filter_keys = [k for k in wf if len(k) == 4 and k[:3] == base_key]
    assert per_filter_keys == [], (
        f"unexpected per-filter snapshot keys (would be stale, "
        f"shadowing the full-pipeline fallback): {per_filter_keys}"
    )


@pytest.mark.exhaustive
def test_pi_oss_pipeline_callbacks_are_fresh_per_call(test_context):
    """Independent calls to the @pipeline runner each get their own
    fresh callbacks — neither the closure nor `set_workflow_key` is
    holding onto a stale `partial(...)` from a prior trace."""
    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.hybrid_pi_oss_mosaic as pi
    g1 = pi.for_mosaic(mosaic, config)
    g2 = pi.for_mosaic(mosaic, config)
    assert g1 is not g2
    assert g1.node_status_fn is not None
    assert g2.node_status_fn is not None


@pytest.mark.exhaustive
def test_subframe_orion_op_counts(test_context):
    """`Subframe`.for_mosaic — no stacking / calibration; each
    panel's best light frame is taken directly, then projected and
    blended in one variadic `_blend_many` pass per filter inside a
    single dynamic scope."""
    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.subframe as sf
    g = trace(sf.for_mosaic._function, mosaic, config)
    counts = _count_by_op(g)

    print(f"\nSubframe op counts:\n{counts}")

    # One representative frame per pane, re-identified under its op; the
    # `{4 x place_image -> merge_images_subframe}` merge is ONE scope node
    # (`subframe_scope`, presented as its terminal), then the stored channel.
    assert counts["subframe_channel"] == 4, counts
    assert counts["compute_viewbox_measured"] == 1, counts
    assert counts["subframe_scope"] == 1, counts
    assert counts["store_mosaic_color_channel_subframe"] == 1, counts


def _ancestor_ops(graph, node) -> set[str]:
    """Op names of everything `node` (transitively) depends on."""
    seen, stack, ops = set(), [node], set()
    while stack:
        n = stack.pop()
        for d in graph.get(n, ()):
            if isinstance(d, Node) and d not in seen:
                seen.add(d)
                stack.append(d)
                fn = getattr(d.task, "_function", d.task)
                ops.add(getattr(fn, "__name__", str(fn)))
    return ops


# Identifies the real Orion frames, which now includes measuring each Run's
# dust (~80 s alone; past 120 s beside five sibling shards).
@pytest.mark.timeout(300)
def test_siril_canvas_never_waits_on_calibration(test_context):
    """The Siril pipeline measures its shared canvas on each channel's RAW
    solved master, so an SPFC refusal (too few calibratable stars) at
    `post_stack_calibrate` / `calibrate_channel` blocks only that channel's
    merges — not the viewbox every channel of the object is placed on
    (2026-10-09: one refused B panel left all 17 Coalsack finishes absent)."""
    mosaic = _identify(test_context)
    config = _make_config()

    import pipelines.hybrid_siril_graxpert_oss_mosaic as siril
    g = trace(siril.for_mosaic._function, mosaic, config)
    viewboxes = [n for n in {*g, *(d for ds in g.values() for d in ds)}
                 if isinstance(n, Node)
                 and getattr(getattr(n.task, "_function", n.task), "__name__", "") == "compute_viewbox_measured"]
    assert len(viewboxes) == 1, viewboxes
    ops = _ancestor_ops(g, viewboxes[0])
    assert "integrate_with_siril" in ops and "plate_solve_image" in ops, ops
    assert not ops & {"post_stack_calibrate", "calibrate_channel", "hdr_merge_masters"}, ops
