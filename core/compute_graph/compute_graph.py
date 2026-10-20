import os
import ray
import structlog
from typing import Any
from beartype import beartype

from context import context
from stacks.dust_runs import DustEpochs
from cache.context_ref import ContextRef
from metadata.classes import DefectsDict, MetadataDict
from metadata.image_metadata import scatter_compute_image_metadata
from clustering import cluster_by_image_center
from calibration import (
    identify_calibration_stacks,
    associate_calibration_stacks,
)
from previews.stack_thumbnails import scatter_stack_thumbnails
from mosaics.identification import identify_mosaics, standalone_targets
from stacks.compute_graph import (
    prepare_stack_for_ui,
    process_stacks_for_ui,
    process_calibration_stacks_for_ui,
)
from targets.compute_graph import process_targets_for_ui
from targets.identification import identify_targets
from targets.types import TargetsDict
from mosaics.compute_graph import process_mosaics_for_ui
from mosaics.types import MosaicsDict
from dependency_graph import Graph, Node
from stacks.identification import identify_stacks
from stacks.composite_stack import RunsDict, StacksDict, CalibrationStacksDict
from decorators import node
from decisions.compute_graph import (
    scatter_customize_targets,
)
from decisions.settings import ClusteringSettings, DevelopmentSettings
from decisions.decisions import decisions_for_objects
from compute_graph.objects import Objects

log = structlog.get_logger()


@ray.remote(num_cpus=0.1)
@beartype
def store_calibration_stacks(
    calibration_stacks: CalibrationStacksDict,
) -> CalibrationStacksDict:
    context.calibration_stacks = calibration_stacks
    log.info("Stored calibration stacks", count=len(calibration_stacks))
    return calibration_stacks



# INVENTORY-COLLAPSE GUARD. Identification is derived from the library, and
# the library is derived from sidecars — a break anywhere upstream shows up
# here as a much smaller object set, which then silently skips cart items
# ("Cart entry refers to missing object") and empties viewer pages. On
# 2026-08-13 the published inventory fell from ~2000 stacks / 416 targets /
# 19 mosaics to 228 / 56 / 3 and NOTHING said so; it surfaced days later,
# by accident, while answering an unrelated question. A collapse is always
# either a real data loss or an upstream bug — never routine — so refuse the
# publish and make the operator look.
_COLLAPSE_FRACTION = 0.5      # publishing < half of last time is a collapse
_COLLAPSE_FLOOR = 20          # small libraries legitimately fluctuate


def _guard_collapse(kind: str, previous, count: int) -> None:
    prev = len(previous or ())
    if prev < _COLLAPSE_FLOOR or count >= prev * _COLLAPSE_FRACTION:
        return
    msg = (
        f"{kind}: refusing to publish {count} when the last pass published "
        f"{prev} — an inventory collapse means the library upstream is "
        f"broken (missing sidecars, an unreadable volume, a poisoned "
        f"identification cache). Investigate before continuing; set "
        f"ASTROFLOW_ALLOW_INVENTORY_COLLAPSE=1 if the drop is intended."
    )
    if os.environ.get("ASTROFLOW_ALLOW_INVENTORY_COLLAPSE") == "1":
        log.warning("inventory collapse ALLOWED by env override",
                    kind=kind, count=count, previous=prev)
        return
    log.error("inventory collapse refused", kind=kind, count=count,
              previous=prev)
    raise RuntimeError(msg)


@ray.remote(num_cpus=0.1)
@beartype
def store_stacks(stacks: dict[str, Any]) -> dict[str, Any]:
    assert None not in stacks.values()
    # stacks values are already ContextRefs (from gather_enriched_stacks)
    _guard_collapse("stacks", context.stacks, len(stacks))
    result = StacksDict(stacks)
    context.stacks = result
    log.info("Stored stacks", count=len(stacks))
    return result


@ray.remote(num_cpus=0.1)
@beartype
def store_targets(targets: dict[str, Any]) -> dict[str, Any]:
    assert None not in targets.values(), "ContextRef itself is None — identify_targets bug"
    # Validate that every ContextRef's underlying value is reachable.
    # We've hit a recurring bug where 2-4 target ContextRefs in the
    # durable cache go missing between writes — `ref.get()` returns
    # None. Surfacing that at write time, with the offending hashes,
    # so the failure is visible at the source instead of propagating
    # into `identify_mosaics` (which crashes on `target.center[0]`).
    missing = [h for h, ref in targets.items() if ref.get() is None]
    if missing:
        raise RuntimeError(
            f"store_targets: {len(missing)} target refs have no underlying "
            f"value in the durable cache: {missing[:8]}. This is the bug "
            f"that breaks identify_mosaics — investigate ContextRef.store / "
            f"diskcache write path."
        )
    _guard_collapse("targets", context.targets, len(targets))
    result = TargetsDict(targets)
    context.targets = result
    log.info("Stored targets", count=len(targets))
    return result


@ray.remote(num_cpus=0.1)
@beartype
def store_mosaics(mosaics: MosaicsDict) -> MosaicsDict:
    assert None not in mosaics.values()
    # mosaics values are already ContextRefs (from identify_mosaics)
    _guard_collapse("mosaics", context.mosaics, len(mosaics))
    context.mosaics = mosaics
    log.info("Stored mosaics", count=len(mosaics))
    return mosaics


@ray.remote(num_cpus=0.1)
def store_ui_stacks(ui_stacks: dict) -> None:
    context.ui_stacks = ui_stacks


@ray.remote(num_cpus=0.1)
def store_ui_calibration_stacks(ui_cal_stacks: dict) -> None:
    context.ui_calibration_stacks = ui_cal_stacks


@ray.remote(num_cpus=0.1)
def store_ui_targets(ui_targets: dict) -> None:
    context.ui_targets = ui_targets


@ray.remote(num_cpus=0.1)
def store_ui_mosaics(ui_mosaics: dict) -> None:
    context.ui_mosaics = ui_mosaics


@ray.remote(num_cpus=0.1)
@beartype
def combine_stacks(*ancestor_results: dict) -> StacksDict:
    """One spatial group's stacks, local and foreign (`stacks.foreign`), as
    the StacksDict `identify_targets` takes. Later dicts win a key: the
    builder lists the local stacks last, so a local stack keeps its stable
    hash over a foreign master of the same subframes."""
    merged = {k: v for d in ancestor_results for k, v in d.items()}
    assert None not in merged.values()
    return StacksDict(merged)


@ray.remote(num_cpus=0.1)
@beartype
def combine_dicts(*ancestor_results: dict[Any, Any]) -> dict[Any, Any]:
    assert isinstance(ancestor_results, tuple)
    for d in ancestor_results:
        assert None not in d.values(), f"### combine_dicts: None in dict {d}"
    result = {key: value for d in ancestor_results for key, value in d.items()}
    assert None not in result.values()
    return result


# Head-pinned: the foreign-stack admission reference reads the durable
# product version log (`stacks.foreign.local_pipeline_stamps`).
@ray.remote(num_cpus=0.1, resources={"diskcache": 0.001})
def scatter_spatial_groups(
    metadata_ref: ContextRef[MetadataDict],
    clustering_settings: ClusteringSettings,
    decision_overrides,
    *ancestor_results,
) -> Graph:
    from cache.context_ref import ContextRef
    from stacks.foreign import (
        assign_to_spatial_groups, identify_foreign_stacks, local_pipeline_stamps,
    )
    from stacks.sky_flat import identify_sky_flats, merge_sky_flats
    from stacks.star_flat import StarFlats, merge_star_flats

    # Three ancestor results in non-deterministic order:
    #   - cluster_by_image_center returns list[tuple[str, ...]]
    #   - identify_camera_fingerprints returns ContextRef[FingerprintMap]
    #   - the star-flat subgraph returns StarFlats (gated, keyed by stable_hash)
    # Discriminate by type.
    star_flats = next(r for r in ancestor_results if isinstance(r, StarFlats))
    spatial_groups = next(
        r for r in ancestor_results
        if not isinstance(r, (ContextRef, StarFlats))
    )
    fingerprint_map_ref = next(
        r for r in ancestor_results if isinstance(r, ContextRef)
    )

    graph = Graph({})

    # Library frontier for run-quiesce gating. This is the @ray.remote
    # builder — the sanctioned impure layer — so the one wall-clock
    # judgment lives here. Normally the frontier is the newest capture
    # time on disk (a pure data proxy for "now"): identification admits
    # only runs the data itself proves finished, so mid-night drains
    # re-derive identical stacks and cost minutes. Once the newest frame
    # is itself older than the gate (session over, downloads quiet), the
    # frontier is nudged past it so the final runs of the night close
    # too — the morning kick then bakes everything.
    import datetime as _dt
    from clustering.quiesce import _utc, effective_frontier
    gate = clustering_settings.run_quiesce_gate
    metadata = metadata_ref.get()
    capture_times = [
        _utc(m.header["DATE-OBS"])
        for m in metadata.values()
        if m.header.get("DATE-OBS") is not None
    ]
    frontier = max(capture_times) if capture_times else None
    if (
        frontier is not None
        and _dt.datetime.now(_dt.timezone.utc) - frontier >= gate
    ):
        frontier = frontier + gate

    def _newest_of(hashes) -> _dt.datetime | None:
        times = [
            _utc(metadata[h].header["DATE-OBS"])
            for h in hashes
            if h in metadata and metadata[h].header.get("DATE-OBS") is not None
        ]
        return max(times) if times else None

    # Calibration frames get their own subset frontier: light frames
    # streaming in must not re-key calibration identification (a ~10 min
    # recompute) every drain when no calibration frame is hot.
    cal_newest = _newest_of(
        h for h, m in metadata.items()
        if str(m.header.get("IMAGETYP", "")).lower() in ("bias", "dark", "flat")
    )
    cal_frontier = effective_frontier(cal_newest, frontier, gate)
    print(f"### quiesce frontier={frontier} cal_frontier={cal_frontier} gate={gate}")

    identify_cal_stacks_task = (
        identify_calibration_stacks,
        metadata_ref,
        clustering_settings.max_run_gap,
        clustering_settings.min_frames_in_stack,
        fingerprint_map_ref,
        cal_frontier,
        gate,
    )

    # Star flats join the calibration stacks first, so every association
    # below (cal-to-cal here, light↔flat in every resolve_calibration
    # caller) sees them as flat masters.
    merge_star_flats_task = (merge_star_flats, star_flats)
    graph[merge_star_flats_task] = [identify_cal_stacks_task]

    # Cal-to-cal association runs through the same decision framework as light
    # calibration (`resolve_calibration`), so it takes the decision overrides.
    associate_cal_task = (associate_calibration_stacks, decision_overrides)
    graph[associate_cal_task] = {merge_star_flats_task}

    from stacks.dust_runs import scatter_dust_runs, store_dust_epochs

    # ORDERED lists, not sets: these deps feed value-consuming nodes
    # (combine_dicts takes *ancestor_results), and dep order is part of the
    # cache key — canonicalize_graph now raises on unordered value-deps.
    # Other people's masters (`metadata.shared_masters` → `stacks.foreign`):
    # each joins the spatial group its pointing matches, so it is identified
    # into the same Target as the local stacks of that panel; the rest form
    # targets of their own. Admission (merge or only show) is decided
    # against the stamps of the masters this instance built.
    shared_ref = context["shared_ref"]
    shared_masters = shared_ref.get() if shared_ref is not None else {}
    local_stamps = local_pipeline_stamps() if shared_masters else ()
    shared_dirs = tuple(str(d) for d in (context.shared_dirs or ()))
    foreign_by_group, foreign_own_groups = assign_to_spatial_groups(
        shared_masters, metadata, spatial_groups, clustering_settings.overlap_fraction)
    if shared_masters:
        print(f"### shared masters: {len(shared_masters)} in {len(shared_dirs)} folder(s), "
              f"{sum(len(v) for v in foreign_by_group.values())} joining local pointings, "
              f"{sum(len(g) for g in foreign_own_groups)} in {len(foreign_own_groups)} pointing(s) of "
              f"their own; {len(local_stamps)} local pipeline stamp(s)")

    def foreign_task(hashes):
        return (identify_foreign_stacks, shared_ref, tuple(sorted(hashes)), local_stamps, shared_dirs)

    stack_tasks = []
    target_tasks = []
    for i, g in enumerate(spatial_groups):
        # Per-group effective frontier: a spatial group with no hot
        # frames gets None, keeping its identify key STABLE across
        # drains — only the group actively being shot re-keys.
        group_frontier = effective_frontier(_newest_of(g), frontier, gate)
        stack_task = (
            identify_stacks,
            metadata_ref,
            clustering_settings.angular_tolerance,
            g,
            clustering_settings.min_frames_in_stack,
            clustering_settings.max_run_gap,
            fingerprint_map_ref,
            group_frontier,
            gate,
        )
        members = stack_task
        if foreign_by_group.get(i):
            # Local LAST: a local stack keeps its stable hash over a foreign
            # master of the same subframes (`combine_stacks`, last wins).
            members = Node(combine_stacks, name=f"group-{i}")
            graph[members] = [foreign_task(foreign_by_group[i]), stack_task]

        target_task = Node(identify_targets, clustering_settings.angular_tolerance, name=str(i))
        customize_targets_task = Node(
            scatter_customize_targets, name=str(i)
        )

        # Each target identification depends on its own spatial group's stacks.
        graph[target_task] = {members}
        graph[customize_targets_task] = {target_task}

        stack_tasks.append(members)
        target_tasks.append(customize_targets_task)

    for j, own in enumerate(foreign_own_groups):
        rest = foreign_task(own)
        target_task = Node(identify_targets, clustering_settings.angular_tolerance, name=f"foreign-{j}")
        customize_targets_task = Node(scatter_customize_targets, name=f"foreign-{j}")
        graph[target_task] = {rest}
        graph[customize_targets_task] = {target_task}
        stack_tasks.append(rest)
        target_tasks.append(customize_targets_task)

    gather_stacks_task = Node(combine_dicts, name="combine_stacks")
    graph[gather_stacks_task] = stack_tasks
    graph[store_stacks] = {gather_stacks_task}

    # Dust-mote epochs per rig and filter, from the lights themselves
    # (stacks/dust_runs.py): one cached map per RUN, folded per rig and
    # filter, stored for `calibrate` and the stackers to look up.
    graph[scatter_dust_runs] = [gather_stacks_task]
    graph[store_dust_epochs] = {scatter_dust_runs}

    # Night-sky flats close the loop the other calibration types do not:
    # their source is a LIGHT stack the user promoted, so the calibration
    # chain genuinely waits on identification here. Nothing else changes —
    # with no promoted stack the merge is the identity and the whole branch
    # cache-hits. They join AFTER cal-to-cal association because a night-sky
    # flat's own dark resolves at plan time like any other step's.
    identify_sky_flat_task = (identify_sky_flats, decision_overrides)
    graph[identify_sky_flat_task] = [store_stacks]
    merge_sky_flat_task = Node(merge_sky_flats, name="merge_sky_flats")
    # ORDERED: merge_sky_flats(sky_flats, cal_stacks) consumes both
    # positionally.
    graph[merge_sky_flat_task] = [identify_sky_flat_task, associate_cal_task]
    graph[store_calibration_stacks] = {merge_sky_flat_task}
    graph[process_calibration_stacks_for_ui] = {merge_sky_flat_task}
    graph[store_ui_calibration_stacks] = {process_calibration_stacks_for_ui}
    # The builder embeds every `prepare_stack_for_ui` result in ITS cached
    # value, so an edit to the per-stack entry must miss the builder too —
    # its own source is unchanged, and it would otherwise replay the old
    # dict forever (2026-09-17: the new `role` field never reached the UI).
    process_stacks_task = (
        process_stacks_for_ui,
        decision_overrides,
        clustering_settings.min_frames_in_stack,
        prepare_stack_for_ui._fn_hash,
    )
    # Ordered: process_stacks_for_ui consumes both results positionally
    # (it unpacks by type, but the call HASH folds argument order).
    graph[process_stacks_task] = [store_stacks, store_calibration_stacks]
    graph[store_ui_stacks] = {process_stacks_task}

    gather_targets_task = Node(combine_dicts, name="combine_targets")
    graph[gather_targets_task] = target_tasks
    graph[store_targets] = {gather_targets_task}
    graph[process_targets_for_ui] = {gather_targets_task}
    graph[store_ui_targets] = {process_targets_for_ui}

    mosaics_task = (identify_mosaics, standalone_targets(decision_overrides))
    graph[mosaics_task] = {gather_targets_task}
    graph[store_mosaics] = {mosaics_task}
    graph[process_mosaics_for_ui] = {mosaics_task}
    graph[store_ui_mosaics] = {process_mosaics_for_ui}

    # gather_objects is the sole terminal node — its result (Objects) flows
    # to downstream graph builders via the chain.
    graph[gather_objects] = {
        store_stacks, store_targets, store_mosaics, store_calibration_stacks,
        store_ui_stacks, store_ui_targets, store_ui_mosaics, store_ui_calibration_stacks,
        store_dust_epochs,
    }

    return graph


@ray.remote(num_cpus=0.1)
@beartype
def build_regroup_graph(
    clustering_settings: ClusteringSettings,
    decision_overrides,
    metadata_ref: ContextRef[MetadataDict],
) -> Graph:
    # `scatter_compute_image_metadata`'s subgraph terminates in
    # `publish_library_refs`, which publishes BOTH `metadata_ref` and
    # `defects_ref` into context and returns `metadata_ref` as the
    # subgraph's final result. By the time this builder fires, both refs
    # are in context. `metadata_ref` flows in as our positional ancestor;
    # `defects_ref` is read from context.
    from stacks.identification import identify_camera_fingerprints
    from stacks.star_flat import identify_star_flats, scatter_star_flats

    defects_ref: ContextRef[DefectsDict] = context["defects_ref"]

    cluster_task = (
        cluster_by_image_center,
        metadata_ref,
        clustering_settings.overlap_fraction,
        clustering_settings.min_component_members,
    )
    fingerprint_task = (
        identify_camera_fingerprints,
        metadata_ref,
        defects_ref,
        clustering_settings.fingerprint_min_anchor_size,
    )
    # Star flats: candidates per (camera, focal length, filter) need the
    # camera fingerprints; every candidate's master is then built in its own
    # task (parallel across groups) and gated, BEFORE any light↔flat
    # association below, so the picker can offer it.
    star_flat_candidates_task = (identify_star_flats, metadata_ref)
    scatter_task = (
        scatter_spatial_groups,
        metadata_ref,
        clustering_settings,
        decision_overrides,
    )

    return Graph(
        {
            cluster_task: {},
            fingerprint_task: {},
            star_flat_candidates_task: [fingerprint_task],
            scatter_star_flats: [star_flat_candidates_task],
            # Ordered: scatter_spatial_groups consumes the results via
            # *ancestor_results, so dep order is part of its call hash.
            scatter_task: [cluster_task, fingerprint_task, scatter_star_flats],
        }
    )


@ray.remote(num_cpus=0.1)
@beartype
def gather_objects() -> Objects:
    """Snapshot all identified objects from context after regrouping."""
    return Objects(
        stacks=context.stacks or StacksDict(),
        targets=context.targets or TargetsDict(),
        mosaics=context.mosaics or MosaicsDict({}),
        calibration_stacks=context.calibration_stacks or CalibrationStacksDict({}),
        dust_epochs=context.dust_epochs or DustEpochs.empty(),
    )


@node
def preview_graph_done() -> None:
    pass


# A graph BUILDER, like `build_regroup_graph` — not a @node. It writes
# `local_context.targets` so the Subframe pipeline body can resolve
# `mosaic.targets` while its traces are built, and a side effect belongs in a
# builder, never in a cached op: as a @node this write was SKIPPED on every
# cache hit. Builders may read and write context; @node functions may not.
# `code_version` went with the @node — it existed only to bust that cache.
@ray.remote(num_cpus=0.1)
def build_preview_graph(
    development_settings: DevelopmentSettings,
    min_frames_in_stack: int,
    objects: Objects,
) -> Graph:
    from decisions.settings import ProcessingConfig
    from cache.context_ref import ContextRef

    cal_stacks_ref = ContextRef.store_large(objects.calibration_stacks)
    config = ProcessingConfig(
        calibration_stacks=cal_stacks_ref,
        development_settings=development_settings,
        min_frames_in_stack=min_frames_in_stack,
    )

    graph = Graph({})
    tasks = [(scatter_stack_thumbnails, cal_stacks_ref, objects.stacks)]

    # Subframe pipeline body resolves `mosaic.targets` against
    # `context.targets`; populate it here so each `run_pipeline`
    # task's trace has access to it.
    from context import local_context as context
    context.targets = objects.targets

    # Auto-preview: the 'Subframe' pipeline is dispatched over every
    # mosaic — semantically "every mosaic is implicitly in the cart
    # with the Subframe pipeline." Same per-pipeline sub-runtime
    # pattern as the cart-driven main-processing graph (see
    # `run_pipeline` above), so each Subframe trace's
    # `node_status_fn` / `graph_update_fn` fires inside its own
    # sub-runtime.
    #
    for mosaic_ref in objects.mosaics.values():
        mosaic = mosaic_ref.get()
        task = (run_pipeline, mosaic, config, "Subframe", "mosaic")
        graph[task] = set()
        tasks.append(task)

    graph[preview_graph_done] = tasks
    return graph


@ray.remote(num_cpus=0.1)
@beartype
def run_pipeline_done() -> None:
    """Per-pipeline terminal sentinel. `GraphRuntime` asserts a
    sub-graph has exactly one node with no descendants; a real
    pipeline trace has many (per-filter sinks, register-tap-path
    nodes, etc.), so we funnel them all into this no-op so the
    sub-runtime has a single completion handle."""
    pass


def _eager_trace_pipeline(
    pipeline_name: str, type_tag: str, entity, cfg,
) -> "Graph | None":
    """Trace a pipeline on the head node (in-process, not via Ray).

    Same code path as the lazy trace inside `run_pipeline` — calls
    `pipeline_fn(entity, cfg)` from the registry and walks tap points.

    The graph builders no longer call this. It existed to pre-populate the
    global ephemeral-file registry before any consumer could dispatch, which
    made "eager and lazy traces are bit-identical" a load-bearing platform
    invariant — and that invariant was a recurring source of bugs. Scope-local
    deletion needs no census (a subgraph reclaims only what it produced), so
    the eager pass is gone and every cart pipeline is traced ONCE, lazily, in
    `run_pipeline` (which stores the tap snapshot itself).

    Still used by structural tests that want a pipeline's traced graph and its
    tap snapshot without running anything. Returns None if the pipeline body
    bails (e.g. MASTER-mode mosaic with missing panels).
    """
    from pipelines.registry import get_pipeline
    from products.product import object_class
    from dependency_graph.graph_runtime import GraphRuntime
    from dependency_graph.workflow_graph_adapter import store_workflow_graph

    pipeline_fn = get_pipeline(pipeline_name, processes=object_class(type_tag))
    graph = pipeline_fn(entity, cfg)
    if graph is None:
        return None
    # Topology-only snapshot so the viewer can render structure
    # before any node completes. Lives here, NOT in the @pipeline
    # runner, so dashboard tap-extraction traces don't pay for it.
    store_workflow_graph(
        (type_tag, entity.stable_hash, pipeline_name), GraphRuntime(graph),
    )
    return graph


def _version_supersede_map(pipeline_name: str, graph) -> dict:
    """`{node: version_log_key}` for every display-tap node in `graph`.
    Attached to the sub-graph `run_pipeline` returns; the runtime consults
    it at node completion to record the landed version and delete the
    superseded (N-1) files.

    Phase 1b (2026-08-08): the map carries the product IDENTITY only —
    (object, pipeline, display, filter), which is hash-generation-free.
    The landed PATH comes from the node's ACTUAL runtime result at
    `_mark_done`, never from `resolve_output_paths`: the predicted path
    diverged from runtime landings, so supersede's landed-file guard made
    the whole mechanism silently refuse for a week while duplicate
    generations piled up on disk."""
    from products import version_log
    return {
        node: version_log.key(tap["object_type"], tap["object_id"],
                              pipeline_name, tap["display"], tap["filter"])
        for node, tap in extract_pipeline_taps(graph)
    }


def _store_tap_snapshot(pipeline_fn, pipeline_name: str, entity, graph) -> None:
    """Persist a cart entry's complete display-tap list so the Ray-free web
    server can read it without tracing.

    CONTENT-ADDRESSED key — `(pipeline_source_hash, stable_hash, content_hash,
    pipeline)`. A pipeline-code change (new source hash) or new images on the
    object (new content_hash) yields a NEW key, so the web UI — which computes
    the SAME key from the live object + pipeline source — always reads the
    current trace; stale snapshots are harmless orphans, never requested, aged
    out by LRU. Every build REWRITES its key, because the key does not cover
    decisions. One atomic write (no two-phase `record_path` to drift). The web
    worker filters the full list to the queried routing entity on read."""
    from cache.context_ref import ContextRef
    from dependency_graph.trace_paths import (
        _walk_and_resolve_filenames_tagged, pipeline_source_version,
    )

    # ALWAYS re-walk and re-store, even when a snapshot sits under this key.
    # The key cannot see everything the trace reads: a run quarantine (any
    # decision override, any context the trace consults) leaves source,
    # stable and content hash unchanged while changing the paths. The old
    # skip-if-present kept the pre-quarantine paths forever — 7098 Panels 2
    # and 4 rebuilt correctly and still read "out of date" with no master
    # outline (2026-10-14). The walk costs 0.2-0.3 s; the graph is already
    # traced.
    #
    # This cart-item snapshot is the COMPLETE trace (every filename node) for
    # the sweep, archive, discard and polygon readers. The display taps in it
    # are ALSO filed per ROUTED entity by `_store_product_taps` below — a
    # mosaic's trace holds its stacks' `Master` taps, its targets' `Channel`
    # taps and its calibration stacks' masters, and each of those objects is
    # viewed by its own URL, which names no cart item.
    #
    # `pipeline_source_version` folds the deployed code SHA into the source
    # component so an op edit re-keys the snapshot (the reader `_snapshot_key`
    # uses the SAME helper, keeping writer/reader keys in lockstep).
    key = (
        pipeline_source_version(pipeline_fn._function),
        entity.stable_hash, entity.content_hash, pipeline_name,
    )
    # The COMPLETE trace (every filename node, not just display-taps): the web
    # server reads it for the dashboard taps (display+filter), the mosaic
    # polygon overlays (stack_files by identity), and file opens — all Ray-free.
    entries = _walk_and_resolve_filenames_tagged(graph)
    if not entries:
        return
    ref = ContextRef.store(("tap_snapshots", str(key)), entries)
    context.or_equals("tap_snapshots", {key: ref})
    _store_product_taps(pipeline_name, entries)

    # Phase 1b: the version log is written ONLY at runtime landings
    # (`_mark_done` → `version_log.supersede` with the node's actual result
    # path). The prediction-time `record()` that lived here wrote predicted
    # paths into the log — the ledger entries that diverged from what the
    # runtime actually built, pinning phantoms and orphaning real files.


def _store_product_taps(pipeline_name: str, entries: list[dict]) -> None:
    """File every display tap of a trace under the entity it ROUTES to, so
    "the images of object X in pipeline P" is one `context.get` for ANY
    object — a mosaic, or a stack / target / calibration stack that is only
    ever an interior node of some cart item's trace and has no cart entry of
    its own. Same content-addressing as the cart snapshot (deployed code
    version + the routed entity's own `content_hash`), so a code change or
    new frames re-key it and the reader misses until the re-trace lands —
    never a stale answer. One `set_many` per cart item × pipeline."""
    from dependency_graph.trace_paths import product_taps_key
    by_entity: dict[tuple, list[dict]] = {}
    for e in entries:
        if e.get("display") is None:
            continue
        key = product_taps_key(e["object_type"], e["object_id"],
                               e["content_hash"], pipeline_name)
        by_entity.setdefault(key, []).append(
            {"display": e["display"], "filter": e["filter"], "path": e["path"]})
    if by_entity:
        context.set_many(by_entity)


@ray.remote(num_cpus=0.1)
def run_pipeline(entity, cfg, pipeline_name: str, type_tag: str) -> "Graph":
    """Reach for the cached trace, funnel terminals into the
    sub-runtime sentinel. Returns the Graph so the parent runtime
    spins up a child `GraphRuntime`.

    Trace is via the @node-cached `trace_pipeline` — body only re-runs
    when the pipeline source, entity, or cfg changes. The cached graph
    is pure data; tap paths are derived on demand by the dashboard from
    the same cached trace + disk (no runtime tap recording).

    Returning a Graph from a `@ray.remote` task is how `run_graph`
    spins up a child `GraphRuntime`. That sub-runtime reads the
    tracer's `node_status_fn` (= `store_node_status`) and
    `graph_update_fn` (= `partial(store_workflow_graph, pipeline_key)`)
    straight off the returned Graph — so per-pipeline live state
    updates fire naturally as nodes complete, without parent-level
    plumbing.
    """
    from pipelines.registry import get_pipeline
    from products.product import object_class
    from dependency_graph import Graph

    pipeline_fn = get_pipeline(pipeline_name, processes=object_class(type_tag))

    # Cached trace. The graph is freshly deserialized from diskcache
    # on each cache hit so in-place mutation downstream is safe. The version
    # arg folds in the deployed code SHA so editing ANY op the pipeline calls
    # (not just the pipeline body) re-derives the trace — otherwise the cached
    # graph keeps the op's old call_hash and no rebuild is triggered.
    from dependency_graph.trace_paths import pipeline_source_version
    source_hash = pipeline_source_version(pipeline_fn._function)
    graph = ray.get(trace_pipeline.remote(
        pipeline_name, source_hash, entity, cfg,
    ))
    if graph is None:
        # MASTER-mode mosaics with missing panels legitimately bail.
        # Surface an empty graph so the parent task's `node_completed`
        # still gets a Graph and closes out the cart entry cleanly.
        return Graph({})

    # Persist the tap snapshot for the Ray-free web server. Done HERE (not in
    # the @node-cached build_main_processing_graph, which is a cache-hit on an
    # unchanged cart and would never re-store on restart) because run_pipeline
    # runs on every `run_entire_graph`. The trace above is the cached
    # `trace_pipeline` (no 500 s re-trace), and `_store_tap_snapshot` skips the
    # walk entirely when the content-addressed snapshot already exists — so a
    # restart with an unchanged cart just confirms presence and moves on.
    _store_tap_snapshot(pipeline_fn, pipeline_name, entity, graph)

    # Deterministic N-1 supersession: the runtime deletes a display tap's
    # previous version at the completion of the node that writes the new one
    # (`version_log.supersede`). Attached every run — including unchanged
    # rebuilds, where the same-hash entry makes supersede a no-op.
    graph.version_supersede = _version_supersede_map(pipeline_name, graph)

    # GraphRuntime asserts a sub-graph has exactly one terminal. Funnel
    # every current terminal into a sentinel so the child sub-runtime
    # has a single completion handle.
    depended_on = {d for ds in graph.values() for d in ds}
    terminals = [n for n in graph if n not in depended_on]
    graph[run_pipeline_done] = terminals
    return graph


@ray.remote(num_cpus=0.1)
@beartype
def main_processing_graph_done() -> None:
    pass


def _find_identity_arg(graph, node):
    """Walk `node.args` + dep ancestors for the first value that has
    `stable_hash` (i.e. an identity-bearing Stack/Target/Mosaic).

    The tracer's convention is "literals first, deps trailing": deps
    get stripped from `Node.args` and stored on graph edges, so an op
    like `stack_files(stack, config, *cal_masters)` where `stack` is a
    dep ends up with `args = (config,)` and `deps = [stack_node, ...]`.
    The entity is somewhere down that dep chain — usually as the
    leading literal of the first dep — so we recurse.
    """
    from dependency_graph.classes import Node
    for a in node.args:
        if hasattr(a, "stable_hash"):
            return a
    for d in graph.get(node, []) or ():
        if isinstance(d, Node):
            found = _find_identity_arg(graph, d)
            if found is not None:
                return found
    return None


def extract_pipeline_taps(graph):
    """Pure walk of a traced `graph` yielding `(node, tap)` per @makes_image
    tap, where `tap` is `{object_type, object_id, content_hash, display,
    filter}` — the ROUTING entity (URL type + stable hash + content hash)
    and the tap's display/filter.

    "Tap" = any Node whose task has a `_display` attribute (set by
    `@makes_image(display=...)`). Same-key collisions inside a single
    graph (a pipeline that calls `merge_images` twice on the same
    (mosaic, filter), say) get the `display` suffixed with ` 2`, ` 3`,
    …; the first occurrence keeps its bare display so single-merge
    pipelines don't show "Merged 1".

    No side effects, no graph mutation.
    """
    from dependency_graph.classes import Node
    from dependency_graph.node_introspection import filter_of
    from products.product import url_type_of

    if graph is None:
        return

    # Per-(type, id, display, filter) sequence count for disambiguation.
    seq: dict[tuple, int] = {}

    for n in list(graph.keys()):
        if not isinstance(n, Node):
            continue
        task = n.task
        fn = getattr(task, "_function", task)
        # NODE-level attr first: a presented subgraph builder carries its
        # terminal's `display` (`present_terminal` — builder ≡ terminal),
        # so a deliverable that lives one level down still registers its
        # tap here — and therefore in the snapshot the UI reads and the
        # version-supersede map built from these yields.
        base_display = (getattr(n, "_display", None)
                        or getattr(fn, "_display", None)
                        or getattr(task, "_display", None))
        if base_display is None:
            continue
        # Ephemeral nodes (inside an `ephemeral_intermediates()` scope and not
        # `final`) are reclaimed after their consumers — not user-facing — so
        # they register no tap, even with a `display=`. `display=` "activates"
        # only when the node is kept (unwrapped or final).
        if getattr(n, "_ephemeral", False):
            continue
        # Find the entity: walks `n.args` for an instance with
        # `stable_hash`, then descends into upstream deps if needed.
        # Lets ops with `stack` as a trailing dep (e.g. `stack_files
        # (stack, config, *cal_masters)`) route taps the same as ops
        # with `stack` as a leading literal.
        entity = _find_identity_arg(graph, n)
        if entity is None:
            continue
        # The routing entity is stamped in URL/cart form (`url_type_of`: a
        # `StarFlatStack` files as "calibration_stack"), so a reader keyed
        # by URL never has to know which class produced the tap.
        object_type = url_type_of(entity)
        object_id = entity.stable_hash
        # Filter precedence: explicit `filter=` literal param wins (mosaic
        # ops take it as a literal arg), else the entity's intrinsic filter
        # (every Stack has exactly one). Centralized in `filter_of`.
        filter_value = filter_of(n, list(n.args) + [entity])
        key = (object_type, object_id, base_display, filter_value)
        seq[key] = seq.get(key, 0) + 1
        display = (
            base_display if seq[key] == 1
            else f"{base_display} {seq[key]}"
        )
        yield n, {"object_type": object_type, "object_id": object_id,
                  "content_hash": getattr(entity, "content_hash", ""),
                  "display": display, "filter": filter_value}


# Pin to the head node. trace_pipeline is orchestration, not compute: its
# @node wrapper reads/writes the result to the *local* diskcache of
# whatever node the task lands on, and its result (a Graph) is file-less,
# so it has no shared-disk source of truth to fall back on (unlike
# file-writing ops, which preflight validates against disk). Left to the
# default SPREAD, it can land on a worker and read/write that worker's
# local diskcache — the "silent-write-to-worker-disk" bug ray_init.py
# warns about, which is why the orchestrator actors are head-pinned too.
# Only the head carries the `diskcache` resource, so requesting a sliver
# of it (the same idiom as ContextActor/StateActor) keeps tracing on the
# head against the canonical diskcache.
@node(resources={"diskcache": 0.001})
def trace_pipeline(
    pipeline_name: str,
    pipeline_source_hash: str,
    entity,
    cfg,
):
    """Pure trace of one pipeline. Returns the traced `Graph`, or `None`
    on a trace failure (the pipeline raised, or wasn't registered).

    Cached via @node on (source hash + args). `pipeline_source_hash`
    is in the args so changing the pipeline body invalidates the cache
    — @node alone would only hash THIS function's body, not the
    looked-up pipeline_fn. `cfg` is the FULL ProcessingConfig (frozen
    dataclass, hashable), so dashboard-trace cfg and live-runtime cfg
    naturally get separate cache entries.

    Side-effect-free: tap paths are derived on demand by the dashboard
    (trace + disk), not committed anywhere here.
    """
    from pipelines.registry import registered_pipelines

    pipeline_fn = registered_pipelines().get((pipeline_name, type(entity)))
    if pipeline_fn is None:
        return None

    try:
        return pipeline_fn(entity, cfg)
    except Exception as e:
        import traceback
        print(f"### trace_pipeline failed: {pipeline_name} for {type(entity).__name__}: {e}")
        traceback.print_exc()
        return None


def _decision_closure_ids(entity, entity_id: str, objects_by_type) -> frozenset:
    """The containment closure of a cart entity, as stable-hash ids: the
    entity itself, plus (for a mosaic) its targets and their stacks, plus
    (for a target) its stacks. Pure id-tuple walking — `Mosaic.target_ids`
    and `Target.stack_ids` are plain hash tuples, resolved against the
    build's own `objects` dicts (never live context). Used to scope the
    decisions baked into a pipeline's ProcessingConfig."""
    from cache.context_ref import ContextRef

    def _unwrap(v):
        return v.get() if isinstance(v, ContextRef) else v

    ids = {entity_id}
    sh = getattr(entity, "stable_hash", None)
    if sh:
        ids.add(sh)
    target_ids = tuple(getattr(entity, "target_ids", ()) or ())
    targets_dict = objects_by_type.get("target", {})
    targets = [entity] if hasattr(entity, "stack_ids") else []
    for tid in target_ids:
        ids.add(tid)
        ref = targets_dict.get(tid)
        if ref is not None:
            targets.append(_unwrap(ref))
    for t in targets:
        for sid in tuple(getattr(t, "stack_ids", ()) or ()):
            ids.add(sid)
    return frozenset(ids)


@node
def build_main_processing_graph(
    development_settings: DevelopmentSettings,
    min_frames_in_stack: int,
    decision_overrides,
    shopping_cart: tuple,
    code_version: str,
    objects: Objects,
) -> Graph:
    # `code_version` (the deployed cluster SHA) is a cache-key arg only — the
    # body never reads it. It forces this @node-cached composed graph to
    # re-derive on ANY code change, not just an edit to this builder's own
    # source. Without it, editing an op deep in a pipeline (e.g.
    # `post_process_stack_oss`) leaves the cached graph holding the op's old
    # call_hash, so preflight sees the old outputs as satisfied and nothing
    # rebuilds. Mirrors `trace_pipeline`'s `pipeline_source_hash` arg.
    from decisions.settings import ProcessingConfig
    from frozendict import frozendict as fd

    stacks = objects.stacks
    targets = objects.targets
    mosaics = objects.mosaics
    from cache.context_ref import ContextRef
    cal_stacks_ref = ContextRef.store_large(objects.calibration_stacks)
    dust_ref = ContextRef.store_large(objects.dust_epochs)

    log.info("Building main processing graph", shopping_cart=shopping_cart)

    # Cart's URL-friendly type-string → @pipeline-registered class via the
    # canonical `object_class` (the cart and routes use strings; @pipeline
    # decorators key on the actual class).
    from products.product import object_class
    objects_by_type = {"stack": stacks, "target": targets, "mosaic": mosaics}

    # Calibration stacks are SHARED infrastructure: `calibration_plan` runs
    # `resolve_calibration` on cal stacks themselves (a flat resolving its
    # dark), so decisions parented on a cal-stack id are consumed inside any
    # pipeline that might select that master. They join every closure —
    # a cal decision legitimately invalidates all potential consumers.
    cal_stack_ids = frozenset(objects.calibration_stacks.keys())

    def config_for(entry_type, entity, entity_id):
        # Scope the baked decisions to the entity's CONTAINMENT CLOSURE plus
        # the shared cal-stack ids — the cache key of an image may depend
        # only on decisions pertaining to that object, a contained object,
        # or shared calibration. One uniform filter for all entry types
        # (a stack's closure is itself; run-level keys carry the stack as
        # parent, matching the old decisions_for_stack semantics exactly).
        # Baking the GLOBAL dict here (the old mosaic/target behaviour)
        # meant any decision write anywhere re-keyed every mosaic's tail —
        # and the old stack filter had the dual gap: cal-parented decisions
        # were invisible to the ops that consume them.
        closure = (
            frozenset({entity_id})
            if entry_type == "stack"
            else _decision_closure_ids(entity, entity_id, objects_by_type)
        )
        decisions = decisions_for_objects(
            decision_overrides, closure | cal_stack_ids,
        )
        return ProcessingConfig(
            calibration_stacks=cal_stacks_ref,
            development_settings=development_settings,
            min_frames_in_stack=min_frames_in_stack,
            decisions=decisions,
            dust_epochs=dust_ref,
        )

    from pipelines.registry import registered_pipelines
    registered = registered_pipelines()
    graph = Graph({})
    pipeline_tasks = []
    for entry in shopping_cart:
        # Cart entries: `(type, id, pipeline_name)`. Each entry
        # becomes a `run_pipeline` Ray task that traces the
        # pipeline body at runtime and returns its Graph as a
        # dynamic sub-graph. The returned Graph carries the
        # tracer's per-pipeline callbacks (`set_workflow_key`),
        # so the per-pipeline workflow-graph state updates fire
        # naturally without any plumbing here.
        type, id, pipeline_name = entry
        if id not in objects_by_type[type]:
            log.warning("Cart entry refers to missing object", entry=entry)
            continue
        if (pipeline_name, object_class(type)) not in registered:
            log.warning("Unknown pipeline in cart", entry=entry)
            continue
        entity = objects_by_type[type][id].get()
        cfg = config_for(type, entity, id)
        task = (run_pipeline, entity, cfg, pipeline_name, type)
        graph[task] = set()
        pipeline_tasks.append(task)

    graph[main_processing_graph_done] = pipeline_tasks
    return graph


def build_preview_compute_graph() -> "Graph | None":
    """Phase 1 of the two-phase run: identification + EVERY mosaic's Subframe
    preview, nothing else. Run to completion BEFORE the main graph so the
    default views exist before hours of heavy work start — and so every
    drain re-establishes "new data gets previews first" (the replan re-runs
    this phase as cache hits + the new subs' previews only). None when the
    dev settings skip previews (tests)."""
    settings = context.settings
    dev = settings.development_settings
    if dev.skip_preview_graph:
        return None
    regroup = (build_regroup_graph, settings.clustering_settings,
               settings.decision_overrides)
    min_frames = settings.clustering_settings.min_frames_in_stack
    return Graph({
        regroup: {scatter_compute_image_metadata},
        (build_preview_graph, dev, min_frames): {regroup},
    })


def build_compute_graph() -> Graph:
    settings = context.settings
    shopping_cart = tuple(context.shopping_cart or [])

    regroup = (build_regroup_graph, settings.clustering_settings, settings.decision_overrides)

    dev = settings.development_settings
    min_frames = settings.clustering_settings.min_frames_in_stack
    overrides = settings.decision_overrides
    # The deployed cluster SHA — a cache-key arg for the composed graph builders
    # so ANY code change re-derives them (an op edit is otherwise invisible to
    # these @node caches, whose own source is unchanged). "" in dev/tests →
    # identical to the pre-change key.
    from state.code_version import deployed_code_version
    code_version = deployed_code_version()

    graph = {
        regroup: {scatter_compute_image_metadata},
        (build_main_processing_graph, dev, min_frames, overrides, shopping_cart, code_version): {regroup},
    }
    if not dev.skip_preview_graph:
        graph[(build_preview_graph, dev, min_frames)] = {regroup}

    return Graph(graph)


# TODO: Add a cleanup task that deletes unreferenced stuff from the cache.
# Can I use some kind of automatic ref-counting?
