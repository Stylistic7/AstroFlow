import os
import ray
import threading
from functools import reduce
from operator import add
import time
from typing import Callable, Any

from cache.types import NeverRun
from dependency_graph import (
    Node,
    Graph,
    CanonicalGraph,
    GraphStatus,
    NodeStatusFn,
    ResultDict,
    RunStats,
    Skipped,
    GenericRemoteFunction,
    canonicalize_graph,
)
from dependency_graph.graph_runtime import GraphRuntime
from .debugging import print_graph, node_to_string

_MIN_STATUS_UPDATE_INTERVAL_SECS: float = 5.0


def _ray_object_count() -> int:
    worker = ray._private.worker.global_worker
    return len(worker.core_worker.get_all_reference_counts())


def _ray_object_store_duplicates() -> dict[tuple[str, str], int]:
    """Find duplicate ContentHashed objects in the Ray object store.

    Returns a dict of (type_name, content_hash) -> count for any
    objects that appear more than once.
    """
    from cache.content_hashed import ContentHashed

    worker = ray._private.worker.global_worker
    counts = worker.core_worker.get_all_reference_counts()
    seen: dict[tuple[str, str], int] = {}
    for obj_id_bytes in counts:
        obj_ref = ray.ObjectRef(bytes.fromhex(obj_id_bytes.decode()))
        try:
            obj = ray.get(obj_ref, timeout=5)
        except Exception:
            continue
        if isinstance(obj, ContentHashed) and obj.content_hash:
            key = (type(obj).__name__, obj.content_hash)
            seen[key] = seen.get(key, 0) + 1
    return {k: v for k, v in seen.items() if v > 1}


StatusFn = Callable[[GraphStatus], Any]


class AncestorsNeverRan(Exception):
    pass


class AncestorsFailed(Exception):
    pass


def matching_nodes(
    graph: CanonicalGraph, fn: GenericRemoteFunction
) -> list[Node]:
    def _matches(node: Node) -> bool:
        return (
            node.task._function.__module__ == fn._function.__module__
            and node.task._function.__name__ == fn._function.__name__
        )

    return [node for node in graph if _matches(node)]


def _submit_node(
    rt: GraphRuntime, node: Node, dry_run: bool, debug: bool,
    node_id: "str | None" = None, memory: "int | None" = None,
) -> ray.ObjectRef:
    """Submit a node to Ray for execution. Returns an ObjectRef.

    When ``node_id`` is given (a memory/resource-gated task the scheduler placed
    on a specific node), pin the task there with ``NodeAffinitySchedulingStrategy
    (node_id, soft=False)`` so Ray honours the placement the admission gate chose
    instead of SPREAD-ing it elsewhere. ``memory`` MUST be passed for a pinned
    ``auto_memory`` op: ``node.task.options(...)`` is the RAW Ray handle and
    bypasses the ``memory=estimate`` injection that ``node.task.remote`` does, so
    the node's raylet would otherwise reserve nothing. ``.options`` overrides only
    the fields we pass, so baked ``resources={"PixInsight":1}`` etc. still apply.
    ``node_id is None`` → the light-task path, unpinned and SPREAD-scheduled as
    before.

    Raises AncestorsNeverRan or AncestorsFailed if the node should be
    skipped due to ancestor state.
    """
    start_time = time.time()

    fn_args = rt.make_fn_args(node)

    if any(isinstance(x, Exception) for x in fn_args):
        raise AncestorsFailed(node)

    if any(x in [NeverRun, Skipped] for x in fn_args):
        raise AncestorsNeverRan(node)

    task = node.task
    if node_id is not None:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        opts = {"scheduling_strategy": NodeAffinitySchedulingStrategy(node_id, soft=False)}
        if memory is not None:
            opts["memory"] = memory
        task = node.task.options(**opts)

    # TODO: This is hacky. Make it more explicit.
    try:
        result = task.remote(*fn_args, dry_run=dry_run)
    except:
        result = task.remote(*fn_args)

    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:
        print(
            f"### _submit_node took {elapsed_time:.3f} s "
            f"for node {node_to_string(node)}"
        )
    return result


def _retrieve_logs(rt: GraphRuntime, node: Node) -> list[Any]:
    """Retrieve captured logs from context for a completed node."""
    from context import context
    from hashing import hash_source_code, hash_call
    try:
        fn_args = rt.make_fn_args(node)
        raw_func = node.task._function
        fn_hash = hash_source_code(raw_func)
        call_hash = hash_call(fn_hash, *fn_args)
        return context.get(("logs", raw_func.__name__, call_hash), [])
    except Exception:
        return []


class GraphCancelled(Exception):
    """Raised when a graph run is drained via drain_event — in-flight
    tasks finished naturally, but waiting tasks were never submitted."""
    pass


def _apply_preflight(graph: Graph) -> Graph:
    """Predict every node's call_hash statically, check disk for
    `final=True` / terminal outputs, and return a filtered Graph that
    contains only the nodes that actually need to run. Falls out of
    content-addressing — see `dependency_graph/preflight.py` for the
    proof sketch. Errors are swallowed: preflight is an optimization,
    never load-bearing for correctness.
    """
    if not graph:
        return graph
    try:
        from dependency_graph.preflight import (
            predict_call_hashes, nodes_to_dispatch, _with_source_nodes,
        )
        # Normalise bare-dep sources into nodes exactly as GraphRuntime does, so
        # the prune set + reconstruction below share that node set (and the
        # top-level meta-graph no longer trips a spurious 'cycle or orphan').
        norm = _with_source_nodes(graph)
        predicted = predict_call_hashes(norm)
        to_run = nodes_to_dispatch(norm, predicted)
    except Exception as e:
        print(f"### preflight skipped due to {type(e).__name__}: {e}")
        return graph
    if len(to_run) == len(norm):
        return graph
    return Graph(
        {n: [d for d in norm[n] if d in to_run] for n in to_run},
        graph_update_fn=getattr(graph, "graph_update_fn", None),
        node_status_fn=getattr(graph, "node_status_fn", None),
    )


def _declared_custom_resources(task) -> dict:
    """The op's declared *custom* Ray resources (e.g. ``{"PixInsight": 1}``),
    or ``{}``. Resource-name agnostic.

    `@node` / `@cached.remote` expose them on ``_ray_resources`` (set in
    `cached.remote`); a raw `@ray.remote` ``RemoteFunction`` carries them on
    ``_resources``. CPU / GPU / memory are intentionally NOT consulted — Ray's
    own scheduler plus ``MAX_IN_FLIGHT`` already govern those. This gate exists
    only for scarce, explicitly-declared custom resources that serialize work
    (e.g. the single PixInsight slot): without it, run_graph hands Ray a deep
    backlog of such tasks and Ray arbitrates the slot in submission order,
    defeating the `node_sort_key` priority."""
    r = getattr(task, "_ray_resources", None)
    if r is None:
        r = getattr(task, "_resources", None)  # raw @ray.remote RemoteFunction
    return r or {}


def _fits(need: dict, avail: dict) -> bool:
    """Whether every resource in ``need`` is currently available."""
    return all(avail.get(name, 0.0) >= amt for name, amt in need.items())


# Headroom kept free above what a task is predicted to need, so the machine can
# drift into "yellow" pressure but not run-and-crash. Absolute (not a fraction):
# the danger is PixInsight, a SEPARATE persistent process whose RSS Ray can't
# see and that spikes fast — a fixed buffer covers that estimation blind spot on
# any machine size. Override live (frozen-SHA deploys still read env at launch)
# via ASTROFLOW_MEM_SAFETY_MARGIN_GB.
_MEM_SAFETY_MARGIN_BYTES = int(
    float(os.environ.get("ASTROFLOW_MEM_SAFETY_MARGIN_GB", "6")) * (1024 ** 3)
)

# Only tasks estimated at/above this (or declaring a custom resource) go through
# the per-node placement gate + NodeAffinity pin. Below it, a task dispatches
# unpinned/SPREAD as before — Ray still reserves its `memory=` estimate, which
# bounds it fine because a small op's estimate is accurate (unlike PixInsight /
# rc-astro CLI heavies, whose real footprint Ray can't see — those are all far
# above this). This keeps the ~30k high-volume 1 GB `compute_image_metadata`
# tasks (and any other small op) on the cheap fast path: no per-node selection,
# no pin, and — since a pass with no gated task never samples — no fan-out.
_GATE_MIN_BYTES = 2 * (1024 ** 3)


def _estimate_node_memory(rt, node) -> int:
    """Predicted peak-RSS bytes for ``node``, for the admission gate.

    Returns 0 for any node without an ``auto_memory`` estimator — the light
    tasks, which stay ungated (governed only by ``MAX_IN_FLIGHT``). For an
    ``auto_memory`` node, resolves its call args and runs the SAME estimator Ray
    uses at submit (``profiling.memory_estimate.estimate_memory``), so the gate
    and Ray's own ``memory=`` reservation agree. Never raises — an estimator
    failure degrades to 0 (ungated) rather than stalling the run."""
    task = getattr(node, "task", None)
    est = getattr(task, "_auto_memory_estimator", None)
    if est is None:
        return 0
    try:
        args = rt.make_fn_args(node)
        return int(est(task._function, tuple(args), {}))
    except Exception:
        return 0


def _custom_fits(need: dict, capacity: dict, reserved: dict) -> bool:
    """Whether a node with total custom ``capacity`` and ``reserved`` (both
    ``{resource: amount}``) has every resource in ``need`` free."""
    return all(
        capacity.get(r, 0.0) - reserved.get(r, 0.0) >= amt for r, amt in need.items()
    )


def _select_node(
    need_mem: int,
    need_custom: dict,
    free_by_node: dict,
    total_by_node: dict,
    reserved_by_node: dict,
    capacity_by_node: dict,
    reserved_custom_by_node: dict,
    *,
    any_in_flight: bool,
    pressure_by_node: "dict | None" = None,
    margin: int = _MEM_SAFETY_MARGIN_BYTES,
) -> "str | None":
    """Pick a node that can run a task needing ``need_mem`` bytes and
    ``need_custom`` resources RIGHT NOW, or ``None`` to hold it for a later pass.

    Candidate = a MEASURED node (present in ``free_by_node``) that has every
    declared custom resource free (``capacity - reserved >= amt``). This single
    per-node check subsumes the old cluster-aggregate custom-resource gate — and
    naturally keeps head-only resources (``fastest_storage``/``diskcache``) off
    workers, since only the head's capacity carries them.

    Memory admission for a candidate ``n`` uses the two-leg min:
        effective_free(n) = min(free_by_node[n], total_by_node[n] - reserved_by_node[n]) - margin
    * ACTUAL leg (measured psutil free) sees PixInsight's out-of-process RSS and
      anything else Ray never reserved.
    * RESERVED leg (total - reserved) covers the lag before a just-launched task
      has allocated and shows up in the measurement.
    A node whose OS memory PRESSURE is yellow/red (``pressure_by_node[n] >= 2``)
    is dropped from candidates regardless of the byte math — the macOS-accurate
    brake the ACTUAL leg misses (``available`` over-counts reclaimable pages).
    We place on the ROOMIEST candidate. ``need_mem == 0`` (a custom-resource-only
    task) passes the memory test trivially.

    If nothing fits and something is in flight, return ``None`` (hold; a
    completion will free room). If nothing is in flight either, we MUST place
    something or the run stalls forever — the idle-escape returns the roomiest
    candidate (or, if no measured node satisfies the custom resources, the
    largest-capacity node that does, so a missing sample can't wedge the run).
    It then runs alone and OOMs loudly if genuinely too big — traceable, unlike a
    silent hang."""
    pressure = pressure_by_node or {}
    scored = []  # (effective_free, node_id)
    for nid, avail in free_by_node.items():
        if not _custom_fits(
            need_custom,
            capacity_by_node.get(nid, {}),
            reserved_custom_by_node.get(nid, {}),
        ):
            continue
        # Pressure veto: a node under OS memory pressure (yellow/red, level >= 2)
        # takes no new heavy task, even when the free/reserved math says there's
        # room — on macOS `available` over-counts reclaimable/compressed pages, so
        # the kernel's pressure level is the truer OOM brake. Only tightens (a
        # heavy task reaching here is already gated). The idle-escape below
        # IGNORES pressure — it must place *something* rather than stall forever.
        if pressure.get(nid, 0) >= 2:
            continue
        total = total_by_node.get(nid, avail)
        eff = min(avail, total - reserved_by_node.get(nid, 0)) - margin
        scored.append((eff, nid))

    if scored:
        best_eff, best_nid = max(scored)
        if need_mem <= 0 or need_mem <= best_eff:
            return best_nid

    if any_in_flight:
        return None  # hold — wait for a completion to free room

    # Idle-escape: nothing in flight, so place something or stall forever.
    if scored:
        return max(scored)[1]  # roomiest measured candidate, run alone
    # No measured candidate — fall back to the largest-capacity node that still
    # satisfies the custom resources (may be None only if NO node advertises a
    # required resource at all, i.e. a genuine misconfiguration).
    fallback, fallback_cap = None, -1.0
    for nid, capacity in capacity_by_node.items():
        if not _custom_fits(need_custom, capacity, reserved_custom_by_node.get(nid, {})):
            continue
        cap_mem = float(capacity.get("memory", 0))
        if cap_mem > fallback_cap:
            fallback, fallback_cap = nid, cap_mem
    return fallback


def run_graph(
    graph: Graph,
    dry_run: bool = False,
    status_fn: StatusFn | None = None,
    status_interval: float = _MIN_STATUS_UPDATE_INTERVAL_SECS,
    trace: bool = False,
    debug: bool = False,
    drain_event: "threading.Event | None" = None,
    preflight: bool = True,
    node_sort_key: "Callable[[Node, list], Any] | None" = None,
) -> tuple[Any, "RunStats"]:
    num_nodes_run: int = 0
    last_status_time: float = 0
    trace_list = []

    if preflight:
        graph = _apply_preflight(graph)

    # No-GC: in-run ephemeral deletion is disabled. cleanup_fn=None makes
    # _notify_ephemeral_registry a no-op (graph_runtime.py), so no ephemeral
    # intermediate is deleted or cache-invalidated — they persist as @node cache
    # hits (the 100GB cache is only ~3% full, not evicting). This removes the
    # recompute churn: the old cleanup deleted + invalidated an intermediate a
    # later same-call_hash producer still needed, forcing a re-run. Validated at
    # 0 recompute over 10.5h. Reclaim orphans deliberately via
    # scripts.sweep_orphans, never mid-run.
    root = GraphRuntime(graph)

    # Constants; tune per cluster/workload. 2x cores (was 4x): each in-flight
    # slot is a fresh Python worker carrying the full scientific stack (~0.3-1 GB
    # RSS each), so the slot COUNT is itself a static memory tax — 4x on a 24-core
    # box spun up 96 workers and ~half the RAM before any real work. The
    # memory-aware admission gate below (not MAX_IN_FLIGHT) is what bounds the
    # heavy ops; this just stops the worker baseline from dominating.
    MAX_IN_FLIGHT = 2 * int(ray.cluster_resources().get("CPU", 1))
    WAIT_BATCH = 16
    TIMEOUT = 0.2  # seconds

    # Per-node memory-aware admission state. The head IS the broker: run_graph is
    # the sole, single-threaded dispatcher, so these in-process dicts need no lock
    # and no per-node actor. Ledgers (persist across samples):
    #   reserved_by_node        : node_id -> summed memory estimates in flight
    #   reserved_custom_by_node : node_id -> {resource: reserved} (e.g. PixInsight)
    #   lease_by_ref            : ObjectRef -> (node_id, mem, custom), released on done
    # Snapshots (re-measured at most every MEM_SAMPLE_INTERVAL s):
    #   free_by_node/total_by_node : MEASURED psutil available/total per node —
    #                                reflects PixInsight's real out-of-process RSS
    #   capacity_by_node           : ray.nodes() Resources (memory + custom caps)
    reserved_by_node: dict[str, int] = {}
    reserved_custom_by_node: dict[str, dict[str, float]] = {}
    lease_by_ref: dict[ray.ObjectRef, tuple] = {}
    free_by_node: dict[str, int] = {}
    total_by_node: dict[str, int] = {}
    pressure_by_node: dict[str, int] = {}   # node_id -> OS pressure (1/2/4, 0 unknown)
    capacity_by_node: dict[str, dict[str, float]] = {}
    MEM_SAMPLE_INTERVAL = 2.0  # seconds between per-node free-RAM re-measurements
    last_mem_sample: float = 0.0

    # Per-node estimate memoized for the run (a retained node is reconsidered
    # every pass; recomputing its estimate — which can read image headers /
    # profile samples — each time would be wasteful).
    mem_est_cache: dict[Node, int] = {}

    def _node_mem(rt, node) -> int:
        if node not in mem_est_cache:
            mem_est_cache[node] = _estimate_node_memory(rt, node)
        return mem_est_cache[node]

    def _refresh_node_snapshot():
        """Re-measure per-node free RAM and re-read per-node capacity. A fresh
        measurement supersedes the in-pass ``free_by_node`` draw-downs (the ACTUAL
        leg re-anchors to truth); the reservation ledgers persist across it. Best
        effort — if the psutil fan-out returns nothing, capacity from
        ``ray.nodes()`` still populates so the idle-escape can place work."""
        nonlocal free_by_node, total_by_node, pressure_by_node, capacity_by_node, last_mem_sample
        from web_server.backend.ray_metrics import sample_node_memory
        meas = sample_node_memory()
        if meas:
            free_by_node = {nid: a for nid, (a, _t, _p) in meas.items()}
            total_by_node = {nid: t for nid, (_a, t, _p) in meas.items()}
            pressure_by_node = {nid: p for nid, (_a, _t, p) in meas.items()}
        capacity_by_node = {
            n["NodeID"]: dict(n.get("Resources", {}))
            for n in ray.nodes() if n.get("Alive")
        }
        last_mem_sample = time.time()

    def _maybe_refresh_snapshot():
        if time.time() - last_mem_sample >= MEM_SAMPLE_INTERVAL:
            _refresh_node_snapshot()

    # Cache of ready nodes. Re-scan the tree only when running low.
    ready_cache: list[tuple[GraphRuntime, Node]] = []
    RESCAN_THRESHOLD = MAX_IN_FLIGHT

    def _refill_cache():
        nonlocal ready_cache
        ready_cache = root.get_all_ready()
        if node_sort_key is not None and ready_cache:
            # Advisory scheduling hint: reorder ready nodes so the caller's
            # preferred ones dispatch first. Purely an ordering of the run —
            # every node still runs, and a node is only ever here once its
            # deps are done, so topological order is unaffected. Dispatch
            # pops from the tail, so larger key == sooner. The node is
            # ready, so `make_fn_args` yields its resolved call args
            # (literals + dep results) for the hint to inspect.
            def _hint(rt_node):
                rt, node = rt_node
                try:
                    resolved = rt.make_fn_args(node)
                except Exception:
                    resolved = list(node.args)
                return node_sort_key(node, resolved)
            ready_cache.sort(key=_hint)

    def _draining() -> bool:
        return bool(drain_event and drain_event.is_set())

    # No eager sample: a pass with no heavy task never samples at all (phase-1
    # metadata stays on the cheap path). The first gated task triggers a refresh
    # via `_maybe_refresh_snapshot` (last_mem_sample starts at 0 → always due).

    while root.is_active():
        if len(ready_cache) < RESCAN_THRESHOLD:
            _refill_cache()

        if debug:
            print(f"Top of loop, {len(ready_cache)} ready nodes")

        # Submit ready nodes incrementally, cap in-flight, process completions.
        pending: list[ray.ObjectRef] = []
        ref_meta: dict[ray.ObjectRef, tuple[GraphRuntime, Node]] = {}

        while ready_cache or pending:
            # Admit ready nodes up to the concurrency cap. A HEAVY task (declares
            # memory and/or a custom resource) is placed on a specific node by the
            # measured per-node gate (`_select_node`) and PINNED there; a task
            # that fits nowhere right now is *retained* — still ready — and
            # reconsidered next pass once a completion frees room. LIGHT tasks
            # (no memory estimate, no custom resource) skip the gate entirely and
            # dispatch unpinned/SPREAD as before. `_select_node` subsumes the old
            # cluster-aggregate custom-resource gate: placement is per-node.
            retained: list[tuple[GraphRuntime, Node]] = []
            while (
                ready_cache
                and not _draining()
                and len(pending) < MAX_IN_FLIGHT
            ):
                rt, node = ready_cache.pop()
                need_custom = _declared_custom_resources(node.task)
                need_mem = _node_mem(rt, node)
                gated = bool(need_custom) or need_mem >= _GATE_MIN_BYTES
                node_id = None
                if gated:
                    _maybe_refresh_snapshot()  # sample only when placing a heavy task
                    node_id = _select_node(
                        need_mem, need_custom,
                        free_by_node, total_by_node, reserved_by_node,
                        capacity_by_node, reserved_custom_by_node,
                        any_in_flight=bool(pending),
                        pressure_by_node=pressure_by_node,
                    )
                    if node_id is None:  # fits nowhere now — hold for a later pass
                        retained.append((rt, node))
                        continue
                rt.start(node)
                try:
                    ref = _submit_node(
                        rt, node, dry_run, debug,
                        node_id=node_id, memory=(need_mem or None),
                    )
                    ref_meta[ref] = (rt, node)
                    pending.append(ref)
                    if gated:  # reserve on the chosen node iff it launched
                        reserved_by_node[node_id] = (
                            reserved_by_node.get(node_id, 0) + need_mem
                        )
                        if need_custom:
                            rc = reserved_custom_by_node.setdefault(node_id, {})
                            for r, amt in need_custom.items():
                                rc[r] = rc.get(r, 0.0) + amt
                        if need_mem and node_id in free_by_node:
                            # In-pass draw-down so same-pass admits see it before
                            # the next measurement would (task hasn't allocated).
                            free_by_node[node_id] -= need_mem
                        lease_by_ref[ref] = (node_id, need_mem, need_custom)
                except AncestorsNeverRan:
                    rt.node_completed(node, NeverRun)
                except AncestorsFailed:
                    rt.node_completed(node, Skipped)
            # Return retained nodes (priority order preserved: highest ends up
            # at the tail, popped first next pass) so a freed resource is
            # reconsidered for them after the completions below.
            if retained:
                ready_cache.extend(reversed(retained))

            if not pending:
                break

            k = min(WAIT_BATCH, len(pending))
            ready_refs, still_pending = ray.wait(
                pending, num_returns=k, timeout=TIMEOUT
            )
            pending = still_pending

            for ref in ready_refs:
                rt, node = ref_meta.pop(ref)
                # Release the reservation on the ref RESOLVING (success or terminal
                # error) — held correctly across Ray retries (ref stays unresolved
                # while retrying). A leak here would only inflate `reserved` →
                # more conservative, never unsafe.
                lease = lease_by_ref.pop(ref, None)
                if lease is not None:
                    nid, mem, custom = lease
                    reserved_by_node[nid] = reserved_by_node.get(nid, 0) - mem
                    rc = reserved_custom_by_node.get(nid)
                    if rc:
                        for r, amt in custom.items():
                            rc[r] = rc.get(r, 0.0) - amt
                if trace:
                    trace_list.append(node)
                if debug:
                    obj_count = _ray_object_count()
                    dupes = _ray_object_store_duplicates()
                    dupe_str = (
                        " DUPLICATES: " + ", ".join(
                            f"{t}({h[:8]}) x{c}" for (t, h), c in dupes.items()
                        )
                    ) if dupes else ""
                    print(
                        f"Task {node.task._function.__name__} finished "
                        f"(ray_objects={obj_count}){dupe_str}"
                    )
                num_nodes_run += 1
                try:
                    result = ray.get(ref)
                    rt.node_completed(
                        node, result, _retrieve_logs(rt, node)
                    )
                except ray.exceptions.RayTaskError as e:
                    print(f"Node {node_to_string(node)} raised exception {e}")
                    rt.node_failed(
                        node, e.cause, _retrieve_logs(rt, node)
                    )

            # Refill the cache when it runs low.
            if len(ready_cache) < RESCAN_THRESHOLD:
                _refill_cache()

            # Periodic status/graph updates fire once per iteration
            # — whether ray.wait returned completions or not. Putting
            # them only inside the no-completions branch hid every
            # mid-progress state from observers when completions
            # arrived in batches.
            now = time.time()
            if now > (last_status_time + status_interval):
                if status_fn:
                    status_fn(root.get_status())
                _fire_graph_updates(root)
                last_status_time = now

        # Inner loop exited with ready_cache and pending both empty.
        # If we're draining, the in-flight set has finished naturally —
        # surface GraphCancelled so the caller can restart.
        if _draining():
            raise GraphCancelled("Graph run drained")

    assert not root.is_active()

    # Final callbacks.
    if status_fn:
        status_fn(root.get_status())
    _fire_graph_updates(root)

    _report_disk_bloat(root)

    return root.final_result, RunStats(
        num_nodes_run=num_nodes_run,
        trace=trace_list,
    )


_DISK_BLOAT_THRESHOLD_BYTES = 500 * 1024 * 1024  # 500 MB


def _report_disk_bloat(rt: "GraphRuntime") -> None:
    """Walk every completed result in the runtime tree; sum the on-disk
    size of producer outputs that are NOT ephemeral AND NOT `final=True`.
    Emit one info-level summary line if the total clears the threshold.

    The audit's B2: pure silent disk bloat — every run accretes hundreds
    of MB of intermediates that the author meant to wrap in
    `ephemeral_intermediates()` but didn't. The framework can't infer
    intent, but it can surface "here's how much you left on disk this
    run" so the author can decide whether to wrap on the next pass.

    Best-effort and never raises — telemetry, not pipeline-critical.
    """
    try:
        import structlog
        from cache.ephemeral_cleanup import _master_path

        total_bytes = 0
        n_files = 0

        def walk(rt):
            nonlocal total_bytes, n_files
            for node, result in rt.results.items():
                if not isinstance(node, Node):
                    continue
                task = node.task
                if getattr(task, "_final", False) or getattr(node, "_ephemeral", False):
                    continue
                master = _master_path(result)
                if master is None or not master.exists():
                    continue
                try:
                    total_bytes += master.stat().st_size
                    n_files += 1
                except OSError:
                    pass
            for child in rt._children.values():
                walk(child)
            for child in rt._completed_children.values():
                walk(child)

        walk(rt)
        if total_bytes >= _DISK_BLOAT_THRESHOLD_BYTES:
            structlog.get_logger().info(
                "disk-bloat: non-ephemeral, non-final producer outputs "
                f"totaled {total_bytes / 1024 / 1024:.1f} MB across "
                f"{n_files} files — consider wrapping any intermediates "
                "you don't intend to keep in `ephemeral_intermediates()`.",
            )
    except Exception:
        pass


def _fire_graph_updates(rt: GraphRuntime) -> None:
    """Fire graph_update_fn callbacks on all runtimes in the tree.

    Walks both active and completed children — a fast-completing
    subgraph can be moved to `_completed_children` between periodic
    status ticks, and we still want its graph_update_fn to fire one
    last time so its UI representation reflects the final state.
    """
    if rt.graph_update_fn is not None:
        rt.graph_update_fn(rt)
    for child in rt._children.values():
        _fire_graph_updates(child)
    for child in rt._completed_children.values():
        _fire_graph_updates(child)
