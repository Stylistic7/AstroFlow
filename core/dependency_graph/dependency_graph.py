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


def _terminals(graph) -> list:
    """The nodes nothing depends on — the SAME set `GraphRuntime.__init__`
    checks for the single-terminal subgraph invariant."""
    depended: set = set()
    for deps in graph.values():
        depended.update(deps)
    return [n for n in graph if n not in depended]


def _restore_single_terminal(norm, pruned: Graph) -> Graph:
    """A dynamic subgraph funnels all its terminals into ONE no-output sentinel
    (`run_pipeline_done`) so `GraphRuntime` sees a single terminal. Preflight's
    reverse-walk seeds from unsatisfied `_final`/leaf nodes and adds only their
    PREDECESSORS — but that funnel is a pure SUCCESSOR of the nodes it joins, so
    it's never re-added. When a `final=True` master's output was reclaimed (so
    the master is unsatisfied) while its downstream consumers + the funnel are
    still on disk (satisfied), preflight keeps the master(s) but drops the
    funnel — leaving `pruned` with multiple bare terminals, which trips
    `GraphRuntime.__init__`'s single-terminal assertion.

    Restore the invariant by re-attaching the original single funnel as a join
    over the surviving terminals (same sentinel, same ignored `None` result the
    parent already expects). No-op unless preflight actually left >1 terminal
    AND the original graph had exactly one to restore."""
    if not pruned:
        return pruned
    pruned_terminals = _terminals(pruned)
    if len(pruned_terminals) <= 1:
        return pruned
    norm_terminals = _terminals(norm)
    if len(norm_terminals) != 1 or norm_terminals[0] in pruned:
        return pruned
    pruned[norm_terminals[0]] = list(pruned_terminals)
    return pruned


def _call_key(rt, node) -> "tuple | None":
    """`(op, call_hash)` of a node about to dispatch — None for a task the
    cache layer does not key (a builder) or when the key cannot be formed."""
    key_of = getattr(node.task, "call_key", None)
    if key_of is None:
        return None
    try:
        return key_of(*rt.make_fn_args(node))
    except Exception:
        return None


def _complete_if_cached(rt, node) -> bool:
    """DRIVER-SIDE HIT SHORT-CIRCUIT — the admission decision "this node
    needs no dispatch at all". Probes the node's cache entry read-only and,
    on a valid hit, completes the node inline.

    Why here and not in the remote task: the check used to ride INSIDE the
    `@ray.remote` body, so a "hit" still paid a full dispatch — and for
    `auto_memory` ops (`max_calls=1`) that means booting a fresh worker that
    imports the whole app to read one cache entry and exit. Measured
    2026-08-26: 1,799 fresh processes for 4,835 hits — hours of wall-clock
    for milliseconds of lookups.

    Deps are DONE at ready-time, so `make_fn_args` binds real results and
    the key is exactly the one the remote lookup would compute. Validity
    rules are identical too (stale-on-disk included): `cached_value` is the
    same probe the subgraph terminal-skip trusts. `log_hit=True` keeps the
    `### node_exec` telemetry stream complete — the hit is recorded by the
    cache layer itself, same as a remote hit would have been.

    Any probe failure returns False and the node dispatches normally: the
    short-circuit is an optimization, never load-bearing for correctness.
    """
    probe = getattr(node.task, "cached_value", None)
    if probe is None:
        return False
    try:
        hit, value = probe(*rt.make_fn_args(node), log_hit=True)
    except Exception:
        return False
    if not hit:
        return False
    rt.node_completed(node, value)
    return True


def _apply_preflight(graph: Graph, is_subgraph: bool = False) -> Graph:
    """Predict every node's call_hash statically, check disk for
    `final=True` / terminal outputs, and return a filtered Graph that
    contains only the nodes that actually need to run. Falls out of
    content-addressing — see `dependency_graph/preflight.py` for the
    proof sketch. Errors are swallowed: preflight is an optimization,
    never load-bearing for correctness.

    `is_subgraph=True` (called from `node_completed` for a dynamic subgraph)
    restores the single-terminal invariant the reverse-walk can break — see
    `_restore_single_terminal`. The top-level graph has no such invariant, so
    it's off by default.
    """
    # DISABLED 2026-08-08 (the GC/preflight removal decision): dispatch the
    # full graph; the cached.remote layer is the single skip mechanism — a
    # satisfied node is a fast cache hit, not a prediction. This also retires
    # the prediction layer as a witness: every "hash flap" artifact this week
    # (version-log prediction entries, probe expectations, snapshots) lived on
    # the prediction side while the runtime's artifacts stayed consistent.
    return graph
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
    # SATISFIED-BOUNDARY HYDRATION. The reverse walk stops at satisfied deps
    # (they're a floor — see nodes_to_dispatch), so a to_run consumer can have
    # a dep outside to_run. Its edge can't just be dropped: `make_fn_args`
    # binds dep results positionally via `_dep_positions`, so a missing edge
    # mis-binds every arg after it. And it can't be dispatched as-is: running
    # it would demand ITS deps' values, un-flooring the walk (that closure
    # requirement is exactly what re-ran 533 reclaimed-ephemeral placements
    # per pass, live 2026-08-08). Instead each satisfied boundary dep is
    # REPLACED, in position, by a `compute_image_metadata(path)` node: the
    # sidecar written at the dep's completion reconstructs its ImageMetadata
    # result without touching pixels, and the node's own cache makes later
    # passes instant. Membership is proof of satisfaction: the walk examined
    # every dep of every to_run node and left it out ONLY when satisfied.
    #
    # A satisfied dep with NO filename_fn (a no-output sentinel) has no value
    # to hydrate — those edges are ordering-only by construction (data-bearing
    # no-output ops bottom out in never-satisfied source nodes, so they can't
    # BE satisfied) and are safely dropped, matching the old edge-drop.
    from dependency_graph.classes import Node
    from dependency_graph.preflight import expected_output_path
    from metadata.image_metadata import compute_image_metadata

    hydration: dict = {}

    def _hydration_node(dep):
        if dep not in hydration:
            path = expected_output_path(dep, predicted, norm)
            hydration[dep] = (
                Node(compute_image_metadata, path) if path is not None else None
            )
        return hydration[dep]

    pruned_map: dict = {}
    for n in to_run:
        deps = []
        for d in norm[n]:
            if d in to_run:
                deps.append(d)
            else:
                sub = _hydration_node(d)
                if sub is not None:
                    deps.append(sub)
        pruned_map[n] = deps
    for sub in hydration.values():
        if sub is not None:
            pruned_map[sub] = []
    pruned = Graph(
        pruned_map,
        graph_update_fn=getattr(graph, "graph_update_fn", None),
        node_status_fn=getattr(graph, "node_status_fn", None),
    )
    # Carry the N-1 supersession map through pruning (same Node objects, so
    # surviving keys still match). Pruned-away satisfied taps already
    # superseded their N-1 the run they landed.
    incoming_versions = getattr(graph, "version_supersede", None)
    if incoming_versions:
        pruned.version_supersede = incoming_versions
    if is_subgraph:
        pruned = _restore_single_terminal(norm, pruned)
    return pruned


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
    defeating run_graph's dispatch order."""
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
    need_cpus: float = 0.0,
) -> "str | None":
    """Pick a node that can run a task needing ``need_mem`` bytes and
    ``need_custom`` resources RIGHT NOW, or ``None`` to hold it for a later pass.

    Candidate = a MEASURED node (present in ``free_by_node``) that has every
    declared custom resource free (``capacity - reserved >= amt``). This single
    per-node check subsumes the old cluster-aggregate custom-resource gate — and
    naturally keeps head-only resources (``fastest_storage``/``diskcache``) off
    workers, since only the head's capacity carries them.

    Memory admission for a candidate ``n`` uses the two-leg min:
        effective_free(n) = min(free_by_node[n], ceiling(n) - reserved_by_node[n]) - margin
        ceiling(n) = min(total_by_node[n], capacity_by_node[n]["memory"])
    * ACTUAL leg (measured psutil free) sees PixInsight's out-of-process RSS and
      anything else Ray never reserved.
    * RESERVED leg (ceiling - reserved) covers the lag before a just-launched
      task has allocated and shows up in the measurement. The ceiling is the
      SMALLER of physical RAM and the node's Ray ``memory`` capacity (``ray
      start --memory``, 0.7 x RAM): the reservation we hand Ray is admitted
      against the latter, so a lease that fits RAM but not Ray's ledger is
      not placed — Ray holds it, pending, while our ledger shows it running
      (24-panel merges, 2026-09-16: 43 GB admitted against 68.7 GB of RAM on
      a 48.1 GB Ray node, "running" with no worker for an hour).
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

    # A placed task is PINNED (NodeAffinity, soft=False): Ray cannot move it,
    # so a node whose TOTAL CPUs are below the request can never run it, and
    # Ray fails it as unschedulable — which aborted and restarted the whole
    # graph (2026-10-08: 12-CPU Siril integrations pinned to the 10-CPU MBP).
    # Free CPUs stay Ray's business; only feasibility is checked here.
    def _cpus_fit(nid) -> bool:
        return float(capacity_by_node.get(nid, {}).get("CPU", float("inf"))) >= need_cpus

    # UNMEASURED nodes are still candidates, on the RESERVED leg alone
    # (capacity − reserved, no psutil truth): before this, a node whose
    # probe timed out was dropped from candidacy entirely, which — combined
    # with cold probes needing warm workers — made idleness self-sustaining
    # (the 2026-08-19 MBP starvation; see state/node_sentinel.py). One
    # conservative admission beats permanent exile; the sentinel actors make
    # the unmeasured case rare, this makes it survivable. Loud, because the
    # exile used to be silent across 500+ one-sided placements.
    for n_id, caps in capacity_by_node.items():
        if n_id in free_by_node or "memory" not in caps:
            continue
        cap = int(caps["memory"])
        free_by_node = {**free_by_node, n_id: cap}
        total_by_node = {**total_by_node, n_id: cap}
        print(f"### _select_node: node {str(n_id)[:8]} UNMEASURED — "
              f"admitting on reserved-ledger leg only")
    scored = []  # (effective_free, node_id)
    for nid, avail in free_by_node.items():
        if not _cpus_fit(nid):
            continue
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
        ceiling = total_by_node.get(nid, avail)
        ray_cap = capacity_by_node.get(nid, {}).get("memory")
        if ray_cap:
            ceiling = min(ceiling, int(ray_cap))
        eff = min(avail, ceiling - reserved_by_node.get(nid, 0)) - margin
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
        if not _cpus_fit(nid):
            continue
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
) -> tuple[Any, "RunStats"]:
    num_nodes_run: int = 0
    last_status_time: float = 0
    trace_list = []

    if preflight:
        graph = _apply_preflight(graph)

    # Every image persists by default (the 2026-08-08 pure-world decision).
    # The one exception is opt-in and local: a dynamic subgraph whose author
    # built it with `Graph(..., delete_intermediates=True)` deletes its own
    # non-terminal images when it concludes — see `Graph.delete_intermediates`
    # and `GraphRuntime._delete_intermediates`. No global registry, no
    # refcounts, nothing to wire up here.
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
    # The dispatch list; admission pops from the TAIL. Work enters only as
    # readiness EVENTS (`drain_newly_ready`), each batch appended in reverse
    # so its first-traced node is popped first, and every later batch sits
    # above everything older. So a completed node's dependents run next:
    # a started chain (stack → SPFC → merge → BXT → SXT → place) is
    # followed to the point where it blocks on a merge before untouched
    # work begins — depth-first, with no policy table and no domain
    # knowledge. Unplaceable nodes are retained in order (below), so a
    # freed slot still finds the most recently advanced node that fits it.
    # Replaces the rescan-below-a-threshold cache (2026-09-15): with a
    # scarce resource busy, retained nodes kept that cache above its
    # threshold and everything that became ready later waited hours.
    ready_cache: list[tuple[GraphRuntime, Node]] = []

    def _pull_ready() -> None:
        ready_cache.extend(reversed(root.drain_newly_ready()))

    def _draining() -> bool:
        return bool(drain_event and drain_event.is_set())

    # No eager sample: a pass with no heavy task never samples at all (phase-1
    # metadata stays on the cheap path). The first gated task triggers a refresh
    # via `_maybe_refresh_snapshot` (last_mem_sample starts at 0 → always due).

    while root.is_active():
        _pull_ready()
        if not ready_cache:
            # Active root, nothing queued: a readiness event was missed.
            # Rescan — rare — and the started() guard makes a duplicate
            # entry harmless.
            ready_cache.extend(reversed(root.get_all_ready()))

        if debug:
            print(f"Top of loop, {len(ready_cache)} ready nodes")

        # Submit ready nodes incrementally, cap in-flight, process completions.
        pending: list[ray.ObjectRef] = []
        # Every (runtime, node) a ref completes — more than one when a second
        # runtime asked for the SAME call while it was in flight (see
        # `in_flight`); all of them see the one result.
        ref_meta: dict[ray.ObjectRef, list[tuple[GraphRuntime, Node]]] = {}
        # Cross-runtime in-flight dedup: `(op, call_hash)` -> the ref computing
        # it. Two cart pipelines for one mosaic each own an identical node;
        # a runtime dedups within itself and the @node cache catches a repeat
        # only once the first call has LANDED — so boot 91 ran three WBPP
        # masters twice, one per Mac (~75 min of PixInsight). Keyed exactly as
        # the cache is, so anything that would become a cache hit later
        # becomes a shared ref now.
        in_flight: dict[tuple, ray.ObjectRef] = {}

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
                if rt.started(node):
                    continue
                if not dry_run and _complete_if_cached(rt, node):
                    _pull_ready()  # its dependents may be hits too: chase them now
                    continue
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
                        need_cpus=float(getattr(node.task, "_ray_num_cpus", 0) or 0),
                    )
                    if node_id is None:  # fits nowhere now — hold for a later pass
                        # FAIL FAST on an UNSATISFIABLE demand: a declared
                        # custom resource with zero capacity on every live
                        # node can never fit — retaining it hot-spins this
                        # loop at 100% CPU forever (the 2026-08-06 finding:
                        # gpu_slots ops livelocked every exhaustive
                        # full-pipeline test for weeks because the test
                        # cluster never advertised the resource; the hang
                        # read as "tests got slow"). "Fits nowhere NOW" is
                        # retryable; "fits nowhere EVER" is a config error
                        # and must be loud.
                        impossible = [
                            r for r in need_custom
                            if all(
                                cap.get(r, 0.0) <= 0.0
                                for cap in capacity_by_node.values()
                            )
                        ]
                        if impossible:
                            raise RuntimeError(
                                f"unsatisfiable resource demand: node "
                                f"{getattr(node, 'name', node)!r} declares "
                                f"{impossible} but no live Ray node advertises "
                                f"any capacity for it/them (capacities: "
                                f"{ {n: c for n, c in capacity_by_node.items()} })"
                            )
                        retained.append((rt, node))
                        continue
                rt.start(node, placed_on=node_id)
                try:
                    key = _call_key(rt, node) if not dry_run else None
                    if key is not None and key in in_flight:
                        ref_meta[in_flight[key]].append((rt, node))
                        continue  # ride the in-flight computation
                    ref = _submit_node(
                        rt, node, dry_run, debug,
                        node_id=node_id, memory=(need_mem or None),
                    )
                    ref_meta[ref] = [(rt, node)]
                    if key is not None:
                        in_flight[key] = ref
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
                    # Name the corpse: which resolved args were NeverRun/
                    # Skipped sentinels. A node whose GRAPH deps all completed
                    # but whose ARGS resolve to sentinels is the trace-
                    # divergence signature (see _report_pass_end: blocked_on=
                    # READY), and without this line it dies silently.
                    try:
                        _args = rt.make_fn_args(node)
                        _bad = [
                            (i, str(node.args[i])[:100])
                            for i, x in enumerate(_args)
                            if x in [NeverRun, Skipped] and i < len(node.args)
                        ]
                        print(
                            f"### ancestors_never_ran "
                            f"node={node_to_string(node)[:120]} "
                            f"sentinel_args={_bad}"
                        )
                    except Exception:
                        pass
                    rt.node_completed(node, NeverRun)
                    _pull_ready()
                except AncestorsFailed:
                    rt.node_completed(node, Skipped)
                    _pull_ready()
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
                completions = ref_meta.pop(ref)
                rt, node = completions[0]
                for k, r in list(in_flight.items()):
                    if r is ref:
                        del in_flight[k]
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
                except ray.exceptions.RayTaskError as e:
                    print(f"Node {node_to_string(node)} raised exception {e}")
                    for rt, node in completions:
                        rt.node_failed(node, e.cause, _retrieve_logs(rt, node))
                else:
                    for rt, node in completions:
                        rt.node_completed(node, result, _retrieve_logs(rt, node))

            # Everything the completions just unblocked goes on top.
            _pull_ready()

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
            # Report the frontier on the drain exit too — a cancelled pass is
            # exactly the one whose leftover work needs naming. (Crash exits
            # like WorkerCrashedError still skip it; the retry pass reports.)
            _report_pass_end(root)
            raise GraphCancelled("Graph run drained")

    assert not root.is_active()

    # Final callbacks.
    if status_fn:
        status_fn(root.get_status())
    _fire_graph_updates(root)

    _report_disk_bloat(root)
    _report_pass_end(root)

    return root.final_result, RunStats(
        num_nodes_run=num_nodes_run,
        trace=trace_list,
    )



def _report_pass_end(rt: "GraphRuntime") -> None:
    """Name the frontier a pass ends on. One compact line per
    (op, blocker) group for every node that never ran this pass:
    `blocked_on=READY` means every dep completed yet the node was never
    dispatched (a dispatch/spawn bug — the smoking gun); otherwise the
    first never-run dep is the blocker. Passes that keep ending "N tasks
    short" without building their terminals (live 2026-08-08: the color
    composites never entering any pass) are undiagnosable without this —
    the UI's NO_STATE_RUNTIME can't distinguish queued from forgotten.
    Observability only; never raises."""
    try:
        from collections import Counter
        from cache.types import NeverRun
        from dependency_graph.classes import Skipped
        from dependency_graph.preflight import _op_name

        def op_of(n):
            try:
                return _op_name(n)
            except Exception:
                return type(n).__name__

        groups: Counter = Counter()
        sample: dict = {}
        for node in rt._graph:
            if not (rt.results.get(node, NeverRun) == NeverRun):
                continue
            # A dep that FAILED (exception result) or was Skipped in a
            # failure cascade is just as blocking as one that never ran —
            # without naming it, a cascade prints as blocked_on=READY and
            # masquerades as a dispatch bug (misread live 2026-09-01: two
            # plate-solve failures read as the 08-08 spawn bug).
            blocker, blocker_kind = None, ""
            for d in rt._graph.get(node, []) or []:
                res = rt.results.get(d, NeverRun)
                if res == NeverRun:
                    blocker = d
                    break
                if res == Skipped or isinstance(res, Exception):
                    blocker, blocker_kind = d, (
                        " (SKIPPED)" if res == Skipped else " (FAILED)")
                    break
            key = (op_of(node),
                   op_of(blocker) + blocker_kind if blocker is not None
                   else "READY")
            groups[key] += 1
            sample.setdefault(key, node)
        # A node that RAN AND FAILED is not NeverRun, so it never entered
        # `groups`. Reporting only never-run nodes meant a pass that died in a
        # failure cascade printed the same "every node ran" line as a clean
        # one — which misread as success twice on 2026-09-09 (438 permission
        # errors on a worker with dead mounts, then a self-inflicted churn
        # loop). Count failures separately and say so.
        failed: Counter = Counter()
        fail_sample: dict = {}
        for node in rt._graph:
            res = rt.results.get(node, NeverRun)
            if isinstance(res, Exception):
                k = (op_of(node), type(res).__name__)
                failed[k] += 1
                fail_sample.setdefault(k, res)
        if failed:
            nf = sum(failed.values())
            print(f"### pass_end: {nf} node(s) FAILED; by (op, error):")
            for (op, err), cnt in failed.most_common(20):
                print(f"### pass_end failed op={op} n={cnt} error={err}: "
                      f"{str(fail_sample[(op, err)])[:160]}")
        if not groups:
            print("### pass_end: every node in the graph ran"
                  + (f" ({sum(failed.values())} FAILED)" if failed else ""))
            return
        total = sum(groups.values())
        print(f"### pass_end: {total} node(s) never ran; frontier by (op, blocker):")
        for (op, blk), cnt in groups.most_common(20):
            print(
                f"### pass_end never_ran op={op} n={cnt} blocked_on={blk} "
                f"sample={str(sample[(op, blk)])[:100]}"
            )
    except Exception as e:
        print(f"### pass_end report failed: {type(e).__name__}: {e}")


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
        from cache.reclaim import _master_path

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
