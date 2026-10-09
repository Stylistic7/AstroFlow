"""run_graph places a memory/resource-gated task on a node that actually has
room for it — measured per-node free RAM (PixInsight's out-of-process footprint
included) + a per-node reservation ledger — then pins it there. These pin the
two pure pieces the dispatch loop depends on: the per-node estimate lookup and
the node-selection decision (`_select_node`). The live pin/placement + the 2 s
psutil fan-out are exercised on the real cluster.
"""
from dependency_graph.dependency_graph import (
    _estimate_node_memory,
    _select_node,
    _custom_fits,
)

_GB = 1024 ** 3


# --- node selection: the two-leg memory decision --------------------------------

def _sel(need_mem, *, free, total=None, reserved=None, capacity=None,
         reserved_custom=None, need_custom=None, any_in_flight=True,
         pressure=None):
    """Thin wrapper with small-integer 'units' so the arithmetic is obvious.
    `free`/`total`/`reserved` are {node: int}; capacity/reserved_custom/pressure
    optional."""
    total = total or {n: 100 for n in free}
    return _select_node(
        need_mem, need_custom or {},
        free, total, reserved or {},
        capacity if capacity is not None else {n: {} for n in free},
        reserved_custom or {},
        any_in_flight=any_in_flight, pressure_by_node=pressure,
    )


def test_picks_the_roomiest_node():
    # eff_A = min(50,100) = 50 ; eff_B = min(30,100) = 30 ; both fit 20 → A
    assert _sel(20, free={"A": 50, "B": 30}) == "A"


def test_none_when_nothing_fits_and_work_is_in_flight():
    # every node's effective_free < need, and something is running to free room
    assert _sel(50, free={"A": 30, "B": 20}, any_in_flight=True) is None


def test_actual_leg_blocks_low_free():
    # eff = min(15,100) = 15 ; need 20 > 15 → hold
    assert _sel(20, free={"A": 15}, any_in_flight=True) is None
    # eff = min(50,100) = 50 ; need 20 ≤ 50 → place
    assert _sel(20, free={"A": 50}, any_in_flight=True) == "A"


def test_reserved_leg_blocks_despite_stale_high_free():
    # The lag/herd guard: psutil still reads 90 free (just-launched tasks haven't
    # allocated), but reservations (85) say no room. min(90,100-85) = 15.
    assert _sel(20, free={"A": 90}, reserved={"A": 85}, any_in_flight=True) is None


def test_reserved_leg_is_capped_at_the_nodes_ray_memory():
    # RAM says 100 total / 90 free, but Ray's `memory` capacity for the node
    # is 70 (ray start --memory): a 40-unit lease with 40 already reserved
    # would be held by Ray (70-40=30 < 40) while our ledger called it running.
    # eff = min(90, min(100, 70) - 40) = 30 → hold.
    assert _sel(40, free={"A": 90}, reserved={"A": 40},
                capacity={"A": {"memory": 70}}, any_in_flight=True) is None
    # Without the Ray cap the same lease is admitted: min(90, 100-40) = 60.
    assert _sel(40, free={"A": 90}, reserved={"A": 40}, any_in_flight=True) == "A"


def test_boundary_is_inclusive():
    # eff = min(40,100) = 40
    assert _sel(40, free={"A": 40}, any_in_flight=True) == "A"   # need == free
    assert _sel(41, free={"A": 40}, any_in_flight=True) is None  # one over


def test_reserve_before_next_prevents_the_herd():
    # Two sequential admissions on ONE node, with the caller's in-pass decrement
    # (free-=need, reserved+=need) applied between them.
    free = {"A": 40}
    assert _select_node(30, {}, free, {"A": 100}, {"A": 0}, {"A": {}}, {},
                        any_in_flight=True) == "A"
    # caller reserves the grant before considering the next task:
    free["A"] -= 30
    assert _select_node(30, {}, free, {"A": 100}, {"A": 30}, {"A": {}}, {},
                        any_in_flight=True) is None  # second denied


# --- the OS memory-pressure veto ------------------------------------------------

def test_pressure_veto_blocks_yellow_and_red():
    # Plenty of free RAM + no reservations, but the node is under memory pressure
    # (yellow=2, red=4) → hold, don't add to the fire.
    for level in (2, 4):
        assert _sel(20, free={"A": 90}, pressure={"A": level}, any_in_flight=True) is None


def test_pressure_green_and_unknown_do_not_block():
    assert _sel(20, free={"A": 50}, pressure={"A": 1}, any_in_flight=True) == "A"  # green
    assert _sel(20, free={"A": 50}, pressure={"A": 0}, any_in_flight=True) == "A"  # unknown
    assert _sel(20, free={"A": 50}, pressure={}, any_in_flight=True) == "A"        # absent


def test_pressure_prefers_the_green_node():
    # A is roomier but yellow; B is green → B wins despite less free RAM.
    assert _sel(20, free={"A": 90, "B": 40}, pressure={"A": 2, "B": 1},
                any_in_flight=True) == "B"


def test_pressure_idle_escape_places_despite_red():
    # Everything red AND nothing in flight → the idle-escape ignores pressure and
    # places something rather than stall the run forever.
    assert _sel(20, free={"A": 90}, pressure={"A": 4}, any_in_flight=False) == "A"


# --- the custom-resource dimension (subsumes the old cluster-aggregate gate) -----

def test_custom_resource_slot_gates_placement():
    cap = {"A": {"PixInsight": 1}, "B": {"PixInsight": 1}}
    # A's PI slot is already reserved even though its memory is roomier → pick B.
    got = _sel(10, free={"A": 90, "B": 40}, capacity=cap,
               reserved_custom={"A": {"PixInsight": 1}},
               need_custom={"PixInsight": 1}, any_in_flight=True)
    assert got == "B"
    # If the only PI-capable node's slot is taken → hold (something in flight).
    got = _sel(10, free={"A": 90}, capacity={"A": {"PixInsight": 1}},
               reserved_custom={"A": {"PixInsight": 1}},
               need_custom={"PixInsight": 1}, any_in_flight=True)
    assert got is None


def test_head_only_resource_never_placed_on_a_worker():
    # Only the head advertises fastest_storage; the worker is roomier but ineligible.
    cap = {"head": {"fastest_storage": 1}, "worker": {}}
    got = _sel(10, free={"head": 30, "worker": 90}, capacity=cap,
               need_custom={"fastest_storage": 1}, any_in_flight=True)
    assert got == "head"


# --- idle-escape (never stall the run) ------------------------------------------

def test_idle_escape_places_roomiest_when_nothing_in_flight():
    # Nothing fits AND nothing is running to free room → must place something.
    assert _sel(50, free={"A": 30, "B": 20}, any_in_flight=False) == "A"


def test_idle_escape_falls_back_to_capacity_when_unmeasured():
    # No psutil measurement at all (fan-out failed), but capacity is known:
    # unmeasured nodes are admitted on their reserved leg alone. A demand no
    # node's capacity holds waits while work is in flight; with nothing in
    # flight the idle-escape takes the largest-capacity node.
    cap = {"head": {"memory": 64, "PixInsight": 1}, "worker": {"memory": 32, "PixInsight": 1}}
    assert _select_node(10, {"PixInsight": 1}, {}, {}, {}, cap, {},
                        any_in_flight=True) == "head"   # fits; 64 > 32
    assert _select_node(70, {"PixInsight": 1}, {}, {}, {}, cap, {},
                        any_in_flight=True) is None
    assert _select_node(70, {"PixInsight": 1}, {}, {}, {}, cap, {},
                        any_in_flight=False) == "head"


# --- _custom_fits ---------------------------------------------------------------

def test_custom_fits():
    assert _custom_fits({"PixInsight": 1}, {"PixInsight": 1}, {})
    assert not _custom_fits({"PixInsight": 1}, {"PixInsight": 1}, {"PixInsight": 1})
    assert _custom_fits({}, {}, {})                       # no demand → always fits
    assert not _custom_fits({"PixInsight": 1}, {}, {})    # resource absent → 0 avail
    assert _custom_fits({"diskcache": 0.001}, {"diskcache": 1.0}, {})


# --- the per-node estimate lookup -----------------------------------------------

class _Task:
    def __init__(self, estimator, function="fn-sentinel"):
        self._auto_memory_estimator = estimator
        self._function = function


class _Node:
    def __init__(self, task):
        self.task = task


class _RT:
    """Stand-in GraphRuntime: make_fn_args returns the node's resolved args."""
    def __init__(self, args):
        self._args = args

    def make_fn_args(self, node):
        return self._args


def test_estimate_uses_the_auto_memory_estimator_with_resolved_args():
    seen = {}

    def estimator(func, args, kwargs):
        seen["call"] = (func, args, kwargs)
        return 7 * _GB

    node = _Node(_Task(estimator, function="the-func"))
    assert _estimate_node_memory(_RT(["a", "b"]), node) == 7 * _GB
    # Called exactly like Ray's auto_memory path: (func, tuple(args), {}).
    assert seen["call"] == ("the-func", ("a", "b"), {})


def test_estimate_is_zero_for_non_auto_memory_nodes():
    assert _estimate_node_memory(_RT([]), _Node(_Task(estimator=None))) == 0


def test_estimate_degrades_to_zero_on_estimator_error():
    def boom(func, args, kwargs):
        raise RuntimeError("profile store unreachable")

    # Never raises out of the gate — a broken estimate must not stall the run.
    assert _estimate_node_memory(_RT([]), _Node(_Task(boom))) == 0


def test_estimate_is_zero_when_task_missing():
    class _NoTask:
        pass
    assert _estimate_node_memory(_RT([]), _NoTask()) == 0


def test_a_node_with_too_few_cpus_is_never_chosen():
    """A placed task is pinned; a node whose TOTAL CPUs are below the request
    can never run it (Ray: unschedulable, which restarted the whole graph,
    2026-10-08). Roomier in memory or not, it is skipped — even by the
    idle-escape."""
    from dependency_graph.dependency_graph import _select_node
    capacity = {"MBP": {"CPU": 10.0}, "Studio": {"CPU": 24.0}}
    assert _select_node(20, {}, {"MBP": 90, "Studio": 50}, {"MBP": 100, "Studio": 100}, {},
                        capacity, {}, any_in_flight=True, need_cpus=12) == "Studio"
    assert _select_node(20, {}, {"MBP": 90}, {"MBP": 100}, {}, {"MBP": {"CPU": 10.0}}, {},
                        any_in_flight=False, need_cpus=12) is None
