"""Layer 1 on hand-built graphs with known optima. No sky, no weather."""
import pytest

from planner.allocate import Edge, Need, solve


def test_two_slots_two_needs_known_optimum():
    needs = [Need("A", ((1.0, 2.0),)), Need("B", ((1.0, 1.0),))]
    edges = [Edge(0, 0, 1.0), Edge(0, 1, 1.0), Edge(1, 1, 1.0)]
    a = solve([1.0, 1.0], needs, edges)
    assert a.hours == {(0, 0): pytest.approx(1.0), (1, 1): pytest.approx(1.0)}
    assert a.objective == pytest.approx(3.0)


def test_a_need_whose_window_closes_is_served_before_one_that_can_wait():
    """OIII (X) can only use slots 0–1 (moonless); Hα (Y) any slot. Equal
    value: the solver still gives X both early slots, Y the late one."""
    needs = [Need("X", ((2.0, 1.0),)), Need("Y", ((2.0, 1.0),))]
    edges = [Edge(0, 0, 1.0), Edge(1, 0, 1.0), Edge(0, 1, 1.0), Edge(1, 1, 1.0), Edge(2, 1, 1.0)]
    a = solve([1.0, 1.0, 1.0], needs, edges)
    assert sum(a.by_need(0).values()) == pytest.approx(2.0)
    assert a.by_need(1) == {2: pytest.approx(1.0)}


def test_tiers_fill_in_order_and_cap_at_their_hours():
    needs = [Need("A", ((1.0, 2.0), (1.0, 0.5)))]
    edges = [Edge(0, 0, 1.0)]
    a = solve([3.0], needs, edges)                  # 3 h of sky, 2 h wanted in total
    assert a.hours[(0, 0)] == pytest.approx(2.0)
    assert a.objective == pytest.approx(2.5)
    assert a.slot_price[0] == pytest.approx(0.0)    # spare sky is worth nothing


def test_a_cloudy_slot_has_no_capacity_and_efficiency_scales_value():
    needs = [Need("A", ((1.0, 1.0),))]
    edges = [Edge(0, 0, 1.0), Edge(1, 0, 0.5)]
    a = solve([0.0, 1.0], needs, edges)            # slot 0: p(clear) = 0
    assert a.hours == {(1, 0): pytest.approx(1.0)}
    assert a.objective == pytest.approx(0.5)


def test_shadow_prices_name_the_contested_slot():
    needs = [Need("A", ((1.0, 2.0),)), Need("B", ((1.0, 1.0),))]
    edges = [Edge(0, 0, 1.0), Edge(0, 1, 1.0)]      # both want the only slot
    a = solve([1.0], needs, edges)
    assert a.slot_price[0] == pytest.approx(1.0)    # one more hour would earn B's 1.0
    assert a.need_price[(0, 0)] == pytest.approx(1.0)  # A: its 2.0 minus the slot's 1.0


def test_rank_takes_the_sky_first_whatever_the_values():
    """The cart's first entry (rank 0) is worth 1/h, the second 9/h; one slot
    of sky: rank 0 gets it, rank 1 the leftover (none)."""
    from planner.allocate import solve_by_rank
    needs = [Need("first", ((1.0, 1.0),)), Need("second", ((1.0, 9.0),))]
    edges = [Edge(0, 0, 1.0), Edge(0, 1, 1.0), Edge(1, 1, 1.0)]
    a = solve_by_rank([1.0, 1.0], needs, edges, ranks=[0, 1])
    assert a.hours == {(0, 0): pytest.approx(1.0), (1, 1): pytest.approx(1.0)}
    assert a.objective == pytest.approx(10.0)
    # ... but only hours it can still use: a satisfied first entry leaves the rest
    a = solve_by_rank([3.0], needs, edges[:2], ranks=[0, 1])
    assert a.hours == {(0, 0): pytest.approx(1.0), (0, 1): pytest.approx(1.0)}


def test_no_edges_solves_to_nothing():
    a = solve([1.0], [Need("A", ((1.0, 1.0),))], [])
    assert a.hours == {} and a.objective == 0.0
