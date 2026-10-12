"""Layer 1: how many expected hours of each need each slot of sky produces.

A transportation problem with diminishing returns, posed as a linear program
and solved by scipy's HiGHS — never a hand-written search (Stylistic7,
2026-10-12: integer and LP problems go to library solvers).

    maximise    Σ  value[need, tier] · efficiency[edge] · x[edge, tier]
    subject to  Σ  x over a slot's edges and tiers   ≤ capacity[slot]
                Σ  x over a need's edges, per tier   ≤ hours[need, tier]
                x ≥ 0

`x` is EXPECTED hours: a slot's capacity is its length times p(clear), so an
uncertain night sells less and the plan hedges across nights. Tiers fill in
order of value on their own. The duals of the two constraint families are
the shadow prices the explanations read: what one more clear hour in a slot
is worth, and how much a need still wants an hour.

Pure: no context, no sky, no weather — those build the inputs
(`planner.horizon`, `weather.probability`).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class Need:
    key: str
    tiers: tuple[tuple[float, float], ...]   # (hours, value per effective hour), spent in order


@dataclass(frozen=True)
class Edge:
    slot: int
    need: int
    efficiency: float                        # 0..1: the altitude × moon factor of that hour


@dataclass(frozen=True)
class Allocation:
    hours: dict[tuple[int, int], float]      # (slot, need) -> expected hours, all tiers
    slot_price: tuple[float, ...]            # per slot: the value of one more hour of capacity
    need_price: dict[tuple[int, int], float]  # (need, tier) -> the value of one more hour wanted
    objective: float

    def by_slot(self, slot: int) -> dict[int, float]:
        return {n: h for (s, n), h in self.hours.items() if s == slot}

    def by_need(self, need: int) -> dict[int, float]:
        return {s: h for (s, n), h in self.hours.items() if n == need}


def solve(capacity: Sequence[float], needs: Sequence[Need], edges: Sequence[Edge]) -> Allocation:
    cap = np.asarray(capacity, dtype=float)
    cols: list[tuple[int, int]] = []          # variable -> (edge index, tier index)
    cost: list[float] = []
    for ei, e in enumerate(edges):
        for ti, (hours, value) in enumerate(needs[e.need].tiers):
            if hours > 0 and value > 0 and e.efficiency > 0:
                cols.append((ei, ti))
                cost.append(-value * e.efficiency)
    if not cols:
        return Allocation({}, tuple(0.0 for _ in cap), {}, 0.0)

    tier_rows = {(n, t): i for i, (n, t) in enumerate(
        (n, t) for n, need in enumerate(needs) for t in range(len(need.tiers)))}
    n_slots = len(cap)
    rows, col_ix = [], []
    for j, (ei, ti) in enumerate(cols):
        rows += [edges[ei].slot, n_slots + tier_rows[(edges[ei].need, ti)]]
        col_ix += [j, j]
    a_ub = coo_matrix((np.ones(len(rows)), (rows, col_ix)),
                      shape=(n_slots + len(tier_rows), len(cols))).tocsr()
    b_ub = np.concatenate([cap, [needs[n].tiers[t][0] for (n, t) in tier_rows]])

    res = linprog(np.asarray(cost), A_ub=a_ub, b_ub=b_ub, bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"allocation did not solve: {res.message}")

    hours: dict[tuple[int, int], float] = {}
    for j, x in enumerate(res.x):
        if x > 1e-9:
            e = edges[cols[j][0]]
            hours[(e.slot, e.need)] = hours.get((e.slot, e.need), 0.0) + float(x)
    prices = -np.asarray(res.ineqlin.marginals)      # a ≤ row's dual, as a value per unit
    return Allocation(
        hours=hours,
        slot_price=tuple(float(p) for p in prices[:n_slots]),
        need_price={k: float(prices[n_slots + i]) for k, i in tier_rows.items()},
        objective=float(-res.fun),
    )
