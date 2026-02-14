from __future__ import annotations

import math
import operator
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def angular_spread(angles: list[float]) -> float:
    if not angles:
        return 0.0
    twopi = 2.0 * math.pi
    a = sorted((x % twopi) for x in angles)
    n = len(a)
    aa = a + [x + twopi for x in a]
    return min(aa[i + n - 1] - aa[i] for i in range(n))


def _greedy_line_clusters(
    indices: list[int], xs: list[float], eps: float
) -> list[list[int]]:
    # indices must correspond to xs in increasing xs order (line, not circular).
    out: list[list[int]] = []
    i = 0
    n = len(indices)
    while i < n:
        j = i
        start = xs[indices[i]]
        # extend as far as possible while span <= eps
        while j + 1 < n and xs[indices[j + 1]] - start <= eps:
            j += 1
        out.append(indices[i : j + 1])
        i = j + 1
    return out


def cluster_angles(
    items: list[T],
    eps: float,
    *,
    key: Callable[[T], float] = operator.itemgetter(1),
) -> list[list[T]]:
    """
    Deterministic clustering of circular angles.
    Produces clusters whose circular spread <= eps, using the minimum number of clusters.
    Tie-break: fewer clusters, then lexicographically smallest cluster start angle, then earliest start index.
    """
    if not items:
        return []

    twopi = 2.0 * math.pi
    xs = [float(key(it)) % twopi for it in items]
    n = len(items)

    # Order indices by angle.
    order = sorted(range(n), key=xs.__getitem__)

    # Duplicate order to allow wrap handling after a cut.
    order2 = order + order
    xs2 = xs  # xs indexed by original indices; wrap handled by adding 2π to the "unwrapped" view below.

    best: list[list[int]] | None = None
    best_key: tuple[int, float, int] | None = None

    # Try each possible cut (start at order[s], treat sequence of length n as a line).
    for s in range(n):
        seq = order2[s : s + n]

        # Build an "unwrapped" increasing view by adding 2π when crossing the cut.
        base = xs[seq[0]]
        unwrapped = [0.0] * n
        unwrapped[0] = base
        for i in range(1, n):
            a = xs[seq[i]]
            if a < base:
                a += twopi
            base = a
            unwrapped[i] = a

        # Greedy line clustering on this unwrapped line.
        # Use local indices [0..n-1] but store original item indices.
        local = list(range(n))
        # Need xs-like access by local index, so we cluster locals and then map.
        out_local: list[list[int]] = []
        i = 0
        while i < n:
            j = i
            start = unwrapped[i]
            while j + 1 < n and unwrapped[j + 1] - start <= eps:
                j += 1
            out_local.append(local[i : j + 1])
            i = j + 1

        clusters = [[seq[i] for i in c] for c in out_local]

        # Deterministic tie-break.
        num = len(clusters)
        start_angle = xs[seq[0]]
        tie = (num, start_angle, s)
        if best is None or tie < best_key:
            best = clusters
            best_key = tie

    assert best is not None

    # Convert to item clusters and sanity-check spread.
    item_clusters = [[items[i] for i in idxs] for idxs in best]
    for c in item_clusters:
        if angular_spread([float(key(it)) for it in c]) > eps + 1e-15:
            raise AssertionError(
                "internal error: produced a cluster exceeding eps"
            )
    return item_clusters
