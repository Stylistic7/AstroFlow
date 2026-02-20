from __future__ import annotations

import numpy as np
import operator
from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def angular_spread_mod_pi(angles: list[float]) -> float:
    angles = np.mod(angles, np.pi)
    angles = np.sort(angles)
    n = len(angles)
    angles = np.concatenate([angles, angles + np.pi])
    return min(angles[i + n - 1] - angles[i] for i in range(n))


def angular_mean_mod_pi(angles: Sequence[float]) -> float:
    if not angles:
        raise ValueError("angles must be non-empty")

    # Double angles → compute 2π-mean → halve result.
    s = sum(np.sin(2 * a) for a in angles)
    c = sum(np.cos(2 * a) for a in angles)

    if s == 0.0 and c == 0.0:
        raise ValueError("angular mean is undefined for these angles")

    mean = 0.5 * np.arctan2(s, c)
    return mean % np.pi


def cluster_angles_mod_pi(
    items: list[T],
    eps: float,
    *,
    key: Callable[[T], float] = operator.itemgetter(1),
) -> list[list[T]]:
    if not items:
        return []

    period = np.pi
    xs = [float(key(it)) % period for it in items]
    n = len(items)

    order = sorted(range(n), key=xs.__getitem__)
    order2 = order + order

    best: list[list[int]] | None = None
    best_key: tuple[int, float, int] | None = None

    for s in range(n):
        seq = order2[s : s + n]

        base = xs[seq[0]]
        unwrapped = [0.0] * n
        unwrapped[0] = base
        for i in range(1, n):
            a = xs[seq[i]]
            if a < base:
                a += period
            base = a
            unwrapped[i] = a

        out: list[list[int]] = []
        i = 0
        while i < n:
            j = i
            start = unwrapped[i]
            while j + 1 < n and unwrapped[j + 1] - start <= eps:
                j += 1
            out.append([seq[t] for t in range(i, j + 1)])
            i = j + 1

        num = len(out)
        start_angle = xs[seq[0]]
        tie = (num, start_angle, s)
        if best is None or tie < best_key:
            best = out
            best_key = tie

    assert best is not None

    clusters = [[items[i] for i in idxs] for idxs in best]
    for idxs in best:
        if angular_spread_mod_pi([xs[i] for i in idxs]) > eps + 1e-15:
            raise AssertionError(
                "internal error: produced a cluster exceeding eps"
            )
    return clusters
