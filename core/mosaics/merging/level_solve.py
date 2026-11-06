"""The level solve shared by the mosaic merger (merging/oss.py) and the
all-sky combiner (allsky/reduce.py): per-image additive sky offsets from
the differences measured in overlaps, as ONE exact weighted least squares.
Not Montage's iterative relaxation: on measured pairs it left about half
the correction unconverged (2026-08-31)."""
from __future__ import annotations

import numpy as np


def solve_level_offsets(n: int, pairs: list[tuple], fixed: dict[int, float] | None = None) -> np.ndarray:
    """Offsets `o` (length n) with `o_i − o_j ≈ d_ij` for every pair
    `(i, j, d_ij, npix)`, each weighted by √npix; closed loops in the
    adjacency force closure, so chain drift cannot accumulate.

    The gauge is the one freedom the pairs cannot see. `fixed` pins chosen
    images at given offsets (one per connected group keeps the gauge put
    between runs); without it the mean offset is zero. An image in no pair
    keeps its pinned value, or zero."""
    o = np.zeros(n)
    for i, v in (fixed or {}).items():
        o[i] = v
    if n < 2 or not pairs:
        return o
    weights = [np.sqrt(npix) for _, _, _, npix in pairs]
    pins = list((fixed or {}).items())
    A = np.zeros((len(pairs) + max(len(pins), 1), n))
    b = np.zeros(A.shape[0])
    for r, ((i, j, dij, _), w) in enumerate(zip(pairs, weights)):
        A[r, i] = w
        A[r, j] = -w
        b[r] = w * dij
    if pins:
        heavy = 1e4 * max(weights)              # a pin outweighs every pair that touches it
        for r, (i, v) in enumerate(pins, len(pairs)):
            A[r, i] = heavy
            b[r] = heavy * v
    else:
        A[-1, :] = 1.0                          # gauge: mean offset zero
    solved, *_ = np.linalg.lstsq(A, b, rcond=None)
    touched = np.zeros(n, bool)
    for i, j, *_ in pairs:
        touched[i] = touched[j] = True
    o[touched] = solved[touched]
    return o
