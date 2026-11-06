"""The shared level solver: the mosaic merger's mean-zero gauge and the
combiner's pinned-reference gauge. Pure numpy."""
import numpy as np
import pytest

from mosaics.merging.level_solve import solve_level_offsets


def test_mean_zero_gauge_recovers_pairwise_offsets():
    # Three images in a row with true skies 0, +5, −3: differences A−B = −5, B−C = +8, A−C = +3.
    pairs = [(0, 1, -5.0, 400), (1, 2, 8.0, 400), (0, 2, 3.0, 100)]
    o = solve_level_offsets(3, pairs)
    assert o.sum() == pytest.approx(0.0, abs=1e-9)
    assert (o - o[0]) == pytest.approx([0.0, 5.0, -3.0], abs=1e-9)


def test_pinned_reference_keeps_the_gauge_and_isolated_images_stay():
    # Images 0..2 connected, image 3 alone, image 4 alone but pinned at 2.5.
    pairs = [(0, 1, -5.0, 400), (1, 2, 8.0, 400)]
    o = solve_level_offsets(5, pairs, fixed={0: 0.0, 4: 2.5})
    assert o == pytest.approx([0.0, 5.0, -3.0, 0.0, 2.5], abs=1e-6)
    # A pin elsewhere in the group moves the whole group, not its differences.
    o2 = solve_level_offsets(5, pairs, fixed={2: 10.0})
    assert o2[:3] == pytest.approx([13.0, 18.0, 10.0], abs=1e-6)
    # Inconsistent loops are reconciled by weight: the heavier pair wins more.
    loop = [(0, 1, 1.0, 10000), (1, 2, 1.0, 10000), (0, 2, 0.0, 100)]
    o3 = solve_level_offsets(3, loop, fixed={0: 0.0})
    assert o3[2] == pytest.approx(-2.0, abs=0.05)
    assert solve_level_offsets(2, [], fixed={1: 7.0}) == pytest.approx([0.0, 7.0])
