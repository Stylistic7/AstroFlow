"""Unit tests for angular clustering and statistics."""
import math
import pytest
import numpy as np

from clustering.angles import (
    angular_spread_mod_pi,
    angular_mean_mod_pi,
    cluster_angles_mod_pi,
)


# -- angular_spread_mod_pi --

def test_spread_identical_angles():
    assert angular_spread_mod_pi([0.5, 0.5, 0.5]) == 0.0


def test_spread_small_range():
    spread = angular_spread_mod_pi([0.1, 0.2, 0.3])
    assert abs(spread - 0.2) < 1e-10


def test_spread_wrapping():
    # Angles near 0 and near pi should have small spread (they're close mod pi).
    spread = angular_spread_mod_pi([0.01, math.pi - 0.01])
    assert spread < 0.03


def test_spread_maximally_spaced():
    spread = angular_spread_mod_pi([0.0, math.pi / 2])
    assert abs(spread - math.pi / 2) < 1e-10


# -- angular_mean_mod_pi --

def test_mean_single_angle():
    assert abs(angular_mean_mod_pi([1.0]) - 1.0) < 1e-10


def test_mean_identical_angles():
    mean = angular_mean_mod_pi([0.5, 0.5, 0.5])
    assert abs(mean - 0.5) < 1e-10


def test_mean_symmetric():
    # Mean of 0.4 and 0.6 should be 0.5.
    mean = angular_mean_mod_pi([0.4, 0.6])
    assert abs(mean - 0.5) < 1e-10


def test_mean_wrapping():
    # Angles near 0 and near pi should average near 0 (or pi), not pi/2.
    mean = angular_mean_mod_pi([0.01, math.pi - 0.01])
    assert mean < 0.02 or mean > math.pi - 0.02


def test_mean_empty_raises():
    with pytest.raises(ValueError):
        angular_mean_mod_pi([])


def test_mean_result_in_range():
    for angle in [0.0, 0.5, 1.0, 2.0, 3.0]:
        mean = angular_mean_mod_pi([angle])
        assert 0 <= mean < math.pi


# -- cluster_angles_mod_pi --

def test_cluster_empty():
    assert cluster_angles_mod_pi([], eps=0.1) == []


def test_cluster_single_group():
    items = [("a", 0.5), ("b", 0.6), ("c", 0.55)]
    clusters = cluster_angles_mod_pi(items, eps=0.2)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_cluster_two_groups():
    items = [("a", 0.1), ("b", 0.15), ("c", 1.5), ("d", 1.55)]
    clusters = cluster_angles_mod_pi(items, eps=0.1)
    assert len(clusters) == 2
    ids_per_cluster = [sorted(x[0] for x in c) for c in clusters]
    assert sorted(ids_per_cluster) == [["a", "b"], ["c", "d"]]


def test_cluster_wrapping():
    # Items near 0 and near pi should cluster together.
    items = [("a", 0.02), ("b", math.pi - 0.02), ("c", 1.5)]
    clusters = cluster_angles_mod_pi(items, eps=0.1)
    # a and b should be in the same cluster (close mod pi).
    for c in clusters:
        ids = [x[0] for x in c]
        if "a" in ids:
            assert "b" in ids
