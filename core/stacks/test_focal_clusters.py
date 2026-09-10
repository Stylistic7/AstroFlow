"""`_focal_clusters` groups FOCALLEN by optical train, via the SHARED clusterer.

It used to be a hand-rolled copy of `cluster_continuous` that differed in one
detail: it compared each value to the cluster's FIRST element instead of the
PREVIOUS one, so it was not single-link and its output depended on where a
cluster happened to start. Real FOCALLEN is a continuum, not discrete levels —
one 350 mm rig in Stylistic7's library reports 346, 347, 348, 349, 350, 351, 352 —
and single-link is what holds such a run together.

Verified 2026-09-09 against the real 32,687-frame library: the switch produces
byte-identical group membership on all 76 star flat groups, so no rebuild.
"""
import math

import pytest

from clustering.continuous import cluster_continuous
from stacks.star_flat import _focal_clusters


def _groups(values):
    """Cluster contents as comparable value-sets."""
    return sorted(
        sorted(values[i] for i in idx) for idx in _focal_clusters(values)
    )


def test_a_continuum_within_tolerance_stays_one_group():
    """Consecutive 1 mm steps are ~0.3% apart, far inside the 2% tolerance."""
    vals = [346, 347, 348, 349, 350, 351, 352]
    assert _groups(vals) == [vals]


def test_a_gap_beyond_tolerance_splits():
    # 541 -> 561 is 3.7%, outside the 2% optical-train tolerance
    assert _groups([540, 541, 541, 561]) == [[540, 541, 541], [561]]


def test_clusters_are_single_link_not_anchored_on_the_first_element():
    """The actual behavioural difference. Chained 1.5% steps are each within
    tolerance of their PREDECESSOR, so they are one optical train; anchoring
    on the cluster's first element would cut the chain at the 2% mark."""
    vals = [1000.0, 1015.0, 1030.0, 1045.0]     # each +1.5%, total +4.5%
    assert _groups(vals) == [vals]


def test_every_input_lands_in_exactly_one_cluster():
    vals = [350, 2062, 351, 540, 1234, 350, 561, 2063]
    idx = _focal_clusters(vals)
    flat = sorted(i for cl in idx for i in cl)
    assert flat == list(range(len(vals))), "indices lost or duplicated"


def test_indices_map_back_to_their_own_values():
    vals = [2062.0, 350.0, 351.0, 2063.0]
    for cl in _focal_clusters(vals):
        picked = [vals[i] for i in cl]
        assert max(picked) <= 1.02 * min(picked) or len(picked) == 1


def test_empty_and_single():
    assert _focal_clusters([]) == []
    assert _focal_clusters([350.0]) == [[0]]


def test_it_is_the_shared_clusterer_not_a_private_copy():
    """Stylistic7's standing rule: use the existing clustering algorithms, fix
    them if buggy, don't scatter new ones. This pins that `_focal_clusters`
    is a thin adapter over `cluster_continuous` (log space, so the 2%
    RELATIVE tolerance becomes an absolute epsilon)."""
    vals = [346.0, 347.0, 350.0, 351.0, 561.0, 2062.0, 2063.0]
    shared = [
        [i for i, _ in cl]
        for cl in cluster_continuous(list(enumerate(vals)),
                                     epsilon=math.log(1.02),
                                     key=lambda t: math.log(t[1]))
    ]
    assert _focal_clusters(vals) == shared
