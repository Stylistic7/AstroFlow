"""Tests for the fingerprint clustering pipeline."""
import numpy as np

from clustering.fingerprint import (
    anchor_reassign,
    attribute_by_consensus,
    attribute_by_locality,
    build_overlap,
    cluster_by_fingerprint,
    consensus_pixels_for_cluster,
    initial_clusters,
    SHAPE_ONLY_IDENTITY,
)


def _h(name: str):
    """Mint a fake FrameHash."""
    return name


def test_build_overlap_matrix():
    # Three frames with hand-computed overlaps:
    #   A = {1,2,3,4}
    #   B = {2,3,5}   -> A∩B = {2,3}   = 2
    #   C = {1,4,6}   -> A∩C = {1,4}   = 2; B∩C = {} = 0
    frames = {_h("A"): {1, 2, 3, 4}, _h("B"): {2, 3, 5}, _h("C"): {1, 4, 6}}
    X, O, frame_list, col_to_pixel = build_overlap(frames)
    # Row order preserves dict insertion order.
    assert frame_list == (_h("A"), _h("B"), _h("C"))
    # X has 4+3+3 = 10 nonzeros
    assert X.nnz == 10
    # O is symmetric with diagonal zeroed.
    Odense = O.toarray()
    assert Odense[0, 1] == 2 and Odense[1, 0] == 2  # A-B
    assert Odense[0, 2] == 2 and Odense[2, 0] == 2  # A-C
    assert Odense[1, 2] == 0 and Odense[2, 1] == 0  # B-C
    assert Odense[0, 0] == 0 and Odense[1, 1] == 0 and Odense[2, 2] == 0
    # col_to_pixel is the column index -> pixel id map; check the pixel
    # ids covered match the union.
    assert set(col_to_pixel) == {1, 2, 3, 4, 5, 6}


def test_initial_clusters_respects_o_min():
    # A-B overlap 2, A-C overlap 2, B-C overlap 0.
    frames = {_h("A"): {1, 2, 3, 4}, _h("B"): {2, 3, 5}, _h("C"): {1, 4, 6}}
    _, O, _, _ = build_overlap(frames)
    # At o_min=3 no edges survive -> 3 separate components.
    labels_3 = initial_clusters(O, o_min=3)
    assert len(set(labels_3.tolist())) == 3
    # At o_min=2 A bridges B and C -> 1 component.
    labels_2 = initial_clusters(O, o_min=2)
    assert len(set(labels_2.tolist())) == 1


def test_anchor_reassign_pulls_orphan_into_anchor():
    # Build a big anchor (10 frames with overlapping defects) plus one
    # orphan that strongly overlaps the anchor's members.
    common = set(range(1, 21))  # 20 shared pixels per anchor frame
    frames: dict = {}
    for i in range(10):
        # Each anchor frame: common + 5 unique pixels
        frames[_h(f"anchor{i}")] = common | {100 + 10 * i + j for j in range(5)}
    # Orphan shares most of `common` -> strong avg overlap with anchor.
    frames[_h("orphan")] = (set(list(common)[:18]) | {999})
    _, O, frame_list, _ = build_overlap(frames)
    # Manually force orphan into its own cluster.
    labels = np.arange(len(frame_list), dtype=np.int32)
    labels[:10] = 0  # all anchor frames -> cluster 0
    # Orphan starts as its own singleton (cluster 10).
    labels = anchor_reassign(
        labels, O, min_anchor_size=5, min_avg_score=0.5, max_iters=5
    )
    # Orphan should be absorbed into the anchor cluster.
    assert labels[-1] == labels[0]


def test_anchor_reassign_guard_keeps_low_signal_singletons():
    # Regression for the User-flat-into-Starbase contamination found during
    # validation. A frame with VERY few defects can chance-overlap with a
    # foreign cluster; the guard prevents random attribution.
    common_user = set(range(1, 21))
    common_starbase = set(range(1001, 1021))
    frames: dict = {}
    for i in range(20):
        frames[_h(f"user{i}")] = common_user | {100 + i}
    for i in range(20):
        f = common_starbase | {2000 + i}
        # Pixel 999 appears in only 6/20 = 30% of starbase frames by chance.
        if i < 6:
            f.add(999)
        frames[_h(f"starbase{i}")] = f
    # Orphan has 1 pixel, which is in 30% of starbase (avg overlap = 0.3,
    # below the 0.5 guard).
    frames[_h("orphan_flat")] = {999}
    _, O, frame_list, _ = build_overlap(frames)
    labels = np.zeros(len(frame_list), dtype=np.int32)
    labels[:20] = 0
    labels[20:40] = 1
    labels[40] = 2  # orphan singleton
    new_labels = anchor_reassign(
        labels, O, min_anchor_size=10, min_avg_score=0.5, max_iters=5
    )
    assert new_labels[40] == 2  # guard kept orphan as singleton

    # Sanity: without the guard, the orphan WOULD be misattributed to starbase.
    no_guard = anchor_reassign(
        labels, O, min_anchor_size=10, min_avg_score=0.0, max_iters=5
    )
    assert no_guard[40] == 1  # misattributed without the guard


def test_cluster_by_fingerprint_filters_low_defect_frames():
    # 5 high-defect frames -> qualified; 2 low-defect -> filtered out.
    common = set(range(1, 110))
    frames: dict = {}
    for i in range(5):
        frames[_h(f"good{i}")] = common | {200 + i}
    frames[_h("flat0")] = {1, 2}  # only 2 defects -> below min_defect=100
    frames[_h("flat1")] = {1}
    result = cluster_by_fingerprint(
        frames, min_defect=100, min_anchor_size=3
    )
    # Expect at least 2 buckets: 1 main cluster + 1 no-fingerprint passthrough.
    no_fp = [(frames_, c) for (frames_, c) in result if c == ()]
    qualified_buckets = [(frames_, c) for (frames_, c) in result if c != ()]
    assert len(qualified_buckets) >= 1
    assert len(no_fp) == 1
    nf_frames = set(no_fp[0][0])
    assert nf_frames == {_h("flat0"), _h("flat1")}


def test_cross_camera_clusters_disjoint():
    # Two synthetic cameras whose defects do not overlap at all -> 2
    # clusters, never merged.
    cam_a = set(range(1, 200))
    cam_b = set(range(10_000, 10_200))
    frames: dict = {}
    for i in range(20):
        frames[_h(f"a{i}")] = cam_a | {1000 + i}
    for i in range(20):
        frames[_h(f"b{i}")] = cam_b | {20_000 + i}
    result = cluster_by_fingerprint(
        frames, min_defect=10, min_anchor_size=5
    )
    # Two clusters, no contamination between A and B.
    qualified = [frames_ for (frames_, c) in result if c != ()]
    assert len(qualified) == 2
    for group in qualified:
        sources = {("a" if str(h).startswith("a") else "b") for h in group}
        assert len(sources) == 1


def test_consensus_pixels_stable_across_resamples():
    # Same camera at full N vs half N -> identical top-N consensus.
    # Build frames where 30 "real defect" pixels are ALWAYS present (freq=1.0)
    # and 70 "noise" pixels appear in only some frames. Top-30 consensus
    # picks the always-present pixels regardless of subset size, sorted by
    # pixel id ascending (tiebreak rule when freqs equal).
    rng = np.random.RandomState(7)
    real_defects = list(range(1000, 1030))  # 30 always-present
    noise_pool = list(range(2000, 2070))    # 70 random
    big_frames: dict = {}
    for i in range(100):
        noise = set(rng.choice(noise_pool, size=30, replace=False).tolist())
        big_frames[_h(f"f{i}")] = set(real_defects) | noise
    X_big, _, frame_list_big, col_to_pixel_big = build_overlap(big_frames)
    cons_big = consensus_pixels_for_cluster(
        list(range(100)), X_big, col_to_pixel_big, tau=0.5, top_n=30
    )
    # Expected: the 30 always-present pixels in id-ascending order.
    assert cons_big == tuple(real_defects)
    # Same camera at half N.
    subset = rng.choice(100, size=50, replace=False).tolist()
    small_frames = {frame_list_big[i]: big_frames[frame_list_big[i]] for i in subset}
    X_small, _, _, col_to_pixel_small = build_overlap(small_frames)
    cons_small = consensus_pixels_for_cluster(
        list(range(50)), X_small, col_to_pixel_small, tau=0.5, top_n=30
    )
    assert cons_big == cons_small


def test_cluster_by_fingerprint_folds_sub_anchor_clusters_into_no_fingerprint():
    # Real-data behavior: noise-fingerprint frames cluster as tiny remnants
    # after anchor reassignment. Those must NOT count as cameras — they go
    # into the no-fingerprint bucket. Construct one big anchor (size >= 5)
    # and one 2-frame remnant whose defects don't overlap the anchor's.
    big_common = set(range(1, 200))
    rem_common = set(range(10_000, 10_200))
    frames: dict = {}
    for i in range(10):
        frames[_h(f"anchor{i}")] = big_common | {1000 + i}
    for i in range(2):
        frames[_h(f"rem{i}")] = rem_common | {20_000 + i}
    result = cluster_by_fingerprint(
        frames, min_defect=10, min_anchor_size=5
    )
    real_clusters = [(f, c) for f, c in result if c != ()]
    no_fp = [(f, c) for f, c in result if c == ()]
    assert len(real_clusters) == 1
    assert set(real_clusters[0][0]) == {_h(f"anchor{i}") for i in range(10)}
    assert len(no_fp) == 1
    assert set(no_fp[0][0]) == {_h("rem0"), _h("rem1")}


def test_cluster_by_fingerprint_passthrough_when_no_candidates():
    # Bias frames have defect_candidates = None.
    frames: dict = {
        _h("bias0"): None,
        _h("bias1"): None,
        _h("bias2"): None,
    }
    result = cluster_by_fingerprint(frames)
    assert len(result) == 1
    frames_returned, consensus = result[0]
    assert set(frames_returned) == {_h("bias0"), _h("bias1"), _h("bias2")}
    assert consensus == ()


# ---- Pass 2: subset attribution -------------------------------------------

_CONS_A = tuple(range(1000, 1030))   # camera A's 30 signature pixels
_CONS_B = tuple(range(5000, 5030))   # camera B's 30 (disjoint from A)


def test_attribute_by_consensus_recovers_low_defect_frame():
    # A frame with too few defects to cluster, but 8 of them are camera A's
    # signature pixels -> confidently A.
    leftover = {
        _h("short_sub"): set(_CONS_A[:8]) | {77, 88},   # 8 A-hits + noise
        _h("noise"): {9000, 9001, 9002},                # 0 A/B hits
    }
    out = attribute_by_consensus(leftover, [_CONS_A, _CONS_B], min_hits=5)
    assert out == {_h("short_sub"): _CONS_A}


def test_attribute_by_consensus_below_min_hits_not_assigned():
    # Only 4 signature hits, below the min_hits=5 floor -> left unattributed.
    leftover = {_h("f"): set(_CONS_A[:4]) | {42}}
    out = attribute_by_consensus(leftover, [_CONS_A], min_hits=5)
    assert out == {}


def test_attribute_by_consensus_ambiguous_not_assigned():
    # Equal overlap with two cameras -> not unambiguous, so not assigned.
    leftover = {_h("f"): set(_CONS_A[:6]) | set(_CONS_B[:6])}
    out = attribute_by_consensus(leftover, [_CONS_A, _CONS_B], min_hits=5)
    assert out == {}


def test_attribute_by_consensus_ignores_empty_and_none():
    out = attribute_by_consensus(
        {_h("none"): None, _h("empty"): set()}, [_CONS_A], min_hits=5
    )
    assert out == {}


def test_attribute_by_consensus_no_consensuses():
    out = attribute_by_consensus({_h("f"): set(_CONS_A)}, [], min_hits=5)
    assert out == {}


# ---- Pass 3: locality / header fallback -----------------------------------

# `sensor_key` shape: (NAXIS1, NAXIS2). No camera string — that is the
# point of the key, and a tuple carrying one would not be a sensor key.
_SENSOR_A = (9576, 6388)
_SENSOR_CCD = (3072, 2048)


def test_attribute_by_locality_folder_proximity():
    # A defect-less light in a folder whose fingerprinted frames are all one
    # camera -> borrows that camera.
    out = attribute_by_locality(
        [_h("x")],
        folder_of={_h("a1"): "/n1", _h("a2"): "/n1", _h("x"): "/n1"},
        sensor_of={_h("x"): _SENSOR_A},
        attributed={_h("a1"): _CONS_A, _h("a2"): _CONS_A},
        sensor_consensuses={_SENSOR_A: [_CONS_A]},
    )
    assert out == {_h("x"): _CONS_A}


def test_attribute_by_locality_single_body_header():
    # No fingerprinted frame in x's folder, but its header maps to exactly one
    # camera library-wide -> attribute by header.
    out = attribute_by_locality(
        [_h("x")],
        folder_of={_h("a1"): "/n1", _h("x"): "/n2"},
        sensor_of={_h("x"): _SENSOR_A},
        attributed={_h("a1"): _CONS_A},
        sensor_consensuses={_SENSOR_A: [_CONS_A]},
    )
    assert out == {_h("x"): _CONS_A}


def test_attribute_by_locality_pure_ccd_gets_header_sentinel():
    # A CCD whose header never fingerprinted at all -> header-only sentinel so
    # its frames still stack together.
    out = attribute_by_locality(
        [_h("ccd0"), _h("ccd1")],
        folder_of={_h("ccd0"): "/fli", _h("ccd1"): "/fli"},
        sensor_of={_h("ccd0"): _SENSOR_CCD, _h("ccd1"): _SENSOR_CCD},
        attributed={},
        sensor_consensuses={},
    )
    assert out == {_h("ccd0"): SHAPE_ONLY_IDENTITY,
                   _h("ccd1"): SHAPE_ONLY_IDENTITY}


def test_attribute_by_locality_ambiguous_multibody_skipped():
    # Two bodies share the header and there is no folder signal -> refuse to
    # guess.
    out = attribute_by_locality(
        [_h("x")],
        folder_of={_h("x"): "/lonely"},
        sensor_of={_h("x"): _SENSOR_A},
        attributed={},
        sensor_consensuses={_SENSOR_A: [_CONS_A, _CONS_B]},
    )
    assert out == {}


def test_attribute_by_locality_folder_beats_ambiguous_header():
    # Header is ambiguous (two bodies), but the folder pins it to one -> use
    # the folder's camera.
    out = attribute_by_locality(
        [_h("x")],
        folder_of={_h("a"): "/n3", _h("x"): "/n3"},
        sensor_of={_h("x"): _SENSOR_A},
        attributed={_h("a"): _CONS_B},
        sensor_consensuses={_SENSOR_A: [_CONS_A, _CONS_B]},
    )
    assert out == {_h("x"): _CONS_B}
