"""Cluster frames by hot-pixel-defect fingerprint overlap.

This module discriminates physical cameras among frames sharing the same
SENSOR GEOMETRY. Same-model sensors (two ASI6200MM Pros, e.g.) produce
distinct hot-pixel defect sets — `extract_defects` exposes the top-N hottest
per frame as `ImageMetadata.defect_candidates`. We cluster frames by how
much their defect sets overlap; one cluster = one physical sensor.

The high-level entry point is `cluster_by_fingerprint(frame_sets)`. It
returns a list of (frames_in_cluster, canonical_consensus_pixels) tuples
plus a no-fingerprint cluster for frames whose `defect_candidates` are
absent or too sparse to participate.

Validated empirically on a 16,819-frame real library (two ASI6200MM Pros,
one user + one third-party): produces 2 clean clusters with 0 cross-source
contamination at the default parameters.

PIPELINE — sparse-matrix throughout for tractability on 10k-frame inputs:

  1. Build X (frames x pixels, binary): X[f, p] = 1 iff frame f lists p
     as a defect candidate.
  2. O = X @ X.T (with diagonal zeroed): O[i, j] is the count of pixels
     that frames i and j both list — pairwise defect-set OVERLAP.
  3. Initial connected components on the graph of edges where O[i,j] >= o_min.
     This is single linkage: ONE edge welds two components, and no later
     stage splits a component again, so `o_min` must sit far above what
     two unrelated sensors share by chance across the WHOLE library's
     cross pairs, not just one pair. See `_DEFAULT_O_MIN`.
  4. Anchor reassignment: iteratively pull each frame into the anchor
     cluster (size >= min_anchor_size) whose members it overlaps most
     with on average; guard `min_avg_score` keeps low-signal frames out
     of any anchor so they don't get randomly attributed.
  5. Consensus pixels per cluster: pixels in >= tau fraction of cluster
     members, sorted by frequency (descending) then pixel id (ascending),
     truncated to top_n. Hash of this canonical tuple is the camera identity.

WHY ANCHOR REASSIGNMENT, NOT EM-STYLE CONSENSUS REFINEMENT
A small cluster's "consensus" is dominated by its own frames — for a
singleton cluster the consensus IS exactly that frame's pixels. So
argmax over |frame ∩ consensus| always lets the small cluster win
reassignment against itself. Anchor reassignment scores frames against
cluster MEMBERS directly via the O matrix, with no derived consensus
in the loop, so there's no self-reference.

WHY PER-RUN HASHING WOULD FAIL
A "shortcut" that hashed each Run's top-N pixels directly was tested on
real data: one physical ASI6200MM Pro produced ~120 distinct hashes
across ~120 Run-proxy directories, because top-N rank order drifts with
scene content / exposure / temperature even when the underlying defect
SET is stable. Global clustering then canonical-top-N over the FULL
cluster pool is what makes the hash stable.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import structlog
from beartype import beartype
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from hashing import hash_list_int
from metadata.classes import FrameHash


log = structlog.get_logger()


class FingerprintMap(dict[FrameHash, tuple[int, ...]]):
    """Frame-hash → canonical camera-consensus pixel tuple.

    Built once globally over ALL fingerprintable light frames so that
    same-camera frames spread across many pointings still aggregate into
    one cluster. Frames absent from the map have no confident camera
    attribution (low-defect, noise fingerprint, or bias).

    Hashable so @node caches keyed on it stay coherent.
    """

    def __hash__(self):  # type: ignore[override]
        return hash_list_int([
            (h, c) for h, c in sorted(self.items())
        ])

    def __str__(self):
        return f"FingerprintMap({len(self)})"


_DEFAULT_MIN_DEFECT = 100
# Measured on the real library (2026-10-22, two ASI6200MM Pro bodies): frames
# of one body share 59+ pixels within a season and 13+ across years and
# telescopes; frames of DIFFERENT bodies share at most 3, once in 17,600
# pairs. At the old value of 3 that one chance pair fused both bodies into a
# single camera id through single linkage, and every stack of the second
# body inherited the first body's darks. 20 is an order of magnitude above
# chance and below the weakest same-body link that matters; a frame linked
# more weakly than that reaches its body through anchor reassignment or
# `attribute_by_consensus` instead of through an edge.
_DEFAULT_O_MIN = 20
_DEFAULT_MIN_ANCHOR_SIZE = 20
_DEFAULT_MIN_AVG_SCORE = 0.5
_DEFAULT_MAX_ITERS = 15
_DEFAULT_CONSENSUS_TAU = 0.5
_DEFAULT_CONSENSUS_TOP_N = 30
# Subset-attribution: a leftover frame is assigned to an already-established
# camera when it shares at least this many of that camera's 30 consensus
# (signature) pixels. Camera consensuses are disjoint and consensus pixels are
# the most-reliable hot pixels, so a genuine frame hits many while a noise
# frame hits ~0 — see the sweep in project_fingerprint_attribution_loss. 5 is
# a wide safety margin (chance of 5 coincidental hits on 30 pixels out of a
# 60 MP sensor is effectively zero).
_DEFAULT_SUBSET_MIN_HITS = 5
# Sentinel "consensus" for frames identified by header/locality alone (no
# usable defect fingerprint — e.g. CCD frames, whose column/cluster defects
# extract_defects can't see). Non-empty so `_camera_id_hash_from` yields a
# real (shape-derived) id; -1 is not a valid flat pixel index, so it never
# collides with a real consensus. CAVEAT: two CCDs of identical geometry
# now share this sentinel, where INSTRUME used to separate them. The
# principled fix is to teach `extract_defects` to see column/cluster
# defects so CCDs fingerprint like CMOS does — not to trust the name.
SHAPE_ONLY_IDENTITY: tuple[int, ...] = (-1,)


def sensor_key(header: dict) -> tuple:
    """The (sensor-shape, pixel-pitch) pre-group key — PHYSICAL only.

    The single definition of "same sensor, before fingerprinting" shared by
    light-stack identification and calibration-stack identification, so the
    two cannot drift apart.

    NAXIS1/NAXIS2 are the only header cards used, because they are the only
    ones that are really pixels: FITS requires them to describe the data
    array, and astropy will not hand you a frame whose NAXIS disagrees with
    its own shape. Everything else here would be a promise, not a
    measurement.

    `INSTRUME` is deliberately NOT here. It is free text the capture software
    chooses; it is identical across every body of a model, and it can change
    under the user mid-session — on 2026-02-16 a reconnecting driver renamed
    one camera from "ASI Camera (1)" to "ZWO ASI2600MM Duo", partitioning one
    sensor's frames into two pre-groups that no amount of downstream
    clustering could rejoin.

    `XPIXSZ`/`YPIXSZ` are not here either, for the same reason at lower
    volume: pixel pitch in microns is metadata the driver asserts, nothing
    validates it, and a session written with a wrong pitch would fork one
    sensor exactly the way the rename did. It also buys almost nothing —
    two bodies sharing a pixel COUNT are separated by their defects, and
    differing optics are already separated by `image_scale` clustering.
    """
    return (header.get("NAXIS1"), header.get("NAXIS2"))


@beartype
def build_overlap(
    frame_sets: dict[FrameHash, dict[int, float] | set[int]],
) -> tuple[csr_matrix, csr_matrix, tuple[FrameHash, ...], tuple[int, ...]]:
    """Build the frame x pixel matrix X and overlap matrix O = X @ X.T.

    `frame_sets[h]` may be a `dict[int, float]` (pixel -> strength, the
    real ImageMetadata.defect_candidates shape) or a `set[int]` (tests).
    Only pixel KEYS are used — strength is ignored here.

    Returns:
        X: (n_frames, n_pixels) binary
        O: (n_frames, n_frames) integer overlap counts, diagonal zeroed
        frame_list: row-index order of X/O
        col_to_pixel: column-index order of X (column k corresponds to pixel id col_to_pixel[k])

    Memory note: for ~16k frames × ~1000 defects each, the COO triple is
    ~16M entries. Holding `rows`/`cols` as Python lists of int costs
    ~28 bytes per entry (~900 MB); pre-allocated numpy arrays are 4 bytes
    each (~128 MB). The order-of-magnitude difference matters on
    multi-camera production libraries.
    """
    frame_list = tuple(frame_sets.keys())
    # Allocate the COO triple up front using numpy. Lower bound on size
    # is total defect count across all frames.
    total = sum(len(frame_sets[h]) for h in frame_list)
    rows = np.empty(total, dtype=np.int32)
    cols = np.empty(total, dtype=np.int32)
    pixel_to_col: dict[int, int] = {}
    idx = 0
    for i, h in enumerate(frame_list):
        for p in frame_sets[h]:
            c = pixel_to_col.get(p)
            if c is None:
                c = len(pixel_to_col)
                pixel_to_col[p] = c
            rows[idx] = i
            cols[idx] = c
            idx += 1
    n_pixels = len(pixel_to_col)
    # csr_matrix accepts ones-implicit binary via data=np.ones.
    data = np.ones(total, dtype=np.int32)
    X = csr_matrix(
        (data, (rows, cols)),
        shape=(len(frame_list), n_pixels),
        dtype=np.int32,
    )
    # Build col_to_pixel before releasing pixel_to_col so we don't hold
    # both a dict and a tuple of comparable size.
    col_to_pixel_arr = np.empty(n_pixels, dtype=np.int64)
    for p, c in pixel_to_col.items():
        col_to_pixel_arr[c] = p
    del pixel_to_col
    del rows, cols, data
    O = (X @ X.T).tocsr()
    O.setdiag(0)
    O.eliminate_zeros()
    return X, O, frame_list, tuple(col_to_pixel_arr.tolist())


@beartype
def initial_clusters(O: csr_matrix, *, o_min: int = _DEFAULT_O_MIN) -> np.ndarray:
    """Connected components on the overlap graph thresholded at O >= o_min.

    Returns a label array of length n_frames; labels are arbitrary ints.
    """
    if o_min <= 0:
        raise ValueError(f"o_min must be positive, got {o_min}")
    Ot = O.copy()
    Ot.data = Ot.data * (Ot.data >= o_min).astype(np.int32)
    Ot.eliminate_zeros()
    _, labels = connected_components(Ot, directed=False)
    return labels


@beartype
def anchor_reassign(
    labels: np.ndarray,
    O: csr_matrix,
    *,
    min_anchor_size: int = _DEFAULT_MIN_ANCHOR_SIZE,
    min_avg_score: float = _DEFAULT_MIN_AVG_SCORE,
    max_iters: int = _DEFAULT_MAX_ITERS,
) -> np.ndarray:
    """Iteratively pull each frame into the anchor cluster (size
    >= min_anchor_size) whose members it overlaps most with on average.

    The score is computed directly from O (against cluster MEMBERS), so a
    small cluster cannot win reassignment against itself via a self-
    referential consensus — see the module docstring.

    Frames whose max avg-overlap with any anchor is below `min_avg_score`
    keep their current label. This prevents low-signal frames (flats,
    noise-fingerprints) from being randomly attributed.
    """
    labels = labels.copy()
    n_frames = O.shape[0]
    for it in range(max_iters):
        sizes = Counter(labels.tolist())
        anchors = sorted(c for c, n in sizes.items() if n >= min_anchor_size)
        if not anchors:
            return labels
        # Build M (n_anchors, n_frames) where M[k, f] = 1/|anchor_k| if frame f is in anchor k.
        # Allocate numpy arrays up front instead of Python lists of int/float.
        ar_rows = np.empty(n_frames, dtype=np.int32)
        ar_cols = np.empty(n_frames, dtype=np.int32)
        ar_vals = np.empty(n_frames, dtype=np.float32)
        idx = 0
        for k, c in enumerate(anchors):
            members = np.where(labels == c)[0]
            inv = 1.0 / len(members)
            end = idx + len(members)
            ar_rows[idx:end] = k
            ar_cols[idx:end] = members
            ar_vals[idx:end] = inv
            idx = end
        M = csr_matrix(
            (ar_vals[:idx], (ar_rows[:idx], ar_cols[:idx])),
            shape=(len(anchors), n_frames),
            dtype=np.float32,
        )
        del ar_rows, ar_cols, ar_vals
        # avg overlap of each frame with each anchor's members.
        scores = (M @ O).toarray().T  # shape (n_frames, n_anchors)
        del M
        best_k = scores.argmax(axis=1)
        max_score = scores.max(axis=1)
        del scores
        anchors_arr = np.array(anchors, dtype=labels.dtype)
        new_labels = anchors_arr[best_k]
        guard = max_score < min_avg_score
        new_labels[guard] = labels[guard]
        if np.array_equal(new_labels, labels):
            return new_labels
        labels = new_labels
    return labels


@beartype
def consensus_pixels_for_cluster(
    cluster_frame_idx: Sequence[int],
    X: csr_matrix,
    col_to_pixel: Sequence[int],
    *,
    tau: float = _DEFAULT_CONSENSUS_TAU,
    top_n: int = _DEFAULT_CONSENSUS_TOP_N,
) -> tuple[int, ...]:
    """Return the cluster's canonical top-N consensus pixels.

    A pixel is "consensus" if it appears in at least `tau` fraction of the
    cluster's frames. Tied pixels are broken by ascending pixel id, so the
    returned tuple is fully deterministic — hashable as a stable camera id.
    """
    if not cluster_frame_idx:
        return ()
    sub = X[list(cluster_frame_idx)]
    col_sum = np.asarray(sub.sum(axis=0)).flatten()
    thresh = tau * len(cluster_frame_idx)
    qual_cols = np.where(col_sum >= thresh)[0]
    if qual_cols.size == 0:
        return ()
    # Sort by frequency descending, then by pixel id ascending.
    pixel_ids = np.array([col_to_pixel[c] for c in qual_cols.tolist()])
    freqs = col_sum[qual_cols]
    # numpy lexsort: last key is primary; primary = -freqs (descending),
    # secondary = pixel_ids (ascending).
    order = np.lexsort((pixel_ids, -freqs))
    chosen = pixel_ids[order][:top_n]
    return tuple(int(p) for p in chosen.tolist())


@beartype
def cluster_by_fingerprint(
    frame_sets: dict[FrameHash, dict[int, float] | set[int] | None],
    *,
    min_defect: int = _DEFAULT_MIN_DEFECT,
    o_min: int = _DEFAULT_O_MIN,
    min_anchor_size: int = _DEFAULT_MIN_ANCHOR_SIZE,
    min_avg_score: float = _DEFAULT_MIN_AVG_SCORE,
    consensus_tau: float = _DEFAULT_CONSENSUS_TAU,
    consensus_top_n: int = _DEFAULT_CONSENSUS_TOP_N,
) -> list[tuple[tuple[FrameHash, ...], tuple[int, ...]]]:
    """Cluster frames by defect-set overlap; return one tuple per cluster.

    Each returned tuple is `(frame_hashes_in_cluster, canonical_consensus_pixels)`.
    Hash the consensus tuple to derive a stable camera id for the cluster.

    Frames with `defect_candidates is None` or fewer than `min_defect`
    pixels are bucketed into a single "no-fingerprint" cluster with an
    empty consensus — the caller maps that to `camera_id_hash = None`.

    The default parameters were validated on a 16,819-frame real library;
    see the module docstring.
    """
    qualified: dict[FrameHash, dict[int, float] | set[int]] = {}
    no_fingerprint: list[FrameHash] = []
    for h, defects in frame_sets.items():
        if defects is None or len(defects) < min_defect:
            no_fingerprint.append(h)
        else:
            qualified[h] = defects

    out: list[tuple[tuple[FrameHash, ...], tuple[int, ...]]] = []

    if qualified:
        X, O, frame_list, col_to_pixel = build_overlap(qualified)
        labels = initial_clusters(O, o_min=o_min)
        labels = anchor_reassign(
            labels,
            O,
            min_anchor_size=min_anchor_size,
            min_avg_score=min_avg_score,
        )
        by_cluster: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels.tolist()):
            by_cluster.setdefault(int(lab), []).append(idx)
        # Anchor-sized clusters become real cameras. Sub-anchor remnants
        # (noise-fingerprint frames that couldn't be confidently attributed)
        # fold into the no-fingerprint bucket — they shouldn't be treated
        # as a real camera.
        for member_idx in by_cluster.values():
            if len(member_idx) < min_anchor_size:
                no_fingerprint.extend(frame_list[i] for i in member_idx)
                continue
            consensus = consensus_pixels_for_cluster(
                member_idx, X, col_to_pixel,
                tau=consensus_tau, top_n=consensus_top_n,
            )
            frames_in_cluster = tuple(frame_list[i] for i in member_idx)
            out.append((frames_in_cluster, consensus))

    if no_fingerprint:
        out.append((tuple(no_fingerprint), ()))

    log.info(
        "cluster_by_fingerprint",
        qualified=len(qualified),
        no_fingerprint=len(no_fingerprint),
        n_clusters=len(out),
        cluster_sizes=sorted(
            (len(frames) for frames, _ in out), reverse=True
        )[:10],
    )
    return out


@beartype
def attribute_by_consensus(
    leftover: dict[FrameHash, dict[int, float] | set[int] | None],
    consensuses: Sequence[Sequence[int]],
    *,
    min_hits: int = _DEFAULT_SUBSET_MIN_HITS,
) -> dict[FrameHash, tuple[int, ...]]:
    """Recover low-defect frames by matching them to an ALREADY-established
    camera consensus — the second pass after `cluster_by_fingerprint`.

    A frame below `min_defect` never enters clustering (too few pixels to
    anchor or connect), yet its few hot pixels are still that camera's real
    hot pixels. Here we assign each such frame to the consensus it shares at
    least `min_hits` pixels with, provided that match is unambiguous (strictly
    more hits than any other consensus). This NEVER invents a camera — it only
    attributes to consensuses discovered by the confident clustering pass, so
    the `min_defect` guard against spurious NEW cameras is untouched. Camera
    consensuses are disjoint (different silicon → different hot pixels), so a
    genuine frame matches exactly one and a noise frame matches none.

    Returns `{frame: consensus}` for the frames it could confidently place.
    """
    cons_sets = [(tuple(c), set(c)) for c in consensuses if c]
    out: dict[FrameHash, tuple[int, ...]] = {}
    if not cons_sets:
        return out
    for h, defs in leftover.items():
        if not defs:
            continue
        pix = set(defs)
        best_hits = second = 0
        best: tuple[int, ...] | None = None
        for cons_t, cons_s in cons_sets:
            hits = len(pix & cons_s)
            if hits > best_hits:
                second, best_hits, best = best_hits, hits, cons_t
            elif hits > second:
                second = hits
        if best is not None and best_hits >= min_hits and best_hits > second:
            out[h] = best
    return out


@beartype
def attribute_by_locality(
    unattributed: Sequence[FrameHash],
    *,
    folder_of: dict[FrameHash, str | None],
    sensor_of: dict[FrameHash, tuple],
    attributed: dict[FrameHash, tuple[int, ...]],
    sensor_consensuses: dict[tuple, Sequence[tuple[int, ...]]],
) -> dict[FrameHash, tuple[int, ...]]:
    """Weak, non-fingerprint fallback for frames with no usable defect signal
    (e.g. CCD frames — `extract_defects` targets CMOS single-pixel hot pixels
    and finds nothing in a CCD's column/cluster defects). Applied ONLY after
    defect clustering + `attribute_by_consensus`, and only to frames still
    unplaced. The cascade, most-precise first:

      1. capture-folder proximity: if the frame's folder contains fingerprinted
         frames of exactly one camera, borrow that consensus (a shared night =
         the same camera; the folder disambiguates multi-body rigs);
      2. single-body geometry: else if the frame's sensor-shape maps to
         exactly one real consensus library-wide, use it;
      3. shape-only identity: else if that geometry never fingerprinted at all
         (a pure-CCD camera), tag with `SHAPE_ONLY_IDENTITY` so same-sensor
         frames still stack together (their camera_id_hash stays distinct per
         shape);
      4. otherwise leave it unattributed (ambiguous: multiple bodies share the
         geometry and there is no locality signal to choose between them).

    Returns `{frame: consensus-or-sentinel}` for the frames it could place.
    """
    folder_cams: dict[str, Counter] = {}
    for g, cons in attributed.items():
        f = folder_of.get(g)
        if f is not None:
            folder_cams.setdefault(f, Counter())[cons] += 1

    out: dict[FrameHash, tuple[int, ...]] = {}
    for h in unattributed:
        cams = folder_cams.get(folder_of.get(h))
        if cams is not None and len(cams) == 1:
            out[h] = next(iter(cams))
            continue
        uniq = set(sensor_consensuses.get(sensor_of.get(h), ()))
        if len(uniq) == 1:
            out[h] = next(iter(uniq))
        elif len(uniq) == 0:
            out[h] = SHAPE_ONLY_IDENTITY
        # else: multiple bodies share this geometry, no folder signal — skip.
    return out
