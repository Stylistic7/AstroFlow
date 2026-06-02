"""Cluster frames by hot-pixel-defect fingerprint overlap.

This module discriminates physical cameras among frames sharing the same
INSTRUME header. Same-model sensors (two ASI6200MM Pros, e.g.) produce
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

from metadata.classes import FrameHash


log = structlog.get_logger()


_DEFAULT_MIN_DEFECT = 100
_DEFAULT_O_MIN = 3
_DEFAULT_MIN_ANCHOR_SIZE = 20
_DEFAULT_MIN_AVG_SCORE = 0.5
_DEFAULT_MAX_ITERS = 15
_DEFAULT_CONSENSUS_TAU = 0.5
_DEFAULT_CONSENSUS_TOP_N = 30


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
    """
    frame_list = tuple(frame_sets.keys())
    pixel_to_col: dict[int, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    for i, h in enumerate(frame_list):
        for p in frame_sets[h]:
            c = pixel_to_col.get(p)
            if c is None:
                c = len(pixel_to_col)
                pixel_to_col[p] = c
            rows.append(i)
            cols.append(c)
    n_pixels = len(pixel_to_col)
    data = np.ones(len(rows), dtype=np.int32)
    X = csr_matrix(
        (data, (rows, cols)),
        shape=(len(frame_list), n_pixels),
        dtype=np.int32,
    )
    O = (X @ X.T).tocsr()
    O.setdiag(0)
    O.eliminate_zeros()
    # Inverse of pixel_to_col: column index -> pixel id.
    col_to_pixel = tuple(
        p for p, _ in sorted(pixel_to_col.items(), key=lambda kv: kv[1])
    )
    return X, O, frame_list, col_to_pixel


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
        ar_rows: list[int] = []
        ar_cols: list[int] = []
        ar_vals: list[float] = []
        for k, c in enumerate(anchors):
            members = np.where(labels == c)[0]
            inv = 1.0 / len(members)
            for m in members.tolist():
                ar_rows.append(k)
                ar_cols.append(m)
                ar_vals.append(inv)
        M = csr_matrix(
            (ar_vals, (ar_rows, ar_cols)),
            shape=(len(anchors), n_frames),
            dtype=np.float32,
        )
        # avg overlap of each frame with each anchor's members.
        scores = (M @ O).toarray().T  # shape (n_frames, n_anchors)
        best_k = scores.argmax(axis=1)
        max_score = scores.max(axis=1)
        new_labels = np.array(
            [anchors[k] for k in best_k.tolist()], dtype=labels.dtype
        )
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
