import ray
import os
import structlog
from pathlib import Path
from beartype import beartype

from file_io.find_image_files import find_image_files
from file_io.generic import read_header, get_content_hash, read_image_geometry, read_image
from file_io.master_integration import read_master_integration_seconds
from calibration.sensor_fingerprint import extract_defects
import metadata.image_types as image_types
from metadata.classes import DefectTable, ImageMetadata, MetadataDict
from metadata import star_photometry
import plate_solving
from dependency_graph import Graph
from context import context
from decorators.decorators import node, measure_performance
from cache.context_ref import ContextRef
from decisions.settings import DevelopmentSettings

log = structlog.get_logger()


def _listing_rels(listing) -> list:
    """The relative paths in a library listing (entries are
    `(rel, size, mtime)` tuples; tolerate a bare-string listing)."""
    return [e[0] if isinstance(e, (tuple, list)) else e for e in listing]



# TODO: I need to associate dark and flat frames *before* computing thumbnails.

# Stack levels of refinement:
#   Computed after reading FITS headers.
#     Middle frame in the run, dark and flat-corrected, autostretched.
#   Computed after plate-solving:
#     Best frame in the run, based on star count, dark and flat-corrected, autostretched.
#   Computed after stacking:
#     Stack master, autostretched.
#   After external edits (optional)
#     Stack master, edited.

# Target levels of refinement:
#   Same as above for stacks. All of the stacks must achieve a level before
#   the target achieves that level.
#
#   Whenever the target level changes, re-run the target post-processing.
#
# Mosaic levels of refinement:
#   Same mechanics as targets. Stacks should be continuously refined,
#   which will occassionaly trigger target refinement, which will
#   occassionally trigger mosaic refinement.

# Alllllll of this could be represented a single static compute graph. Nodes
# like "first frame, autostretch" will already have been run and cached, and can
# be skipped with almost no overhead (don't generate Ray tasks at all).

# Successive refinement is a key concept in AstroFlow.
#
# Many artifacts can be generated in extremely cheap, or eventually extremely
# expensive ways. Each stable object (stack, target, mosaic), could have
# potentially many successive increases in quality.
#
# Each of these objects should have a unified way to represent where they are
# along the ladder. The system should automatically run the next-highest rung,
# until every object has hit maximum quality.
#
# At any moment, the user should be able to prioritize any particular object(s).
# That just means running the compute graph once with those tasks in isolation,
# then resuming normal re-running.
#
# Order of operations:
#
# Read FITS headers from all images. This is very fast, and often includes
# approximate RA, Dec, etc.
#   RA, Dec, image scale, rotation, and width x height define a stack.
#   Do this on the node closest to the disk.
#   This gives plate_solutions[filename] as a first guess.
#   Images without FITS headers will be ignored until later.
#
# Group images into stacks, and stacks into targets, and targets into mosaics.
#
# Start with the first image in each run as representative of the run.
#   This is not necessarily correct, but it's a good guess for many datasets.
#   This produces immediate feedback to the user, which inspires confidence.
#   The best preview of a stack will be refined over time.
#     Before stacking, the best preview is the single subframe with most stars.
#     After stacking, the best preview is the post-processed (BXT, etc.) image.


@ray.remote(num_cpus=0.1)
@beartype
def scatter_compute_image_metadata(*filenames_in: list[Path]) -> Graph:
    """Build the startup metadata subgraph — DELTA-ONLY.

    Per input dir: a fresh scan (or, for a disconnected volume, the last
    durable record — "that disk hasn't changed") yields a fingerprint +
    listing. `compute_image_metadata` nodes are emitted ONLY for files that
    are NEW since the previous record, or whose SIDECAR IS MISSING; an
    unchanged, fully-sidecar'd library emits ZERO per-file nodes. Files
    whose size/mtime moved are deliberately NOT recomputed and NEVER have
    their sidecar deleted — see the write-once note below. Downstream, pure per-dir pieces
    (`build_dir_library`, fingerprint-keyed) and the pure union
    (`build_library_dicts`) are diskcache hits when nothing changed, so
    startup collapses to the scans plus a handful of cache lookups. The
    impure remainder is `publish_library_refs`: two context puts and the
    deferred persistence of the fresh scan records (deferred so a run that
    dies mid-way re-deltas the same files next start instead of swallowing
    them).

    CHANGED files (same path, different size/mtime) get their op cache entry
    invalidated AND their sidecar dropped here — `compute_image_metadata`
    keys on the filename literal and cold-starts from the sidecar
    unconditionally, so without both resets an in-place change would serve
    stale metadata forever.

    Explicit `filenames_in` (tests only — no production caller passes any)
    become ad-hoc compute nodes + one synthetic piece so they still reach
    the union.
    """
    from metadata.library_scan import (
        delta as scan_delta, load_record, scan_or_reuse,
    )
    from file_io.sidecar import sidecar_path_for
    from hashing import hash_call

    graph_dict: dict = {}
    pieces: list[tuple] = []
    records_to_persist: list[tuple] = []

    for dir in context.input_dirs:
        old = load_record(dir)          # BEFORE scan (scan doesn't persist)
        record, fresh = scan_or_reuse(dir)
        added, changed, removed = (
            scan_delta(old, record) if fresh else ([], [], [])
        )
        if fresh and (added or changed or removed or old is None):
            records_to_persist.append(
                (record.dir, record.fingerprint, record.listing)
            )
        log.info(
            "library scan", dir=str(dir), files=len(record.listing),
            fresh=fresh, added=len(added), changed=len(changed),
            removed=len(removed),
        )
        root = Path(record.dir)
        # SIDECARS ARE WRITE-ONCE AND NEVER DELETED (2026-08-13 decision).
        # A frame's pixels do not change after capture, and AstroFlow is
        # never permitted to modify a raw file — so a sidecar, once
        # written, stays valid forever. The previous design also acted on
        # a `changed` delta (same path, different size/mtime): it deleted
        # the sidecar AND evicted the op cache, and if that pass didn't
        # finish, the frame was left with neither and became INVISIBLE to
        # the sidecar-sourced library. That is how 25,576 of 32,666 frames
        # silently vanished and the inventory collapsed from ~416 targets
        # to 56, skipping most of the shopping cart. Re-solving after a
        # genuine pixel edit is a deliberate user action (future UI), not
        # something inferred from filesystem metadata that a network mount,
        # a clock, or a copy tool can perturb.
        #
        # The ONE thing we do infer: a listed file with NO sidecar needs
        # computing. One stat() per file per graph build, and the library
        # heals itself instead of staying permanently short.
        missing_sidecar = [
            rel for rel in _listing_rels(record.listing)
            if not sidecar_path_for(root / rel).exists()
        ]
        if missing_sidecar:
            log.warning(
                "library sidecars missing — recomputing", dir=str(dir),
                missing=len(missing_sidecar), listed=len(record.listing),
            )
        dir_nodes = {
            (compute_image_metadata, root / rel)
            for rel in (*added, *missing_sidecar)
        }
        piece = (
            build_dir_library, record.dir, record.fingerprint, record.listing,
        )
        # Ordering-only deps: the piece reads SIDECARS (written by the
        # compute nodes), not their return values — build_dir_library's
        # exact-arity signature keeps dep results out of its args/cache key.
        graph_dict[piece] = dir_nodes
        pieces.append(piece)

    extra = sorted({Path(f) for f in filenames_in if f is not None})
    if extra:
        listing = tuple(
            (str(f), f.stat().st_size if f.exists() else 0, 0) for f in extra
        )
        piece = (
            build_dir_library, "<explicit>",
            hash_call("explicit_files", listing), listing,
        )
        graph_dict[piece] = {(compute_image_metadata, f) for f in extra}
        pieces.append(piece)

    quarantined = tuple(sorted(context.quarantined_files or ()))
    union = (build_library_dicts, quarantined)
    # ORDERED deps (a list, in deterministic input_dirs order) — a set here
    # made the pieces' results swap argument positions between boots, which
    # flapped the union's call hash AND its output refs' identity between two
    # variants, re-keying everything downstream (identify_camera_fingerprints
    # recomputed ~46s every boot; trace caches re-keyed). Dep order IS part
    # of a value-consuming node's cache key — never pass it as a set.
    graph_dict[union] = pieces

    # Shared master folders (`metadata.shared_masters`): other people's
    # masters. Same per-folder piece + union shape as the raw library above;
    # a file whose sidecar is missing, or recorded other bytes, is measured
    # in place first (plate solve included).
    from metadata.shared_masters import (
        build_shared_dicts, build_shared_library, compute_shared_master,
        measurements_due, scan_shared_dir,
    )
    shared_pieces: list[tuple] = []
    for dir in context.shared_dirs or []:
        old = load_record(dir)
        record, fresh = scan_or_reuse(dir, scanner=scan_shared_dir)
        if fresh and (old is None or old.fingerprint != record.fingerprint):
            records_to_persist.append((record.dir, record.fingerprint, record.listing))
        root = Path(record.dir)
        to_measure = {
            (compute_shared_master, *due)
            for due in (measurements_due(root, old, record) if fresh else ())
        }
        log.info("shared masters scan", dir=str(dir), files=len(record.listing), fresh=fresh,
                 to_measure=len(to_measure))
        piece = (build_shared_library, record.dir, record.fingerprint, record.listing)
        graph_dict[piece] = to_measure
        shared_pieces.append(piece)
    shared_union = (build_shared_dicts,)
    graph_dict[shared_union] = shared_pieces

    publisher = (publish_library_refs, tuple(records_to_persist))
    graph_dict[publisher] = [union, shared_union]
    return Graph(graph_dict)


class DuplicateFileException(Exception):
    pass


# num_cpus stays DELIBERATELY below a solve's real core usage. Measured
# 2026-08-13: raising it to 1 (on the theory that a solve pegs a core, so the
# declaration should say so) cut head throughput 55.5 -> 45.2 sidecars/min.
# Concurrency here is meant to be governed by MAX_IN_FLIGHT (= 2 x cluster
# CPU) and the 1 GiB `memory` reservation, NOT by CPU: at 0.25 the cluster
# sits at ~7.5/34 CPU and 68 GiB/89.6 GiB memory, so the dispatcher's own cap
# binds. num_cpus=1 made CPU binding for the first time and capped the head at
# 24 concurrent, and the lost oversubscription had been usefully hiding the
# I/O latency of pulling 120 MB frames off the volumes. This op is also on the
# light-task path (estimate < _GATE_MIN_BYTES), so it dispatches unpinned and
# SPREAD — worker share is a scheduling non-issue; the worker simply completes
# fewer per minute because its frames cross the network.
@node(num_cpus=0.25, auto_memory=True, identity_root=True)
def compute_image_metadata(
    filename: os.PathLike,
    plate_solve: bool | None = None,
) -> ImageMetadata:
    log.info("Computing image metadata", filename=str(filename))

    if not filename.exists():
        raise FileNotFoundError(f'File "{filename}" not found')

    filename = filename.resolve()

    # Cold-start optimization: try to reconstruct from sidecar without FITS I/O.
    from metadata.sidecar_io import (
        read_image_metadata, write_image_metadata, write_defect_candidates,
    )
    from file_io.sidecar import get_section, set_section

    # Skip files previously marked as broken (e.g., truncated downloads).
    # The marker is self-invalidating: a producer rewrite touches the
    # file's mtime, so a marker older than the file is treated as stale
    # and the next read gets to retry. This lets us avoid touching the
    # sidecar at all on producer body-run (no `discard_sidecar`).
    broken = get_section(filename, "broken")
    if broken is not None:
        broken_at = broken.get("set_at", 0.0) if isinstance(broken, dict) else 0.0
        if filename.stat().st_mtime <= broken_at:
            log.warning("Skipping broken file", filename=str(filename))
            raise ValueError(f"Skipping broken file: {filename}")

    cached = read_image_metadata(filename)
    if cached is not None:
        log.debug("Restored image metadata from sidecar", filename=str(filename))
        return cached

    try:
        content_hash = get_content_hash(filename)
        header = read_header(filename)
        geometry = read_image_geometry(filename)
    except (ValueError, OSError) as e:
        log.error("File is broken, marking for skip", filename=str(filename), error=str(e))
        set_section(filename, "broken", {"error": str(e), "set_at": filename.stat().st_mtime})
        return None
    # str() because the XISF reader runs ast.literal_eval on keyword values, so
    # an all-digit STBLHASH round-trips as an int; a master's stable_hash must
    # stay a string identifier (else downstream str-keyed lookups KeyError).
    stable_hash = str(header.get("STBLHASH", content_hash))

    # `plate_solve` decides whether to solve:
    #   None  → raw-frame path (identification): solve lights, by IMAGETYP.
    #   bool  → produced-image path (the op's `@node(plate_solve=...)`): solve
    #           when asked, but NEVER a calibration master (bias/dark/flat have
    #           no star field). Calibration is recognized by its IMAGETYP, so a
    #           default of True is safe for the shared stacker without editing it.
    if plate_solve is None:
        should_solve = image_types.is_light_frame(header)
    else:
        is_calibration = (
            image_types.is_bias_frame(header)
            or image_types.is_dark_frame(header)
            or image_types.is_flat_frame(header)
        )
        should_solve = plate_solve and not is_calibration

    if should_solve:
        plate_solution = plate_solving.solve_field(filename)
        # DONTSOLV synthetic frames legitimately return None; everything
        # else must solve.
        if plate_solution is None and not header.get("DONTSOLV"):
            if plate_solve is not None:
                # PRODUCED-IMAGE path (a master, plate_solve=True): a
                # deliverable that won't solve is a real failure — the
                # mosaic can't place it. Fail the op.
                raise plate_solving.PlateSolveFailedException(
                    f"Failed to plate-solve: {filename}"
                )
            # RAW-FRAME path (identification): record the frame as
            # UNSOLVED and carry on, so it still gets a sidecar and stays
            # VISIBLE to the library. Raising here meant the sidecar was
            # never written, which under the sidecar-sourced library made
            # the frame cease to exist — a solve failure silently became
            # data loss, with no count anywhere. It also made the frame
            # eligible for recompute on EVERY pass (no sidecar => "needs
            # computing"), so a permanently-unsolvable frame burned a
            # solver slot forever. Unsolved frames are excluded from
            # spatial clustering by identify_stacks on their own merits.
            log.warning("frame did not plate-solve; recording as unsolved",
                        filename=str(filename))
    else:
        plate_solution = None

    # One pixel read serves both statistics (always wanted) and
    # defect extraction (conditional — only raw uint16 mono frames).
    # The cost is paid once per file via the sidecar cache.
    image, statistics, defect_candidates = _read_and_analyze_pixels(
        filename, header, geometry,
    )
    master_integration_s = read_master_integration_seconds(filename, header)
    # MRS noise floor of the master, written inline by ImageIntegration as
    # `NOISE00` (== finalNoiseEstimate). None on subframes / FastIntegration.
    _noise00 = header.get("NOISE00")
    try:
        noise_estimate = float(_noise00) if _noise00 is not None else None
    except (TypeError, ValueError):
        noise_estimate = None

    result = ImageMetadata(
        content_hash=content_hash,
        stable_hash=stable_hash,
        filename=filename,
        header=header,
        width_pixels=geometry.width_pixels,
        height_pixels=geometry.height_pixels,
        channels=geometry.channels,
        dtype=geometry.dtype,
        plate_solution=plate_solution,
        master_integration_s=master_integration_s,
        statistics=statistics,
        noise_estimate=noise_estimate,
    )

    write_image_metadata(result)
    # Defect candidates are written separately so that downstream
    # consumers that only need ImageMetadata (i.e. every node except
    # `identify_camera_fingerprints`) don't pay the ~1.8 GB deserialize
    # cost in their worker heaps. See `metadata/classes.py:DefectTable`.
    write_defect_candidates(filename, defect_candidates)

    if plate_solve is None and plate_solution is not None:
        # Raw subframe, solved: measure its Gaia star photometry into its
        # own sidecar section — the star flat's data. Best-effort and AFTER
        # the metadata is banked: a missing catalogue or unreadable file
        # must not cost the frame its sidecar; the section stays absent and
        # the backfill sweep (`metadata/star_photometry_backfill.py`) retries.
        try:
            star_photometry.compute_star_photometry(filename)
        except Exception as e:
            log.warning("star photometry failed; the sweep will retry",
                        filename=str(filename), error=str(e))
    return result


def _read_and_analyze_pixels(
    filename: Path, header: dict, geometry,
):
    """Read the image once; return (image, statistics, defect_candidates).

    Statistics are computed for every image (used by the web UI auto-
    stretch / structure-emphasis / HDR display, the seam-metric QA,
    and the panel comparison view). Defect extraction stays
    conditional on the historical "raw uint16 mono light/dark/flat,
    not a master" gate.

    A single read serves both — pixels stay hot in cache, and the
    sidecar caches the resulting metadata so we pay this once per
    file. On read failure we return (None, None, None); the caller
    stores the file as broken via the existing sidecar marker.
    """
    import numpy as np
    from metadata.image_statistics import compute_image_statistics

    try:
        image = read_image(filename)
    except (ValueError, OSError) as e:
        log.warning(
            "Skipping pixel analysis; image read failed",
            filename=str(filename), error=str(e),
        )
        return None, None, None

    try:
        statistics = compute_image_statistics(image)
    except Exception as e:
        # Stats failure shouldn't block defect extraction or the
        # metadata record itself; log and continue with None.
        log.warning(
            "compute_image_statistics failed",
            filename=str(filename), error=str(e),
        )
        statistics = None

    defects_eligible = (
        geometry.channels == 1
        and geometry.dtype == np.dtype(np.uint16)
        and not image_types.is_master(header)
        and (
            image_types.is_light_frame(header)
            or image_types.is_dark_frame(header)
            or image_types.is_flat_frame(header)
        )
    )
    defect_candidates = extract_defects(image) if defects_eligible else None

    return image, statistics, defect_candidates




# Head-pinned (diskcache is the head-only resource): this op reads every
# sidecar in the dir listing, and on a worker those reads traverse the
# network mount — measured 565s vs 83s on the head for the same 33k-file
# listing (2026-09-05 night, placement was luck-of-the-scheduler).
def _library_identity(metadata: dict) -> tuple:
    """What a consumer of the library dict actually depends on: WHICH frames
    (content hash) and WHERE they are (absolute path).

    `hash_list([metadata])` sees only the first half — `make_hashable`
    collapses each `ImageMetadata` to its own `content_hash`, which is
    content-addressed and therefore path-INDEPENDENT by design. So moving
    the library (Astro/Astro2 -> AstroRaw, 2026-08-18) changed no ref hash,
    invalidated nothing, and every cached Run went on serving paths that no
    longer existed: every WBPP stack died on FileNotFoundError and it took
    three rounds of manual eviction to clear. With paths in the identity a
    relocated library self-heals — `identify_stacks` /
    `identify_calibration_stacks` take the ref as an argument, so they
    re-run once and rebuild Runs with live paths, while `stack_files`
    (keyed on path-independent content hashes) keeps its key and not one
    master rebuilds.
    """
    return tuple(sorted((h, str(m.filename)) for h, m in metadata.items()))


@node(num_cpus=0.25, resources={"diskcache": 0.001})
def build_dir_library(
    dir: str, fingerprint: str, listing: tuple,
) -> tuple:
    """One dir's library piece: ({content_hash: ImageMetadata} ref,
    {content_hash: defects} ref), built from the SIDECARS of the files in
    `listing` — the canonical per-file store, written by
    `compute_image_metadata` (whose delta nodes are ordering-deps of this
    piece, so new files' sidecars exist by the time it runs).

    Cached on (dir, fingerprint, listing): an unchanged dir is a diskcache
    hit, and a DISCONNECTED dir's fingerprint literally cannot change — its
    piece hits forever without touching the absent volume. That is the
    whole "disconnected = unchanged" mechanism; no special-casing here.

    EXACT arity matters: with exactly three parameters the runtime passes
    only the literals (dep results stay out of the args and the cache key)
    — the compute nodes are ordering-only dependencies.

    Includes user-supplied external masters (already in the listing via
    `find_image_files(include_masters=True)`); their presence in the piece
    is what lets the trace @node's cache key invalidate when the user drops
    a fresh master in. Quarantine is NOT applied here (pieces stay
    quarantine-independent; the union filters), so a quarantine change
    never re-reads sidecars.
    """
    from metadata.sidecar_io import read_defect_candidates, read_image_metadata

    root = Path(dir)
    metadata: dict = {}
    defects: dict = {}
    for rel, _size, _mtime in listing:
        # The tests-only "<explicit>" piece carries absolute paths.
        f = Path(rel) if dir == "<explicit>" else root / rel
        try:
            meta = read_image_metadata(f)
            if meta is None or not isinstance(meta.content_hash, str):
                continue  # never analyzed / marked broken → next delta retries
            if meta.content_hash in metadata:
                continue  # duplicate frame content — first wins
            metadata[meta.content_hash] = meta
            d = read_defect_candidates(f)
            if d is not None:
                defects[meta.content_hash] = d
        except Exception as e:
            log.warning(
                "library piece: skipping unreadable sidecar",
                file=str(f), error=str(e),
            )
    log.info(
        "library piece built", dir=dir,
        entries=len(metadata), with_defects=len(defects),
    )
    return (ContextRef.store_large(metadata, _library_identity(metadata)),
            ContextRef.store_large(DefectTable.from_sets(defects)))


@node(num_cpus=0.25)
def build_library_dicts(quarantined: tuple, *pieces: tuple) -> tuple:
    """Union the per-dir pieces into the canonical (MetadataDict ref,
    DefectTable ref) pair, dropping quarantined content hashes.

    Pure and cached on (quarantined, *piece refs): unchanged pieces + an
    unchanged quarantine set = a diskcache hit returning the same refs —
    the union costs nothing on an unchanged library. A quarantine change
    re-runs only this cheap union, never the sidecar-reading pieces.

    NO existence filtering here (the old `load_metadata` dropped any frame
    failing `os.path.exists`, which made unplugging a disk mutate the
    MetadataDict → stack content hashes moved → spurious re-derives).
    Listing membership IS the filter now; frames on a disconnected volume
    remain first-class members and only genuine file READS can fail,
    locally, at the node that attempts them.
    """
    quarantined_set = set(quarantined)
    raw_metadata: dict = {}
    tables: list[DefectTable] = []
    taken: set = set(quarantined_set)
    for piece in pieces:
        meta_ref, defect_ref = piece
        for content_hash, value in meta_ref.get().items():
            if content_hash in quarantined_set or content_hash in raw_metadata:
                continue
            raw_metadata[content_hash] = value
        table = defect_ref.get().drop(taken)          # first piece wins, as for metadata
        taken.update(table.frames.tolist())
        tables.append(table)
        defect_ref.release()                          # the union is this piece's only reader
    # CANONICAL insertion order: the dicts' pickled bytes (and therefore the
    # stored refs' content hashes, and therefore every downstream cache key
    # embedding them — fingerprints, traces) must not depend on piece
    # iteration order. Sort by content hash so the value is byte-stable.
    metadata = MetadataDict()
    for content_hash in sorted(raw_metadata):
        metadata[content_hash] = raw_metadata[content_hash]
    defects = DefectTable.concat(tables)              # sorted by frame: byte-stable
    log.info(
        "library dicts built", entries=len(metadata), defects=len(defects),
    )
    return (ContextRef.store_large(metadata, _library_identity(metadata)),
            ContextRef.store_large(defects))


@ray.remote(num_cpus=0.1)
@beartype
@measure_performance
def publish_library_refs(
    records: tuple, pair: tuple, shared_ref: ContextRef,
) -> ContextRef[MetadataDict]:
    """The startup chain's single impure terminal: publish the refs and
    persist the fresh scan records.

    * `context["metadata_ref"]` / `context["defects_ref"]` — the fixed keys
      every other consumer (web server, trace cache keys,
      `identify_camera_fingerprints`) reads with one cheap RPC.
    * `context["shared_ref"]` — the shared master folders' masters
      (`metadata.shared_masters`), which `scatter_spatial_groups` turns into
      foreign stacks.
    * Persist each fresh dir record — deliberately HERE, after the pieces
      and union succeeded, so a run that dies earlier re-deltas the same
      files next start instead of swallowing them.
    * Returns `metadata_ref` — the subgraph's `final_result`, which
      `build_regroup_graph` consumes (same contract as the old
      `load_defects` terminal).
    """
    from metadata.library_scan import ScanRecord, store_record

    # Cache hits hand back refs with no ObjectRef; put the values back into
    # the object store once so every consumer reads them zero-copy.
    metadata_ref, defects_ref = pair[0].in_plasma(), pair[1].in_plasma()
    context["metadata_ref"] = metadata_ref
    context["defects_ref"] = defects_ref
    context["shared_ref"] = shared_ref
    for dir, fingerprint, listing in records:
        store_record(ScanRecord(
            dir=dir, fingerprint=fingerprint,
            listing=tuple(tuple(e) for e in listing),
        ))
    return metadata_ref


@ray.remote(num_cpus=0.1)
@beartype
@measure_performance
def load_library_now() -> ContextRef[MetadataDict]:
    """Tests/scripts convenience: the whole library chain in one Ray task —
    scan, per-dir pieces, union, publish — using the graph ops' underlying
    functions (`._function`), so the result is identical to a full
    `scatter_compute_image_metadata` run. Unlike the graph path this builds
    the pieces uncached (fixtures churn their libraries between calls, so a
    fingerprint cache would just miss anyway). Replaces the old
    `load_metadata.remote()` entry point."""
    from metadata.library_scan import ScanRecord, scan_or_reuse, store_record

    pieces = []
    records = []
    for dir in context.input_dirs:
        record, fresh = scan_or_reuse(dir)
        if fresh:
            records.append(record)
        pieces.append(build_dir_library._function(
            record.dir, record.fingerprint, record.listing,
        ))
    quarantined = tuple(sorted(context.quarantined_files or ()))
    metadata_ref, defects_ref = build_library_dicts._function(
        quarantined, *pieces,
    )
    context["metadata_ref"] = metadata_ref
    context["defects_ref"] = defects_ref
    for record in records:
        store_record(record)
    return metadata_ref
