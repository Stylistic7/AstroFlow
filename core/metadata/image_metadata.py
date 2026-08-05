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
from metadata.classes import ImageMetadata, MetadataDict, DefectsDict
import plate_solving
from dependency_graph import Graph
from context import context
from decorators.decorators import node, measure_performance
from cache.context_ref import ContextRef
from decisions.settings import DevelopmentSettings

log = structlog.get_logger()

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


# TODO: Don't revisit the same failed images over and over, it just takes up
# time at every startup.
#
# TODO: Add a task that checks that input and output directories exist. Many
# tasks cannot run without them.
@ray.remote(num_cpus=0.1)
@beartype
def scatter_compute_image_metadata(*filenames_in: list[Path]) -> Graph:
    filenames_in = set(f for f in filenames_in if f is not None)
    filenames: set[str] = set(filenames_in)

    # TODO: Don't do anything if the directory's modification time hasn't
    # changed since the last time it was listed.
    for dir in context.input_dirs:
        # Include user-supplied external masters (`Calibration Masters/
        # MasterDark*.fit` etc.) in the same sweep as raw subframes. Their
        # ImageMetadata flowing into the diskcache is what makes the trace
        # @node's cache key (which embeds metadata_ref.content_hash)
        # invalidate when the user drops in a fresh master.
        artifacts_for_dir = set(find_image_files(dir, include_masters=True))
        log.info("Found image files in directory", directory=str(dir), count=len(artifacts_for_dir))
        filenames.update(artifacts_for_dir)

    compute_tasks = {(compute_image_metadata, Path(f)) for f in filenames}
    # `load_defects` runs after `load_metadata` so they share the same
    # one-time cache scan ordering. Together they publish both
    # `metadata_ref` and `defects_ref` into context before the regroup
    # graph starts. `load_defects` is the single terminal node — its
    # return value becomes the subgraph's `final_result`.
    graph = Graph(
        {
            load_metadata: compute_tasks,
            load_defects: {load_metadata},
        }
    )

    return graph


class DuplicateFileException(Exception):
    pass


@node(num_cpus=0.25, auto_memory=True)
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
            raise plate_solving.PlateSolveFailedException(
                f"Failed to plate-solve: {filename}"
            )
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
    # cost in their worker heaps. See `metadata/classes.py:DefectsDict`.
    write_defect_candidates(filename, defect_candidates)
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




@ray.remote(num_cpus=0.1)
@beartype
@measure_performance
def load_metadata() -> ContextRef[MetadataDict]:
    """Build the canonical MetadataDict from cached compute_image_metadata
    entries.

    Includes both raw subframes AND user-supplied external masters under
    `input_dirs`. The master inclusion lets the trace @node's cache key
    (which embeds `metadata_ref.content_hash`) invalidate when the user
    drops a fresh master file in. Downstream consumers that only want
    light frames already filter internally: `clustering/spatial.py` gates
    spatial_groups on `is_light_frame and not is_master`, and
    `identify_calibration_stacks` re-applies `not is_master` as a
    second-line defense for masters with misleading headers.

    The input_dirs gate prevents AstroFlow-produced masters in
    `output_dir` from ever entering — only user-supplied files participate.
    """
    quarantined_files = context.quarantined_files or set()

    # TODO: This won't support disconnected storage media.
    input_dirs = [str(Path(d).resolve()) for d in (context.input_dirs or [])]

    def is_input_file(filename) -> bool:
        resolved = str(Path(filename).resolve())
        return any(resolved.startswith(d) for d in input_dirs)

    metadata = MetadataDict()
    for key, value in context.get_matching(("compute_image_metadata",)).items():
        try:
            if not isinstance(value.content_hash, str):
                continue
            if value.content_hash in quarantined_files:
                continue
            if not is_input_file(value.filename):
                continue
            if not os.path.exists(value.filename):
                continue
            existing = metadata.get(value.content_hash)
            if existing is None:
                metadata[value.content_hash] = value
        except Exception as e:
            log.warning("Rejecting bad metadata key", key=key, error=str(e), value_type=type(value).__name__)

    ref = ContextRef.store_large(metadata)
    # Publish the (small) ContextRef under a fixed key so other callers
    # can grab it via one cheap RPC instead of re-running this whole
    # scan. `load_metadata` itself is ~20-30 s on a populated cache
    # (every entry gets an os.path.exists), and the trace's @node cache
    # key needs metadata_ref.content_hash, so without this publish the
    # web server pays the full scan cost on every request before it
    # can even check whether the trace is cached.
    context["metadata_ref"] = ref
    return ref


@ray.remote(num_cpus=0.1)
@beartype
@measure_performance
def load_defects(
    metadata_ref: ContextRef[MetadataDict],
) -> ContextRef[MetadataDict]:
    """Build the canonical DefectsDict from cached sidecars and publish
    it via `context["defects_ref"]`.

    Reads only the `defect_candidates` sidecar section. Used by
    `identify_camera_fingerprints` (the only op that needs per-frame
    defects); every other op skips this fetch and stays small.

    **Returns the upstream `metadata_ref` unchanged** so that this task
    can sit as the chain-terminal of `scatter_compute_image_metadata`
    without altering the value that downstream chain consumers receive
    (`build_regroup_graph`, ad-hoc test fixtures, etc.). The
    DefectsDict itself flows out of band via `context["defects_ref"]`.
    Same input-dirs gate as `load_metadata`: only user-supplied
    frames participate.
    """
    from metadata.sidecar_io import read_defect_candidates

    quarantined_files = context.quarantined_files or set()
    input_dirs = [str(Path(d).resolve()) for d in (context.input_dirs or [])]

    def is_input_file(filename) -> bool:
        resolved = str(Path(filename).resolve())
        return any(resolved.startswith(d) for d in input_dirs)

    defects = DefectsDict()
    for key, value in context.get_matching(("compute_image_metadata",)).items():
        try:
            if not isinstance(value.content_hash, str):
                continue
            if value.content_hash in quarantined_files:
                continue
            if not is_input_file(value.filename):
                continue
            if not os.path.exists(value.filename):
                continue
            if value.content_hash in defects:
                continue
            d = read_defect_candidates(Path(value.filename))
            if d is not None:
                defects[value.content_hash] = d
        except Exception as e:
            log.warning(
                "Rejecting bad defects key", key=key,
                error=str(e), value_type=type(value).__name__,
            )

    ref = ContextRef.store_large(defects)
    context["defects_ref"] = ref
    return metadata_ref
