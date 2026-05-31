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
from metadata.classes import ImageMetadata, MetadataDict
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

    graph = Graph(
        {load_metadata: {(compute_image_metadata, Path(f)) for f in filenames}}
    )

    return graph


class DuplicateFileException(Exception):
    pass


@node(num_cpus=0.25)
def compute_image_metadata(
    filename: os.PathLike,
) -> ImageMetadata:
    log.info("Computing image metadata", filename=str(filename))

    if not filename.exists():
        raise FileNotFoundError(f'File "{filename}" not found')

    filename = filename.resolve()

    # Cold-start optimization: try to reconstruct from sidecar without FITS I/O.
    from metadata.sidecar_io import read_image_metadata, write_image_metadata
    from file_io.sidecar import get_section, set_section

    # Skip files previously marked as broken (e.g., truncated downloads).
    if get_section(filename, "broken") is not None:
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
        set_section(filename, "broken", {"error": str(e)})
        return None
    stable_hash = header.get("STBLHASH", content_hash)

    if image_types.is_light_frame(header):
        plate_solution = plate_solving.solve_field(filename)
        # DONTSOLV synthetic frames legitimately return None; everything
        # else must solve.
        if plate_solution is None and not header.get("DONTSOLV"):
            raise plate_solving.PlateSolveFailedException(
                f"Failed to plate-solve light frame: {filename}"
            )
    else:
        plate_solution = None

    defect_candidates = _maybe_extract_defects(filename, header, geometry)
    master_integration_s = read_master_integration_seconds(filename, header)

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
        defect_candidates=defect_candidates,
        master_integration_s=master_integration_s,
    )

    write_image_metadata(result)
    return result


def _maybe_extract_defects(
    filename: Path, header: dict, geometry
) -> dict[int, float] | None:
    """Run the hot-pixel detector only on frame types where it's meaningful.

    Lights/darks/flats with raw uint16 mono pixels carry stable defect
    signal. Biases are too short (RTS noise dominates), masters are
    post-rejection float32 (the raw defects are gone), and anything
    multi-channel is debayered/rendered. For all those cases, return
    None and skip the pixel read entirely.
    """
    import numpy as np

    if geometry.channels != 1 or geometry.dtype != np.dtype(np.uint16):
        return None
    if image_types.is_master(header):
        return None
    if not (
        image_types.is_light_frame(header)
        or image_types.is_dark_frame(header)
        or image_types.is_flat_frame(header)
    ):
        return None
    try:
        image = read_image(filename)
    except (ValueError, OSError) as e:
        log.warning(
            "Skipping defect extraction; image read failed",
            filename=str(filename), error=str(e),
        )
        return None
    return extract_defects(image)




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
