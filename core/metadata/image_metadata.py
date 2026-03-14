import ray
import os
from pathlib import Path

from file_io.find_image_files import find_image_files
from file_io.generic import read_header, get_content_hash, read_image_geometry
import metadata.image_types as image_types
from metadata.classes import ImageMetadata, MetadataDict
import plate_solving
from dependency_graph import Graph
from context import context
from decorators.decorators import cached, measure_performance
from hashing.types import LargeObject

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
@ray.remote
def scatter_compute_image_metadata(*filenames_in: list[Path]) -> Graph:
    filenames_in = set(f for f in filenames_in if f is not None)
    filenames: set[str] = set(filenames_in)

    # TODO: Don't do anything if the directory's modification time hasn't
    # changed since the last time it was listed.
    for dir in context.input_dirs:
        artifacts_for_dir = set(find_image_files(dir))
        print(f"### Directory {dir} contains {len(artifacts_for_dir)} files")
        filenames.update(artifacts_for_dir)

    graph = Graph(
        {load_metadata: {(compute_image_metadata, Path(f)) for f in filenames}}
    )

    return graph


class DuplicateFileException(Exception):
    pass


@cached.remote(num_cpus=0.25)
def compute_image_metadata(filename: Path) -> ImageMetadata:
    print(f"### compute_image_metadata filename={filename}")

    if filename is None or not filename.exists():
        raise FileNotFoundError(f'File "{filename}" not found')

    filename = filename.resolve()
    content_hash = get_content_hash(filename)
    header = read_header(filename)
    geometry = read_image_geometry(filename)

    plate_solution = (
        plate_solving.solve_field(filename)
        if image_types.is_light_frame(header)
        else None
    )

    return ImageMetadata(
        content_hash=content_hash,
        filename=filename,
        header=header,
        width_pixels=geometry.width_pixels,
        height_pixels=geometry.height_pixels,
        channels=geometry.channels,
        dtype=geometry.dtype,
        plate_solution=plate_solution,
    )


@ray.remote(resources={"fastest_storage": 0.001})
@measure_performance
def load_metadata() -> LargeObject:
    quarantined_files = context.quarantined_files or set()

    # TODO: This won't support disconnected storage media.
    input_dirs = [str(Path(d).resolve()) for d in (context.input_dirs or [])]

    def is_input_file(filename) -> bool:
        resolved = str(Path(filename).resolve())
        return any(resolved.startswith(d) for d in input_dirs)

    metadata = MetadataDict()
    for key, value in context.get_matching(("compute_image_metadata",)).items():
        try:
            if isinstance(value.content_hash, str):
                if value.content_hash not in quarantined_files:
                    if is_input_file(value.filename):
                        if "master" not in os.path.basename(value.filename).lower():
                            if os.path.exists(value.filename):
                                if not image_types.is_master(value.header):
                                    metadata[value.content_hash] = value
        except Exception as e:
            print(
                f"### Rejecting bad key {key} because of exception {e} for "
                f"value type {type(value)}"
            )

    # TODO: Pass the LargeObject everywhere, and remove this from the context.
    context.metadata = metadata

    return LargeObject(metadata)
