import os
import warnings
import ray
import numpy as np
from beartype import beartype
from astropy.wcs import WCS, FITSFixedWarning  # type: ignore

warnings.filterwarnings("ignore", category=FITSFixedWarning)
from scipy.interpolate import RegularGridInterpolator  # type: ignore
from itertools import product
from pathlib import Path
from typing import Any
import structlog
from astropy.coordinates import SkyCoord  # type: ignore
from astropy import units as u  # type: ignore
from dataclasses import dataclass

from stacks.composite_stack import Stack
from file_io.generic import read_image
from file_io.tiff_file_io import write_tiff_image
from file_io.types import TiffFilePath, WithSidecar
from context import context
from dependency_graph import Graph
from previews.types import QualityLevel, ALL_IMPLEMENTATIONS, SUBFRAME, MASTER, DOWNSCALED_MASTER
from products.registry import get_filename_fn, MediaType
from mosaics.types import Mosaic, Viewbox
from mosaics.artifacts import placed_filename
from decorators import node
from metadata.classes import ImageMetadata
from plate_solving import PlateSolution
from products import makes_image

class ProjectionFailedException(Exception):
    pass

_GB = 1024 ** 3
_PLACE_TILE_MEMORY_REQUIREMENT = 128 * 1024 * 1024  # 128 MB
_PLACE_IMAGE_BUILDER_MEMORY = 2 * _GB  # transient: load image + build interpolator, then ray.put
_COMBINE_PLACE_TILES_MEMORY = 2 * _GB  # one viewbox-size float32 + incoming tile results

log = structlog.get_logger()


@dataclass(frozen=True)
class PlacedImage:
    pass


def round_to_nearest_multiple(number: float, multiple: int) -> int:
    return int(np.ceil(number / multiple) * multiple)


@ray.remote(memory=_PLACE_TILE_MEMORY_REQUIREMENT, scheduling_strategy="SPREAD")
@beartype
def place_tile(
    viewbox: Viewbox,
    tile_y: int,
    tile_x: int,
    wcs: Any,
    interp: RegularGridInterpolator,
    skip_empty: bool = True,
) -> tuple[int, int, np.ndarray | None]:
    tile_y_min = (
        viewbox.y_min + tile_y * viewbox.tile_height * viewbox.resolution
    )
    tile_y_max = tile_y_min + viewbox.tile_height * viewbox.resolution

    tile_x_min = (
        viewbox.x_min + tile_x * viewbox.tile_width * viewbox.resolution
    )
    tile_x_max = tile_x_min + viewbox.tile_width * viewbox.resolution

    if skip_empty:
        # Compute the four corner points in RA/Dec.
        corners_x = [tile_x_min, tile_x_max, tile_x_min, tile_x_max]
        corners_y = [tile_y_min, tile_y_min, tile_y_max, tile_y_max]
        ras, decs = viewbox.projection.unproject(
            np.array(corners_x), np.array(corners_y)
        )
        ras, decs = np.degrees(ras), np.degrees(decs)

        # If none of the corners of this tile fall inside the source image's
        # footprint, this tile can be skipped.
        inside_footprint = [
            wcs.footprint_contains(
                SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
            )
            for ra, dec in zip(ras, decs)
        ]
        if not any(inside_footprint):
            return tile_y, tile_x, None

    # Compute the RA, Dec of all the tile's pixels.
    ys = np.linspace(
        tile_y_min, tile_y_max, viewbox.tile_height, endpoint=False
    )
    xs = np.linspace(tile_x_min, tile_x_max, viewbox.tile_width, endpoint=False)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    ras, decs = viewbox.projection.unproject(xx, yy)
    ras, decs = np.degrees(ras), np.degrees(decs)

    source_xs, source_ys = wcs.all_world2pix(ras, decs, 0, quiet=True)
    tile_data = interp((source_ys, source_xs)).astype(np.float32)
    return tile_y, tile_x, tile_data


def _make_interpolator(filename: Path):
    # TODO: The subframes need to be calibrated before being placed.
    image = read_image(filename)

    # TODO: This seems to take up a huge amount of memory with method 'cubic.'
    # Maybe tiling would reduce the intermediate memory usage.
    source_height, source_width = image.shape
    return RegularGridInterpolator(
        (range(source_height), range(source_width)),
        image,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )


@node(memory=_COMBINE_PLACE_TILES_MEMORY)
def combine_place_tiles(
    mode: QualityLevel,
    mosaic: Mosaic,
    stack: Stack,
    filter: str,
    viewbox: Viewbox,
    *tile_results: tuple[int, int, np.ndarray | None],
) -> WithSidecar[TiffFilePath]:
    """Reducer: assemble the N place_tile results into one TIFF.

    Each tile_result is a (tile_y, tile_x, ndarray | None). None means
    the tile fell outside the source image footprint and was skipped.
    """
    projected_image = np.zeros(
        (
            viewbox.num_tiles_y * viewbox.tile_height,
            viewbox.num_tiles_x * viewbox.tile_width,
        ),
        dtype=np.float32,
    )
    for tile_y, tile_x, tile_data in tile_results:
        if tile_data is None:
            continue
        ty = tile_y * viewbox.tile_height
        tx = tile_x * viewbox.tile_width
        projected_image[
            ty : ty + viewbox.tile_height,
            tx : tx + viewbox.tile_width,
        ] = tile_data

    image = projected_image[: viewbox.num_pixels_y, : viewbox.num_pixels_x]

    if not np.any(image):
        raise ProjectionFailedException(
            f"Placed image for stack {stack.stable_hash} in mosaic "
            f"{mosaic.stable_hash} is entirely blank. This likely indicates "
            f"a WCS or viewbox mismatch."
        )

    output_fn = get_filename_fn(MediaType.IMAGE, PlacedImage, mode)
    output_filename = output_fn(mosaic, stack, filter, viewbox)
    write_tiff_image(output_filename, image, {"FILTER": filter, "STBLHASH": stack.stable_hash})
    log.info("Place image complete", output_filename=output_filename)
    return WithSidecar(output_filename)


@ray.remote(num_cpus=0.1, memory=_PLACE_IMAGE_BUILDER_MEMORY)
@makes_image(PlacedImage, ALL_IMPLEMENTATIONS, [SUBFRAME, MASTER, DOWNSCALED_MASTER], placed_filename)
def place_image(
    mode: QualityLevel,
    mosaic: Mosaic,
    stack: Stack,
    filter: str,
    viewbox: Viewbox,
    image_metadata: ImageMetadata,
) -> Graph:
    """Graph-builder for placing a source image onto the mosaic viewbox.

    Previously this was a single @node task that fanned out place_tile
    subtasks and awaited them internally. That pattern caused a classic
    Ray deadlock: each parent held the memory estimate of a full placed
    image (~10 GB) while waiting, so once enough parents piled up the
    cluster couldn't schedule their subtasks and the system deadlocked.

    Now this function builds an explicit Graph that Ray's scheduler
    can see: N cheap place_tile tasks and a single combine_place_tiles
    reducer. The builder itself uses transient memory to load the source
    image and build the scipy interpolator, then ray.put()s them and
    exits — no long-lived reservation.
    """
    assert image_metadata.plate_solution is not None, (
        f"place_image requires a plate solution on {image_metadata.filename}; "
        "DONTSOLV synthetic frames cannot be placed."
    )
    log.info("Placing image", mode=mode, mosaic=str(mosaic), stack=str(stack),
             filename=image_metadata.filename, viewbox=str(viewbox))

    wcs_ref = ray.put(WCS(image_metadata.plate_solution.header))
    interp_ref = ray.put(_make_interpolator(image_metadata.filename))

    tile_tasks = tuple(
        (place_tile, viewbox, tile_y, tile_x, wcs_ref, interp_ref, False)
        for tile_y, tile_x in product(
            range(viewbox.num_tiles_y), range(viewbox.num_tiles_x)
        )
    )
    combine_task = (combine_place_tiles, mode, mosaic, stack, filter, viewbox)

    return Graph({
        **{t: set() for t in tile_tasks},
        combine_task: set(tile_tasks),
    })
