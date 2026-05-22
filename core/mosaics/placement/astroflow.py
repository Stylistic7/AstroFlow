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
from targets.types import Target
from file_io.generic import read_image
from file_io.tiff_file_io import write_tiff_image
from file_io.types import TiffFilePath, WithSidecar
from context import context
from mosaics.types import Mosaic, Viewbox
from decorators import node
from metadata.classes import ImageMetadata
from plate_solving import PlateSolution
from products import makes_image
from products.identity import PlacedImageId

class ProjectionFailedException(Exception):
    pass

_GB = 1024 ** 3
_PLACE_TILE_MEMORY_REQUIREMENT = 128 * 1024 * 1024  # 128 MB

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

def _project_in_tiles(
    viewbox: Viewbox,
    interp: RegularGridInterpolator,
    wcs: WCS,
    skip_empty: bool = True,
):
    wcs_ref = ray.put(wcs)

    projected_image = np.zeros(
        (
            viewbox.num_tiles_y * viewbox.tile_height,
            viewbox.num_tiles_x * viewbox.tile_width,
        ),
        dtype=np.float32,
    )

    interp_ref = ray.put(interp)
    remaining_refs = [
        place_tile.remote(
            viewbox, tile_y, tile_x, wcs_ref, interp_ref, skip_empty
        )
        for tile_y, tile_x in product(
            range(viewbox.num_tiles_y), range(viewbox.num_tiles_x)
        )
    ]
    while remaining_refs:
        ready_refs, remaining_refs = ray.wait(remaining_refs, num_returns=1)
        tile_y, tile_x, tile_data = ray.get(ready_refs[0])
        if tile_data is None:
            continue

        tile_y_idx = tile_y * viewbox.tile_height
        tile_x_idx = tile_x * viewbox.tile_width
        projected_image[
            tile_y_idx : tile_y_idx + viewbox.tile_height,
            tile_x_idx : tile_x_idx + viewbox.tile_width,
        ] = tile_data

    return projected_image[0 : viewbox.num_pixels_y, 0 : viewbox.num_pixels_x]

# Explicit memory ceiling (not auto_memory) because place_image spawns
# place_tile subtasks and waits for them internally via ray.wait().
# auto_memory's p95 captured the TOTAL RSS while the parent sat waiting
# (including the interpolator + accumulated tile results ≈ 10 GB) and
# then reserved that much — which starved the subtasks of scheduling
# headroom and deadlocked the cluster. 2 GB covers the parent's own
# footprint (interpolator + output buffer) while leaving room for many
# concurrent place_tile subtasks.
@node
@makes_image(identity=PlacedImageId)
def place_image(
    mosaic: Mosaic, target: Target, filter: str, viewbox: Viewbox,
    image_metadata: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """Project one panel's master into the mosaic's viewbox.

    Pipeline-agnostic: the downscaling — if any — is fully captured
    by the `viewbox` size and by the upstream master (already
    downscaled for the PixInsight Downscaled pipeline). place_image
    just projects whatever it's given. PlacedImageId includes the
    viewbox, and the on-disk filename includes a viewbox content-
    hash prefix so PI Master and PI Downscaled outputs land in
    distinct files."""
    assert image_metadata.plate_solution is not None, (
        f"place_image requires a plate solution on {image_metadata.filename}; "
        "DONTSOLV synthetic frames cannot be placed."
    )
    log.info("Placing image", mosaic=str(mosaic), target=str(target),
             filter=filter, filename=image_metadata.filename, viewbox=str(viewbox))

    wcs = WCS(image_metadata.plate_solution.header)
    interp = _make_interpolator(image_metadata.filename)
    image = _project_in_tiles(viewbox, interp, wcs, skip_empty=False)

    output_filename = place_image.filename_fn(mosaic, target, filter, viewbox)

    if not np.any(image):
        raise ProjectionFailedException(
            f"Placed image for target {target.stable_hash} in mosaic "
            f"{mosaic.stable_hash} ({filter}) is entirely blank. This "
            f"likely indicates a WCS or viewbox mismatch."
        )

    write_tiff_image(output_filename, image, {
        "FILTER": filter,
        "STBLHASH": target.stable_hash,
    })
    # A placed image is one panel by definition. merge_images sums
    # input counts to track "most complete merge so far." Stored in the
    # context cache (not the file sidecar) because @makes_image wipes
    # the sidecar after every producer run.
    from mosaics.merging import _set_panel_count
    _set_panel_count(output_filename, 1)
    log.info("Place image complete", output_filename=output_filename)

    return WithSidecar(output_filename)
