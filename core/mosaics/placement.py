import os
import ray
import numpy as np
from astropy.wcs import WCS  # type: ignore
from scipy.interpolate import RegularGridInterpolator  # type: ignore
from itertools import product
from pathlib import Path
from typing import Any
import structlog
from astropy.coordinates import SkyCoord  # type: ignore
from astropy import units as u  # type: ignore
from dataclasses import dataclass

from customization.types import CustomizedStack
from file_io.generic import read_image, read_header
from file_io.tiff_file_io import write_tiff_image
from file_io.types import TiffFilePath
from context import context
from previews.types import QualityLevel
from mosaics.types import Mosaic, Viewbox, Mask
from mosaics.trimming import (
    trim_image,
    project_to_viewbox,
    largest_interior_rectangle,
)
from mosaics.artifacts import placed_filename
from decorators import cached
from metadata.classes import ImageMetadata
from products import makes_image

_PLACE_IMAGE_MEMORY_REQUIREMENT = 6 * 1024 * 1024 * 1024  # 6 GB
_PLACE_TILE_MEMORY_REQUIREMENT = 128 * 1024 * 1024  # 128 MB

log = structlog.get_logger()


@dataclass(frozen=True)
class PlacedImage:
    pass


def round_to_nearest_multiple(number: float, multiple: int) -> int:
    return int(np.ceil(number / multiple) * multiple)


@ray.remote(memory=_PLACE_TILE_MEMORY_REQUIREMENT, scheduling_strategy="SPREAD")
def place_tile(
    viewbox: Viewbox,
    tile_y: int,
    tile_x: int,
    wcs: Any,
    interp: RegularGridInterpolator,
    skip_empty: bool = False,
) -> tuple[int, int, np.ndarray]:
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
    skip_empty: bool = False,
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


@cached.remote
def _place_image(
    mode: QualityLevel,
    mosaic: Mosaic,
    stack: CustomizedStack,
    filter: str,
    viewbox: Viewbox,
    mask: Mask,
    image_metadata: ImageMetadata,
) -> TiffFilePath:
    # if context.dangerously_accept_existing_files and os.path.exists(filename):
    #   print(f'### Dangerously skipping place_large_image because '
    #         f'file {filename} already exists')
    #   return filename

    filename = image_metadata.filename

    print(
        f"### place_image {mode} mosaic {str(mosaic)} stack {str(stack)} {filename=} viewbox {str(viewbox)}"
    )
    assert filename, f"place_image: bad filename {filename}"

    wcs = WCS(image_metadata.plate_solution.header)

    interp = _make_interpolator(filename)

    skip_empty = mode == QualityLevel.MASTER
    image = _project_in_tiles(viewbox, interp, wcs, skip_empty=skip_empty)

    # Trim the image to its own largest interior rectangle.
    if mode == QualityLevel.MASTER:
        print(f"### _place_image trimming largest_interior_rectangle")
        box = largest_interior_rectangle(
            project_to_viewbox(stack.footprint, viewbox)
        )
        image = trim_image(image, box)

    # Maybe trim the image again to the provided mask.
    if mask:
        image = trim_image(image, mask)

    # TODO: This is really awful; can the structure be improved?
    if mode == QualityLevel.SUBFRAME:
        output_filename = place_image_subframe.filename_fn(
            mosaic, stack, filter, viewbox, mask
        )
    elif mode == QualityLevel.MASTER:
        output_filename = place_image_master.filename_fn(
            mosaic, stack, filter, viewbox, mask
        )

    write_tiff_image(output_filename, image, {"FILTER": filter})
    print(f"### _place_image {output_filename=}")

    return output_filename


@makes_image(PlacedImage).as_subframe(placed_filename)
@ray.remote
def place_image_subframe(
    mode: QualityLevel,
    mosaic: Mosaic,
    stack: CustomizedStack,
    filter: str,
    viewbox: Viewbox,
    mask: Mask,
    image_metadata: ImageMetadata,
) -> TiffFilePath:
    return _place_image.remote(
        mode, mosaic, stack, filter, viewbox, mask, image_metadata
    )


@makes_image(PlacedImage).as_master(placed_filename)
@ray.remote(memory=_PLACE_IMAGE_MEMORY_REQUIREMENT)
def place_image_master(
    mode: QualityLevel,
    mosaic: Mosaic,
    stack: CustomizedStack,
    filter: str,
    viewbox: Viewbox,
    mask: Mask,
    image_metadata: ImageMetadata,
) -> TiffFilePath:
    return _place_image.remote(
        mode, mosaic, stack, filter, viewbox, mask, image_metadata
    )


# TODO: These trampolines are ugly; can they be improved?
@ray.remote
def place_image(
    mode: QualityLevel,
    mosaic: Mosaic,
    stack: CustomizedStack,
    filter: str,
    viewbox: Viewbox,
    mask: Mask,
    image_metadata: ImageMetadata,
) -> TiffFilePath:
    if mode == QualityLevel.SUBFRAME:
        return place_image_subframe.remote(
            mode, mosaic, stack, filter, viewbox, mask, image_metadata
        )
    elif mode == QualityLevel.MASTER:
        return place_image_master.remote(
            mode, mosaic, stack, filter, viewbox, mask, image_metadata
        )
