import os
import ray
import numpy as np
from astropy.wcs import WCS  # type: ignore
from scipy.interpolate import RegularGridInterpolator  # type: ignore
from itertools import product
from pathlib import Path
from typing import Any, Dict
import structlog
from astropy.coordinates import SkyCoord  # type: ignore
from astropy import units as u  # type: ignore
from xisf import XISF  # type: ignore
from shapely import box  # type: ignore

from customization.types import CustomizedStack
from file_io.generic import get_content_hash, read_image
from context import context
from previews import QualityLevel
from mosaics.types import Mosaic, Viewbox, TileIndices
from mosaics.trimming import trim_image
from mosaics.artifacts import placed_filename
from task.astroflow_node import task
from stacks.composite_stack import CompositeStack

_PLACE_IMAGE_MEMORY_REQUIREMENT = 6 * 1024 * 1024 * 1024  # 6 GB
_PLACE_TILE_MEMORY_REQUIREMENT = 128 * 1024 * 1024  # 128 MB

log = structlog.get_logger()


def round_to_nearest_multiple(number: float, multiple: int) -> int:
  return int(np.ceil(number / multiple) * multiple)


@ray.remote(memory=_PLACE_TILE_MEMORY_REQUIREMENT, scheduling_strategy="SPREAD")
def place_tile(viewbox: Viewbox,
               tile_y: int,
               tile_x: int,
               wcs: Any,
               interp: RegularGridInterpolator,
               skip_empty: bool = False) -> tuple[int, int, np.ndarray]:
  tile_y_min = viewbox.y_min + tile_y * viewbox.tile_height * viewbox.resolution
  tile_y_max = tile_y_min + viewbox.tile_height * viewbox.resolution

  tile_x_min = viewbox.x_min + tile_x * viewbox.tile_width * viewbox.resolution
  tile_x_max = tile_x_min + viewbox.tile_width * viewbox.resolution

  if skip_empty:
    # Compute the four corner points in RA/Dec.
    corners_x = [tile_x_min, tile_x_max, tile_x_min, tile_x_max]
    corners_y = [tile_y_min, tile_y_min, tile_y_max, tile_y_max]
    ras, decs = viewbox.projection.unproject(np.array(corners_x),
                                             np.array(corners_y))
    ras, decs = np.degrees(ras), np.degrees(decs)

    # If none of the corners of this tile fall inside the source image's
    # footprint, this tile can be skipped.
    inside_footprint = [
        wcs.footprint_contains(SkyCoord(ra * u.deg, dec * u.deg, frame="icrs"))
        for ra, dec in zip(ras, decs)
    ]
    if not any(inside_footprint):
      return tile_y, tile_x, None

  # Compute the RA, Dec of all the tile's pixels.
  ys = np.linspace(tile_y_min, tile_y_max, viewbox.tile_height, endpoint=False)
  xs = np.linspace(tile_x_min, tile_x_max, viewbox.tile_width, endpoint=False)
  xx, yy = np.meshgrid(xs, ys, indexing="xy")
  ras, decs = viewbox.projection.unproject(xx, yy)
  ras, decs = np.degrees(ras), np.degrees(decs)

  source_xs, source_ys = wcs.all_world2pix(ras, decs, 0, quiet=True)
  tile_data = interp((source_ys, source_xs)).astype(np.float32)
  return tile_y, tile_x, tile_data


def _get_wcs(filename: Path):
  # TODO: The subframes need to be calibrated before being placed.
  content_hash = get_content_hash(filename)
  print(f'### _get_wcs: filename {filename} has content_hash {content_hash}')
  plate_solution = context.metadata[content_hash].plate_solution
  return WCS(plate_solution.header)


def _make_interpolator(filename: Path):
  # TODO: The subframes need to be calibrated before being placed.
  image = read_image(filename)

  img_median = np.median(image)
  img_mad = np.median(np.abs(image - img_median))

  print(f'### place_image: image shape {image.shape} dtype {image.dtype} '
        f'min {np.min(image)} max {np.max(image)} '
        f'img_median {img_median} img_mad {img_mad}')

  # target_median = 1.5e-3
  # target_mad = 2e-4
  # image = (image - img_median) * (target_mad / img_mad) + target_median

  # epsilon = 1e-6
  # image[image == 0.0] = epsilon

  # assert not np.any(image == 0.0)

  # TODO: This seems to take up a huge amount of memory with method 'cubic.'
  # Maybe tiling would reduce the intermediate memory usage.
  source_height, source_width = image.shape
  return RegularGridInterpolator((range(source_height), range(source_width)),
                                 image,
                                 method='linear',
                                 bounds_error=False,
                                 fill_value=0)


def _project_in_tiles(viewbox: Viewbox,
                      interp: RegularGridInterpolator,
                      wcs: WCS,
                      skip_empty: bool = False):
  wcs_ref = ray.put(wcs)

  projected_image = np.zeros((viewbox.num_tiles_y * viewbox.tile_height,
                              viewbox.num_tiles_x * viewbox.tile_width),
                             dtype=np.float32)

  interp_ref = ray.put(interp)
  remaining_refs = [
      place_tile.remote(viewbox, tile_y, tile_x, wcs_ref, interp_ref,
                        skip_empty)
      for tile_y, tile_x in product(range(viewbox.num_tiles_y),
                                    range(viewbox.num_tiles_x))
  ]
  while remaining_refs:
    ready_refs, remaining_refs = ray.wait(remaining_refs, num_returns=1)
    tile_y, tile_x, tile_data = ray.get(ready_refs[0])
    if tile_data is None:
      continue

    tile_y_idx = tile_y * viewbox.tile_height
    tile_x_idx = tile_x * viewbox.tile_width
    projected_image[tile_y_idx:tile_y_idx + viewbox.tile_height,
                    tile_x_idx:tile_x_idx + viewbox.tile_width] = tile_data

  return projected_image[0:viewbox.num_pixels_y, 0:viewbox.num_pixels_x]


def _save_image_to_file(mode: QualityLevel, mosaic: Mosaic,
                        stack: CompositeStack, tile_indices: TileIndices,
                        filter: str, image: np.ndarray) -> Path:
  filename = placed_filename(mosaic, mode, stack, tile_indices, filter)
  filename.parent.mkdir(parents=True, exist_ok=True)
  fits_meta: Dict[str, Any] = {}
  fits_meta['FILTER'] = [{'value': str(filter), 'comment': ''}]
  ims_meta = {'FITSKeywords': fits_meta}
  XISF.write(filename,
             np.expand_dims(image, -1),
             creator_app="AstroFlow",
             image_metadata=ims_meta)
  print(f'### Wrote filename {filename}')
  return filename


@task.remote(memory=_PLACE_IMAGE_MEMORY_REQUIREMENT)
def place_image(mode: QualityLevel, mosaic: Mosaic, stack: CustomizedStack,
                tile_indices: TileIndices, filter: str, viewbox: Viewbox,
                mask: box, filename: Path) -> Path:
  # if context.dangerously_accept_existing_files and os.path.exists(filename):
  #   print(f'### Dangerously skipping place_large_image because '
  #         f'file {filename} already exists')
  #   return filename

  print(
      f'### place_image {mode} mosaic {mosaic.stable_hash} filename {filename}')
  assert filename, f"place_image: bad filename {filename}"

  wcs = _get_wcs(filename)
  interp = _make_interpolator(filename)

  skip_empty = mode == QualityLevel.MASTER
  image = _project_in_tiles(viewbox, interp, wcs, skip_empty=skip_empty)

  if mask:
    image = trim_image(image, mask)

  return _save_image_to_file(mode, mosaic, stack, tile_indices, filter, image)
