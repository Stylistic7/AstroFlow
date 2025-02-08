import os
import ray
import numpy as np
from astropy.wcs import WCS  # type: ignore
from scipy.interpolate import RegularGridInterpolator  # type: ignore
import largestinteriorrectangle  # type: ignore
import cv2 as cv
from itertools import product
from pathlib import Path
from typing import Any
import structlog
import astropy.io.fits as pyfits  # type: ignore
from astropy.coordinates import SkyCoord  # type: ignore
from astropy import units as u  # type: ignore

from file_io.generic import get_content_hash, read_image
from context import context
from previews import QualityLevel
from mosaics.types import Mosaic, Viewbox, TileIndices, Tiling
from dependency_graph.astroflow_node import AstroFlowNode
from stacks.composite_stack import CompositeStack

_TILE_SIZE = 2048  # pixels
_PLACE_IMAGE_MEMORY_REQUIREMENT = 12 * 1024 * 1024 * 1024  # 12 GB
_PLACE_TILE_MEMORY_REQUIREMENT = 128 * 1024 * 1024  # 128 MB

log = structlog.get_logger()


def round_to_nearest_multiple(number: float, multiple: int) -> int:
  return int(np.ceil(number / multiple) * multiple)


def _compute_tiling(mode: QualityLevel, mosaic: Mosaic,
                    viewbox: Viewbox) -> Tiling:
  tile_width = _TILE_SIZE
  tile_height = _TILE_SIZE

  def round_to_nearest_multiple(number: float, multiple: int) -> int:
    return int(np.ceil(number / multiple) * multiple)

  num_samples_x = np.degrees(viewbox.x_span) / (
      mosaic.image_scale / 3600) if mode == QualityLevel.MASTER else 1500
  num_samples_x = round_to_nearest_multiple(num_samples_x, multiple=tile_width)

  resolution = viewbox.x_span / num_samples_x
  num_samples_y = round_to_nearest_multiple(viewbox.y_span / resolution,
                                            multiple=tile_height)

  print(f'### _compute_tiling image_scale {mosaic.image_scale} '
        f'num_samples_y {num_samples_y} num_samples_x {num_samples_x}')

  num_tiles_y = int(np.ceil(num_samples_y / tile_height))
  num_tiles_x = int(np.ceil(num_samples_x / tile_width))
  return Tiling(num_tiles_y, num_tiles_x, tile_height, tile_width)


@ray.remote(memory=_PLACE_TILE_MEMORY_REQUIREMENT, scheduling_strategy="SPREAD")
def place_tile(viewbox: Viewbox,
               tiling: Tiling,
               tile_y: int,
               tile_x: int,
               wcs: Any,
               interp: RegularGridInterpolator,
               skip_empty: bool = False) -> tuple[int, int, np.ndarray]:
  y_step = viewbox.y_span / (tiling.num_tiles_y * tiling.tile_height)
  x_step = viewbox.x_span / (tiling.num_tiles_x * tiling.tile_width)

  tile_y_min = viewbox.y_min + tile_y * tiling.tile_height * y_step
  tile_y_max = tile_y_min + tiling.tile_height * y_step

  tile_x_min = viewbox.x_min + tile_x * tiling.tile_width * x_step
  tile_x_max = tile_x_min + tiling.tile_width * x_step

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
  ys = np.linspace(tile_y_min, tile_y_max, tiling.tile_height, endpoint=False)
  xs = np.linspace(tile_x_min, tile_x_max, tiling.tile_width, endpoint=False)
  xx, yy = np.meshgrid(xs, ys, indexing="xy")
  ras, decs = viewbox.projection.unproject(xx, yy)
  ras, decs = np.degrees(ras), np.degrees(decs)

  source_xs, source_ys = wcs.all_world2pix(ras, decs, 0, quiet=True)
  tile_data = interp((source_ys, source_xs)).astype(np.float32)
  return tile_y, tile_x, tile_data


def trim_image(projected_image):
  grid = projected_image != 0
  cv_grid = grid.astype("uint8") * 255
  contours, _ = cv.findContours(cv_grid, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
  contour = contours[0][:, 0, :]

  trim_left, trim_top, trim_width, trim_height = largestinteriorrectangle.lir(
      grid, contour)
  log.info(f'trim_top {trim_top} trim_left {trim_left} trim_height '
           f'{trim_height} trim_width {trim_width}')

  trim_bottom = trim_top + trim_height
  trim_right = trim_left + trim_width

  if trim_left == 0 and trim_top == 0 and trim_width == projected_image.shape[
      1] and trim_height == projected_image.shape[0]:
    assert False, "largestinteriorrectangle matched entire image"

  trimmed_image = np.zeros(projected_image.shape, np.float32)
  trimmed_image[trim_top:trim_bottom,
                trim_left:trim_right] = projected_image[trim_top:trim_bottom,
                                                        trim_left:trim_right]
  return trimmed_image


def _get_wcs(filename: Path):
  # TODO: The subframes need to be calibrated before being placed.
  content_hash = get_content_hash(filename)
  print(f'### _get_wcs: filename {filename} has content_hash {content_hash}')
  plate_solution = context.metadata[content_hash].plate_solution
  return WCS(plate_solution.header)


def _make_interpolator(filename: Path):
  # TODO: The subframes need to be calibrated before being placed.
  image = read_image(filename)

  print(f'### place_image: image shape {image.shape} dtype {image.dtype} '
        f'min {np.min(image)} max {np.max(image)}')

  # TODO: This seems to take up a huge amount of memory with method 'cubic.'
  # Maybe tiling would reduce the intermediate memory usage.
  source_height, source_width = image.shape
  return RegularGridInterpolator((range(source_height), range(source_width)),
                                 image,
                                 method='linear',
                                 bounds_error=False,
                                 fill_value=0)


def _project_in_tiles(viewbox: Viewbox,
                      tiling: Tiling,
                      interp: RegularGridInterpolator,
                      wcs: WCS,
                      skip_empty: bool = False):
  wcs_ref = ray.put(wcs)

  num_samples_y = tiling.num_tiles_y * tiling.tile_height
  num_samples_x = tiling.num_tiles_x * tiling.tile_width
  projected_image = np.zeros((num_samples_y, num_samples_x), dtype=np.float32)

  interp_ref = ray.put(interp)
  remaining_refs = [
      place_tile.remote(viewbox, tiling, tile_y, tile_x, wcs_ref, interp_ref,
                        skip_empty)
      for tile_y, tile_x in product(range(tiling.num_tiles_y),
                                    range(tiling.num_tiles_x))
  ]
  while remaining_refs:
    ready_refs, remaining_refs = ray.wait(remaining_refs, num_returns=1)
    tile_y, tile_x, tile_data = ray.get(ready_refs[0])
    if tile_data is None:
      continue

    tile_y_idx = tile_y * tiling.tile_height
    tile_x_idx = tile_x * tiling.tile_width
    projected_image[tile_y_idx:tile_y_idx + tiling.tile_height,
                    tile_x_idx:tile_x_idx + tiling.tile_width] = tile_data

  return projected_image


def _save_image_to_file(mode: QualityLevel, mosaic: Mosaic,
                        stack: CompositeStack, tile_indices: TileIndices,
                        filter: str, projected_image: np.ndarray) -> Path:
  tile_indices_str = '_'.join(str(i) for i in tile_indices)
  filename = Path(context.output_dir) / 'Mosaics' / mosaic.name / Path(
      f'placed_image_{mode}_{mosaic.stable_hash}_{stack.stable_hash}_'
      f'{tile_indices_str}_{filter}.fits')
  hdu = pyfits.PrimaryHDU(projected_image)
  filename.parent.mkdir(parents=True, exist_ok=True)
  hdu.writeto(str(filename), overwrite=True)
  print(f'### Wrote filename {filename}')
  return filename


@AstroFlowNode(cache_result=True, memory=_PLACE_IMAGE_MEMORY_REQUIREMENT)
def place_large_image(mode: QualityLevel, mosaic: Mosaic, stack: CompositeStack,
                      tile_indices: TileIndices, filter: str, viewbox: Viewbox,
                      filename: Path) -> Path:
  # if context.dangerously_accept_existing_files and os.path.exists(filename):
  #   print(f'### Dangerously skipping place_large_image because '
  #         f'file {filename} already exists')
  #   return filename

  wcs = _get_wcs(filename)
  interp = _make_interpolator(filename)
  tiling = _compute_tiling(mode, mosaic, viewbox)
  projected_image = _project_in_tiles(viewbox,
                                      tiling,
                                      interp,
                                      wcs,
                                      skip_empty=True)
  projected_image = trim_image(projected_image)

  return _save_image_to_file(mode, mosaic, stack, tile_indices, filter,
                             projected_image)


@AstroFlowNode(cache_result=True, debug=True)
def place_small_image(mode: QualityLevel, mosaic: Mosaic, stack: CompositeStack,
                      tile_indices: TileIndices, filter: str, viewbox: Viewbox,
                      filename: Path) -> Path:
  wcs = _get_wcs(filename)
  interp = _make_interpolator(filename)

  num_samples_x = 1500
  resolution = viewbox.x_span / num_samples_x
  num_samples_y = int(viewbox.y_span / resolution)

  tiling = Tiling(1, 1, num_samples_y, num_samples_x)

  projected_image = _project_in_tiles(viewbox,
                                      tiling,
                                      interp,
                                      wcs,
                                      skip_empty=False)

  return _save_image_to_file(mode, mosaic, stack, tile_indices, filter,
                             projected_image)
