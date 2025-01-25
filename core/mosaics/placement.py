import ray
import os
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

from file_io.generic import get_content_hash, read_image
from context import context
from previews import QualityLevel
from mosaics.types import Mosaic, Viewbox, TileIndices
from dependency_graph.astroflow_node import AstroFlowNode

_MEMORY_REQUIREMENT = 8 * 1024 * 1024 * 1024  # 8 GB

log = structlog.get_logger()


def _compute_ras_decs(mode: QualityLevel,
                      viewbox: Viewbox,
                      multiple: int = 4) -> tuple[np.ndarray, np.ndarray]:
  if mode == QualityLevel.SUBFRAME:
    num_samples_x = context.subframe_mosaic_samples
  elif mode == QualityLevel.MASTER:
    num_samples_x = context.master_mosaic_samples

  def round_to_nearest_multiple(number: float) -> int:
    return int(np.ceil(number / multiple) * multiple)

  num_samples_x = round_to_nearest_multiple(num_samples_x)
  resolution = viewbox.x_span / num_samples_x
  num_samples_y = round_to_nearest_multiple(viewbox.y_span / resolution)

  xx, yy = np.meshgrid(
      np.linspace(viewbox.x_min, viewbox.x_min + viewbox.x_span, num_samples_x),
      np.linspace(viewbox.y_min, viewbox.y_min + viewbox.y_span, num_samples_y))
  ras, decs = viewbox.projection.unproject(xx, yy)
  ras, decs = np.degrees(ras), np.degrees(decs)

  return ras, decs


def _reshape_tiles(
    image: np.ndarray, tile_size: tuple[int, int] = (32, 32)) -> np.ndarray:
  image_height, image_width = image.shape
  tile_height, tile_width = tile_size
  tiles = image.reshape(image_height // tile_height, tile_height,
                        image_width // tile_width, tile_width)
  return tiles.swapaxes(1, 2)


@ray.remote(scheduling_strategy="SPREAD")
def place_tile(tile_y: int, tile_x: int, wcs: Any, ra_tiles: np.ndarray,
               dec_tiles: np.ndarray,
               interp: RegularGridInterpolator) -> tuple[int, int, np.ndarray]:
  ra_tile = ra_tiles[tile_y, tile_x, :, :]
  dec_tile = dec_tiles[tile_y, tile_x, :, :]

  source_xs, source_ys = wcs.all_world2pix(ra_tile, dec_tile, 1, quiet=True)
  return tile_y, tile_x, interp((source_ys, source_xs))


def _trim_image(projected_image):
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


# TODO: Provide some feedback on progress.
#
# TODO: Placement is perhaps fast enough to not bother storing placed images
# on disk, where they take up a huge amount of space.
#
# TODO: Differentiate the runtime of this function in subframe and master
# modes.
#
# TODO: Perhaps don't save these results, since they are so huge, and
# are now relatively fast to compute.
@AstroFlowNode(memory=_MEMORY_REQUIREMENT)
def place_image(mode: QualityLevel, mosaic: Mosaic, tile_indices: TileIndices,
                viewbox: Viewbox, filename: Path) -> np.ndarray | Path:
  # TODO: The subframes need to be calibrated before being placed.
  print(f'### place_image: filename {filename}')
  image = read_image(filename)
  content_hash = get_content_hash(filename)
  plate_solution = context.metadata[content_hash].plate_solution

  log.info(f'place_image: image shape {image.shape} dtype {image.dtype} '
           f'min {np.min(image)} max {np.max(image)}')

  wcs = WCS(plate_solution.header)

  # TODO: This seems to take up a huge amount of memory with method 'cubic.'
  # Maybe tiling would reduce the intermediate memory usage.
  source_height, source_width = image.shape
  interp = RegularGridInterpolator((range(source_height), range(source_width)),
                                   image,
                                   method='linear',
                                   bounds_error=False,
                                   fill_value=0)
  del image

  multiple = 4

  ras, decs = _compute_ras_decs(mode, viewbox, multiple)
  projected_image = np.zeros_like(ras)

  viewbox_height, viewbox_width = ras.shape
  tile_height = int(viewbox_height / multiple)
  tile_width = int(viewbox_width / multiple)
  ra_tiles = _reshape_tiles(ras, (tile_height, tile_width))
  dec_tiles = _reshape_tiles(decs, (tile_height, tile_width))
  projected_image_tiles = _reshape_tiles(projected_image,
                                         (tile_height, tile_width))

  num_tiles_y, num_tiles_x = ra_tiles.shape[0:2]
  wcs_ref = ray.put(wcs)
  ra_tiles_ref = ray.put(ra_tiles)
  dec_tiles_ref = ray.put(dec_tiles)
  interp_ref = ray.put(interp)
  remaining_refs = [
      place_tile.remote(tile_y, tile_x, wcs_ref, ra_tiles_ref, dec_tiles_ref,
                        interp_ref)
      for tile_y, tile_x in product(range(num_tiles_y), range(num_tiles_x))
  ]
  while remaining_refs:
    ready_refs, remaining_refs = ray.wait(remaining_refs, num_returns=1)
    tile_y, tile_x, value = ray.get(ready_refs[0])
    projected_image_tiles[tile_y, tile_x, :, :] = value

  log.info(f'### projected_image shape {projected_image.shape}')
  log.info(
      f'### projected_image count nonzero {np.count_nonzero(projected_image)}')

  # TODO: Don't keep two copies of the huge image
  # projected_image = _trim_image(projected_image)

  if mode == QualityLevel.SUBFRAME:
    return projected_image
  elif mode == QualityLevel.MASTER:
    tile_indices_str = '_'.join(str(i) for i in tile_indices)
    filename = Path(context.output_dir) / 'Mosaics' / mosaic.name / Path(
        f'placed_image_{mosaic.stable_hash}_{tile_indices_str}.fits')
    hdu = pyfits.PrimaryHDU(projected_image)
    filename.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(str(filename), overwrite=True)
    print(f'### Wrote filename {filename}')
    return filename
