import ray
import numpy as np

from mosaics.types import Mosaic, Viewbox
from artifacts import Artifact
from geometry.projections import mercator
from geometry.viewbox import compute_viewbox
from geometry.polygons import project_polygon
from previews import QualityLevel

_TILE_SIZE = 2048  # pixels


# TODO: The viewbox should include the sampling resolution, too.
# TODO: Maybe unify the storage and retrieval of artifacts that arent'
# images, e.g. arbitrary Python objects.
def compute_mosaic_viewbox(mode: QualityLevel,
                           mosaic: Mosaic) -> list[Artifact]:
  print(f'### Entering compute_mosaic_viewbox for mosaic {mosaic} mode {mode}')

  def round_to_nearest_multiple(number: float, multiple: int) -> int:
    return int(np.ceil(number / multiple) * multiple)

  # TODO: Put this in the UI somewhere.
  projection = mercator.MercatorProjection()

  # Project from spherical coordinates on the sky into 2D coordinates, in
  # radians. Compute the min and span in each dimension.
  xs, ys = project_polygon(mosaic.footprint, projection)
  x_min, y_min, x_span, y_span = compute_viewbox(list(zip(xs, ys)))

  # Compute the number of pixels along the x axis. In master mode, use the
  # mosaic's image scale, given in degrees per pixel. In subframe mode, make
  # a thumbnail-sized image.
  num_pixels_x = x_span / np.deg2rad(
      mosaic.image_scale / 3600) if mode == QualityLevel.MASTER else 2048
  finished_pixels_x = int(np.ceil(num_pixels_x))

  # Round the number of pixels up to a multiple of the tile size.
  num_pixels_x = round_to_nearest_multiple(num_pixels_x, multiple=_TILE_SIZE)

  # Compute the effective resolution, with the rounded-up number of pixels.
  resolution = x_span / num_pixels_x

  # Compute the number of pixels in the y-axis, using the same resolution.
  num_pixels_y = y_span / resolution
  finished_pixels_y = int(np.ceil(num_pixels_y))
  num_pixels_y = round_to_nearest_multiple(num_pixels_y, multiple=_TILE_SIZE)

  num_tiles_y = int(np.ceil(num_pixels_y / _TILE_SIZE))
  num_tiles_x = int(np.ceil(num_pixels_x / _TILE_SIZE))

  print(f'### viewbox x_min {x_min} x_span {x_span} '
        f'y_min {y_min} y_span {y_span} resolution {resolution} '
        f'num_pixels_x {num_pixels_x}')

  return Viewbox(projection=projection,
                 x_min=x_min,
                 x_span=x_span,
                 y_min=y_min,
                 y_span=y_span,
                 resolution=resolution,
                 num_pixels_y=finished_pixels_y,
                 num_pixels_x=finished_pixels_x,
                 num_tiles_y=num_tiles_y,
                 num_tiles_x=num_tiles_x,
                 tile_height=_TILE_SIZE,
                 tile_width=_TILE_SIZE)
