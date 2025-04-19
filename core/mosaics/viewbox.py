import ray
import numpy as np

from attributes import AstrometricAttributes
from mosaics.types import Viewbox
from geometry.projections import mercator
# TODO: Rename or unify geometry_compute_viewbox
from geometry.viewbox import compute_viewbox as geometry_compute_viewbox
from geometry.polygons import project_polygon
from previews.types import QualityLevel

_TILE_SIZE = 2048  # pixels


def compute_viewbox(mode: QualityLevel,
                    object: AstrometricAttributes) -> Viewbox:
  print(f'### Entering compute_viewbox for object {object} mode {mode}')

  def round_to_nearest_multiple(number: float, multiple: int) -> int:
    return int(np.ceil(number / multiple) * multiple)

  # TODO: Put this in the UI somewhere.
  projection = mercator.MercatorProjection()

  # Project from spherical coordinates on the sky into 2D coordinates, in
  # radians. Compute the min and span in each dimension.
  xs, ys = project_polygon(object.footprint, projection)
  x_min, y_min, x_span, y_span = geometry_compute_viewbox(list(zip(xs, ys)))

  # Compute the number of pixels along the x axis. In master mode, use the
  # object's image scale, given in degrees per pixel. In subframe mode, make
  # a thumbnail-sized image.
  # TODO: This needs to be updated for other levels like post-processing.
  num_pixels_x = x_span / np.deg2rad(
      object.image_scale / 3600) if mode == QualityLevel.MASTER else 2048
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
