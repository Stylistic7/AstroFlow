import numpy as np

import mercator
import footprints
from context import context
from projection import compute_viewbox


def compute_mosaic_viewbox(mosaic, mode):
  # Determine the output image shape and resolution, which is a 3D to 2D
  # projection.
  proj = mercator.MercatorProjection()
  xs, ys = zip(
      *footprints.project_footprint(mosaic.footprint, proj).exterior.coords)
  x_min, y_min, x_span, y_span = compute_viewbox(list(zip(xs, ys)))

  # TODO: Put this in the UI somewhere.
  if mode == 'subframe':
    num_samples_x = context.subframe_mosaic_samples
  elif mode == 'master':
    num_samples_x = context.master_mosaic_samples

  resolution = x_span / num_samples_x
  num_samples_y = int(y_span / resolution)
  xx, yy = np.meshgrid(np.linspace(x_min, x_min + x_span, num_samples_x),
                       np.linspace(y_min, y_min + y_span, num_samples_y))
  ras, decs = proj.unproject(xx, yy)
  ras, decs = np.degrees(ras), np.degrees(decs)

  dest_width, dest_height = ras.shape
  dest_ys, dest_xs = np.meshgrid(range(dest_height), range(dest_width))
  assert np.min(dest_ys) == 0
  assert np.max(dest_ys) == dest_height - 1
  assert np.min(dest_xs) == 0
  assert np.max(dest_xs) == dest_width - 1
  assert dest_xs.shape == dest_ys.shape == ras.shape == decs.shape

  return dict(dest_xs=dest_xs, dest_ys=dest_ys, ras=ras, decs=decs)
