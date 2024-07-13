import ray
import numpy as np

from artifact import CacheArtifact
from decorators import saves_artifacts
import mercator
import footprints
from context import context
from projection import compute_viewbox

# def compute_viewbox(plate_solutions):
#   ras = [f[0] for p in plate_solutions for f in p.footprint]
#   decs = [f[1] for p in plate_solutions for f in p.footprint]
#   # print(f'got {len(ras)} ras and {len(decs)} decs')

#   min_ra, max_ra = np.min(ras), np.max(ras)
#   min_dec, max_dec = np.min(decs), np.max(decs)

#   return [min_ra, min_dec, max_ra - min_ra, max_dec - min_dec]


def compute_mosaic_viewbox_artifacts(mode, mosaic_id):
  mosaic = context.mosaics[mosaic_id]
  return [
      CacheArtifact(('mosaic_viewbox', mode, mosaic_id, mosaic.content_hash))
  ]


@ray.remote
@saves_artifacts(compute_mosaic_viewbox_artifacts)
def compute_mosaic_viewbox(mode, mosaic_id):
  print(f'### Entering compute_mosaic_viewbox for mosaic_id {mosaic_id}')
  mosaic = context.mosaics[mosaic_id]

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

  viewbox = dict(dest_xs=dest_xs, dest_ys=dest_ys, ras=ras, decs=decs)

  # TODO: This is sort of abusing the Artifact system. Maybe there can be
  # another kind of Artifact that is free-form (not FITS image).
  key = next(iter(compute_mosaic_viewbox_artifacts(mode, mosaic_id))).key
  context[key] = viewbox
  print(f'### Wrote key {key}')
