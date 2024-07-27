import ray
import numpy as np

from .classes import Viewbox
from artifact import CacheArtifact
from decorators import saves_artifacts
from geometry.projections import mercator
import geometry.footprints as footprints
from context import context
from geometry.viewbox import compute_viewbox


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

  viewbox = Viewbox(projection=proj,
                    x_min=x_min,
                    y_min=y_min,
                    x_span=x_span,
                    y_span=y_span)

  key = next(iter(compute_mosaic_viewbox_artifacts(mode, mosaic_id))).key
  context[key] = viewbox
  print(f'### Wrote key {key}')
