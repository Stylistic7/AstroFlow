import ray
import numpy as np

from .classes import Viewbox
from artifact import CacheArtifact
from decorators import saves_artifacts
from geometry.projections import mercator
import geometry.footprints as footprints
from context import context
from geometry.viewbox import compute_viewbox
from hashing import hash_list, hash_string


def _compute_viewbox(mosaic):
  # TODO: Put this in the UI somewhere.
  proj = mercator.MercatorProjection()

  xs, ys = zip(
      *footprints.project_footprint(mosaic.footprint, proj).exterior.coords)
  x_min, y_min, x_span, y_span = compute_viewbox(list(zip(xs, ys)))

  return Viewbox(projection=proj,
                 x_min=x_min,
                 y_min=y_min,
                 x_span=x_span,
                 y_span=y_span)


def compute_mosaic_viewbox_artifacts(mode, mosaic_id):
  mosaic = context.mosaics[mosaic_id]
  viewbox = _compute_viewbox(mosaic)
  content_hash = hash_list([mosaic.content_hash, hash_string(repr(viewbox))])
  return [CacheArtifact(('mosaic_viewbox', mode, mosaic_id), content_hash)]


# TODO: The viewbox should include the sampling resolution, too.
# TODO: Maybe unify the storage and retrieval of artifacts that arent'
# images, e.g. arbitrary Python objects.
@ray.remote
@saves_artifacts(compute_mosaic_viewbox_artifacts)
def compute_mosaic_viewbox(mode, mosaic_id):
  print(f'### Entering compute_mosaic_viewbox for mosaic_id {mosaic_id}')
  mosaic = context.mosaics[mosaic_id]
  artifacts = compute_mosaic_viewbox_artifacts(mode, mosaic_id)
  key = next(iter(artifacts)).key
  context[key] = _compute_viewbox(mosaic)
  print(f'### Wrote key {key}')
  return artifacts
