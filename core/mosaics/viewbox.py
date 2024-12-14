import ray
import numpy as np
from spherical_geometry import great_circle_arc, vector

from .classes import Viewbox
from artifacts import CacheArtifact, Artifact
from artifacts.decorators import saves_artifacts
from geometry.projections import mercator
import geometry.footprints as footprints
from context import context
from geometry.viewbox import compute_viewbox
from geometry.polygons import project_polygon
from hashing import hash_list, hash_string
from mosaics.mosaic import Mosaic
from previews import QualityLevel
from mosaics.types import MosaicHash
from dependency_graph.astroflow_node import AstroFlowNode


def _compute_viewbox(mosaic: Mosaic) -> Viewbox:
  # TODO: Put this in the UI somewhere.
  proj = mercator.MercatorProjection()

  xs, ys = project_polygon(mosaic.footprint, proj)
  x_min, y_min, x_span, y_span = compute_viewbox(list(zip(xs, ys)))

  return Viewbox(projection=proj,
                 x_min=x_min,
                 y_min=y_min,
                 x_span=x_span,
                 y_span=y_span)


# def compute_mosaic_viewbox_artifacts(mode: QualityLevel,
#                                      mosaic_id: MosaicHash) -> list[Artifact]:
#   mosaic = context.mosaics[mosaic_id]
#   viewbox = _compute_viewbox(mosaic)
#   content_hash = hash_list([mosaic.content_hash, hash_string(repr(viewbox))])
#   return [CacheArtifact(('mosaic_viewbox', mode, mosaic_id), content_hash)]


# TODO: The viewbox should include the sampling resolution, too.
# TODO: Maybe unify the storage and retrieval of artifacts that arent'
# images, e.g. arbitrary Python objects.
# @ray.remote
# @saves_artifacts(compute_mosaic_viewbox_artifacts)
@AstroFlowNode()
def compute_mosaic_viewbox(mode: QualityLevel,
                           mosaic_id: MosaicHash) -> list[Artifact]:
  print(f'### Entering compute_mosaic_viewbox for mosaic_id {mosaic_id}')
  mosaic = context.mosaics[mosaic_id]
  # artifacts = compute_mosaic_viewbox_artifacts(mode, mosaic_id)
  # artifact = next(iter(artifacts))
  # assert isinstance(artifact, CacheArtifact)
  # key = artifact.key
  # context[key] = _compute_viewbox(mosaic)
  # print(f'### Wrote key {key}')
  # return artifacts
  return _compute_viewbox(mosaic)
