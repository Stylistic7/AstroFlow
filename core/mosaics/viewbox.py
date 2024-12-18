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


# TODO: The viewbox should include the sampling resolution, too.
# TODO: Maybe unify the storage and retrieval of artifacts that arent'
# images, e.g. arbitrary Python objects.
def compute_mosaic_viewbox(mosaic: Mosaic) -> list[Artifact]:
  print(f'### Entering compute_mosaic_viewbox for mosaic {Mosaic}')

  # TODO: Put this in the UI somewhere.
  proj = mercator.MercatorProjection()

  xs, ys = project_polygon(mosaic.footprint, proj)
  x_min, y_min, x_span, y_span = compute_viewbox(list(zip(xs, ys)))

  return Viewbox(projection=proj,
                 x_min=x_min,
                 y_min=y_min,
                 x_span=x_span,
                 y_span=y_span)
