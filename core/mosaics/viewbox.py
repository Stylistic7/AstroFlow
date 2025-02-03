import ray
import numpy as np

from mosaics.types import Mosaic, Viewbox
from artifacts import Artifact
from geometry.projections import mercator
from geometry.viewbox import compute_viewbox
from geometry.polygons import project_polygon
from previews import QualityLevel


# TODO: The viewbox should include the sampling resolution, too.
# TODO: Maybe unify the storage and retrieval of artifacts that arent'
# images, e.g. arbitrary Python objects.
def compute_mosaic_viewbox(mode: QualityLevel,
                           mosaic: Mosaic) -> list[Artifact]:
  print(f'### Entering compute_mosaic_viewbox for mosaic {mosaic}')

  # TODO: Put this in the UI somewhere.
  proj = mercator.MercatorProjection()

  xs, ys = project_polygon(mosaic.footprint, proj)
  x_min, y_min, x_span, y_span = compute_viewbox(list(zip(xs, ys)))

  return Viewbox(projection=proj,
                 x_min=x_min,
                 y_min=y_min,
                 x_span=x_span,
                 y_span=y_span)
