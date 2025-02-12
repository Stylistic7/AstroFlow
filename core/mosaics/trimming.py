from xisf import XISF  # type: ignore
from shapely.geometry import Polygon, box
from spherical_geometry.polygon import SphericalPolygon  # type: ignore
import numpy as np
import structlog
import ray
import cv2
from typing import Optional

from geometry.polygons import project_polygon
from mosaics.types import Viewbox

log = structlog.get_logger()


def draw_polygon(image: np.ndarray,
                 polygon: Polygon,
                 color: float = 1.0,
                 thickness: int = 2) -> np.ndarray:
  points = np.array(polygon.exterior.coords, dtype=np.int32)
  cv2.polylines(image, [points],
                isClosed=True,
                color=color,
                thickness=thickness)

  return image


def project_to_viewbox(polygon: SphericalPolygon, viewbox: Viewbox) -> Polygon:
  xs, ys = project_polygon(polygon, viewbox.projection)
  xs = [(x - viewbox.x_min) / viewbox.resolution for x in xs]
  ys = [(y - viewbox.y_min) / viewbox.resolution for y in ys]
  return Polygon(zip(xs, ys))


@ray.remote
def process(polygon: Polygon, x1: float, x_coords: list[float],
            y_coords: list[float]) -> tuple[Optional[box], float]:
  max_area = 0.0
  best_rectangle = None

  for x2 in x_coords:
    if x2 < x1:
      continue

    for i in range(len(y_coords)):
      for j in range(i + 1, len(y_coords)):
        y1, y2 = y_coords[i], y_coords[j]
        candidate = box(x1, y1, x2, y2)

        if polygon.contains(candidate):
          if candidate.area > max_area:
            best_rectangle, max_area = candidate, candidate.area

  return best_rectangle, max_area


# TODO: This is brute force. It's tolerably slow, but can be improved.
def largest_interior_rectangle(polygon: Polygon):
  x_coords = sorted(set([p[0] for p in polygon.exterior.coords]))
  y_coords = sorted(set([p[1] for p in polygon.exterior.coords]))

  tasks = [process.remote(polygon, x, x_coords, y_coords) for x in x_coords]
  results = ray.get(tasks)

  max_area = 0.0
  best_rectangle = None
  for r, a in results:
    if a > max_area:
      best_rectangle, max_area = r, a

  return best_rectangle


def trim_image(image: np.ndarray, box: box) -> np.ndarray:
  x_min, y_min, x_max, y_max = box.bounds
  x_min, y_min, x_max, y_max = map(int, (x_min, y_min, x_max, y_max))
  output_image = np.zeros_like(image)
  output_image[y_min:y_max, x_min:x_max] = image[y_min:y_max, x_min:x_max]

  return output_image
