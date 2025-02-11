from xisf import XISF  # type: ignore
from shapely.geometry import Polygon, box
from spherical_geometry.polygon import SphericalPolygon  # type: ignore
from pathlib import Path
import numpy as np
import structlog
import ray
from itertools import combinations
from typing import Optional, Tuple
import cv2
from functools import partial

from dependency_graph.astroflow_node import AstroFlowNode
from geometry.polygons import project_polygon
from mosaics.types import Viewbox

log = structlog.get_logger()

_TRIM_IMAGE_MEMORY_REQUIREMENT = 4 * 1024 * 1024 * 1024  # 4 GB


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


@AstroFlowNode(memory=_TRIM_IMAGE_MEMORY_REQUIREMENT)
def trim_xisf_image(polygon: Polygon, filename: Path) -> Path:
  box = largest_interior_rectangle(polygon)

  xisf = XISF(filename)
  file_meta = xisf.get_file_metadata()
  file_meta
  ims_meta = xisf.get_images_metadata()
  ims_meta
  im_data = xisf.read_image(0)
  im_data = np.expand_dims(trim_image(np.squeeze(im_data), box), -1)
  XISF.write(filename,
             im_data,
             creator_app="AstroFlow",
             image_metadata=ims_meta[0],
             xisf_metadata=file_meta)
  return filename


def project_to_viewbox(polygon: SphericalPolygon, viewbox: Viewbox) -> Polygon:
  xs, ys = project_polygon(polygon, viewbox.projection)
  xs = [(x - viewbox.x_min) / viewbox.resolution for x in xs]
  ys = [(y - viewbox.y_min) / viewbox.resolution for y in ys]
  return Polygon(zip(xs, ys))


# @ray.remote
# def check_rectangle(polygon: Polygon, p1: Tuple[float, float],
#                     p2: Tuple[float, float]) -> Tuple[Optional[box], float]:
#   """Check if the axis-aligned rectangle formed by two points is inside the polygon."""
#   xmin, xmax = min(p1[0], p2[0]), max(p1[0], p2[0])
#   ymin, ymax = min(p1[1], p2[1]), max(p1[1], p2[1])

#   rect = box(xmin, ymin, xmax, ymax)
#   if polygon.contains(rect):
#     return rect, rect.area
#   return None, 0.0

# def largest_interior_rectangle(polygon: Polygon) -> Optional[box]:
#   max_area: float = 0.0
#   best_rectangle: Optional[box] = None

#   expanded = polygon.buffer(1e-3)
#   vertices: list[Tuple[float, float]] = list(polygon.exterior.coords)
#   tasks = [
#       check_rectangle.remote(expanded, p1, p2)
#       for p1, p2 in combinations(vertices, 2)
#   ]

#   results: list[Tuple[Optional[box], float]] = ray.get(tasks)

#   for rect, area in results:
#     if rect and area > max_area:
#       max_area = area
#       best_rectangle = rect

#   return best_rectangle


@ray.remote
def compute_batch(entries, func):
  """Computes func(entry) for a batch of entries and returns the best result."""
  max_entry, max_value = None, float('-inf')
  for entry in entries:
    result, value = func(entry)
    if value > max_value:
      max_entry, max_value = result, value
  return max_entry, max_value


def find_max(generator, func, batch_size=1000, num_returns=50):
  """Finds the entry in a generator that maximizes a function using Ray with batch processing."""
  max_entry, max_value = None, float('-inf')
  futures = []
  batch = []

  for entry in generator:
    batch.append(entry)
    if len(batch) >= batch_size:
      futures.append(compute_batch.remote(batch, func))
      batch = []  # Start a new batch

    if len(futures) >= num_returns:
      done, futures = ray.wait(futures, num_returns=num_returns)
      for entry, value in ray.get(done):
        if value > max_value:
          max_entry, max_value = entry, value

  # Process any remaining batch
  if batch:
    futures.append(compute_batch.remote(batch, func))

  # Process any remaining results
  for entry, value in ray.get(futures):
    if value > max_value:
      max_entry, max_value = entry, value

  return max_entry, max_value


def generate_boxes(polygon: Polygon):
  # Extract unique x and y coordinates from polygon vertices
  x_coords = sorted(set([p[0] for p in polygon.exterior.coords]))
  y_coords = sorted(set([p[1] for p in polygon.exterior.coords]))

  # Iterate over all possible rectangle pairs (x1, x2) and (y1, y2)
  for i in range(len(x_coords)):
    for j in range(i + 1, len(x_coords)):
      for k in range(len(y_coords)):
        for l in range(k + 1, len(y_coords)):
          x1, x2 = x_coords[i], x_coords[j]
          y1, y2 = y_coords[k], y_coords[l]
          yield box(x1, y1, x2, y2)


def check_rectangle(polygon: Polygon, box: box) -> float:
  if polygon.contains(box):
    return box, box.area
  else:
    return None, 0.0


def largest_interior_rectangle(polygon: Polygon) -> Optional[box]:
  fn = partial(check_rectangle, polygon)
  max_entry, _ = find_max(generate_boxes(polygon), fn)
  return max_entry


def trim_image(image: np.ndarray, box: box) -> np.ndarray:
  x_min, y_min, x_max, y_max = box.bounds
  x_min, y_min, x_max, y_max = map(int, (x_min, y_min, x_max, y_max))
  output_image = np.zeros_like(image)
  output_image[y_min:y_max, x_min:x_max] = image[y_min:y_max, x_min:x_max]

  return output_image
