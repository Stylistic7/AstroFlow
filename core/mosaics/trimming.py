from shapely.geometry import Polygon
from spherical_geometry.polygon import SphericalPolygon  # type: ignore
import numpy as np
import structlog
import cv2

from geometry.polygons import project_polygon
from mosaics.types import Viewbox

log = structlog.get_logger()


def draw_polygon(
    image: np.ndarray, polygon: Polygon, color: float = 1.0, thickness: int = 2
) -> np.ndarray:
    points = np.array(polygon.exterior.coords, dtype=np.int32)
    cv2.polylines(
        image, [points], isClosed=True, color=color, thickness=thickness
    )

    return image


def project_to_viewbox(polygon: SphericalPolygon, viewbox: Viewbox) -> Polygon:
    xs, ys = project_polygon(polygon, viewbox.projection)
    xs = [(x - viewbox.x_min) / viewbox.resolution for x in xs]
    ys = [(y - viewbox.y_min) / viewbox.resolution for y in ys]
    return Polygon(zip(xs, ys))
