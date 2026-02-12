from xisf import XISF  # type: ignore
from shapely.geometry import Polygon
from shapely import box  # type: ignore
from spherical_geometry.polygon import SphericalPolygon  # type: ignore
import numpy as np
import structlog
import ray
import cv2
from typing import Optional

from clustering import cluster_continuous
from geometry.polygons import project_polygon
from mosaics.types import Viewbox, Mask

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


@ray.remote
def process(
    polygon: Polygon,
    x1: float,
    x_clusters: list[float],
    y_clusters: list[float],
) -> tuple[Optional[Polygon], float]:
    max_area = 0.0
    best_rectangle = None

    for x_cluster in x_clusters:
        x2 = min(x_cluster)
        if x2 < x1:
            continue

        for i in range(len(y_clusters)):
            for j in range(i + 1, len(y_clusters)):
                y1, y2 = max(y_clusters[i]), min(y_clusters[j])
                candidate = box(x1, y1, x2, y2)

                if polygon.contains(candidate):
                    if candidate.area > max_area:
                        best_rectangle, max_area = candidate, candidate.area

    return best_rectangle, max_area


# TODO: This is brute force. It's tolerably slow, but can be improved.
def largest_interior_rectangle(polygon: Polygon, epsilon=20) -> box:
    x_clusters = cluster_continuous(
        [p[0] for p in polygon.exterior.coords],
        epsilon=epsilon,
        key=lambda x: x,
    )
    y_clusters = cluster_continuous(
        [p[1] for p in polygon.exterior.coords],
        epsilon=epsilon,
        key=lambda y: y,
    )

    tasks = [
        process.remote(polygon, max(x_cluster), x_clusters, y_clusters)
        for x_cluster in x_clusters
    ]
    results = ray.get(tasks)

    max_area = 0.0
    best_rectangle = None
    for r, a in results:
        if a > max_area:
            best_rectangle, max_area = r, a

    return best_rectangle


def trim_image(
    image: np.ndarray, mask: Mask, draw_outline: bool = True
) -> np.ndarray:
    x_min, y_min, x_max, y_max = mask.bounds
    x_min, y_min, x_max, y_max = map(int, (x_min, y_min, x_max, y_max))

    output_image = np.zeros_like(image)
    output_image[y_min:y_max, x_min:x_max] = image[y_min:y_max, x_min:x_max]

    if draw_outline:
        # Draw white outline (value 1.0) of the mask
        output_image[y_min, x_min:x_max] = 1.0  # Top edge
        output_image[y_max - 1, x_min:x_max] = 1.0  # Bottom edge
        output_image[y_min:y_max, x_min] = 1.0  # Left edge
        output_image[y_min:y_max, x_max - 1] = 1.0  # Right edge

    return output_image
