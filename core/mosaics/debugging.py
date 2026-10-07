import matplotlib.pyplot as plt
import numpy as np
from spherical_geometry.polygon import SphericalPolygon  # type: ignore
from shapely import Polygon  # type: ignore

from context import context
from targets.types import TargetHash, TargetsDict
from mosaics.types import Mosaic
from geometry.projection import Projection
from geometry.polygons import project_polygon
from geometry.projections.mercator import MercatorProjection


def draw_polygon(polygon: Polygon, color: str = "black") -> None:
    x, y = zip(*polygon.exterior.coords)
    plt.plot(x, y, color=color)


def draw_spherical_polygon(
    polygon: SphericalPolygon, projection: Projection, color: str = "black"
) -> None:
    x, y = project_polygon(polygon, projection)
    plt.plot(x, y, color=color)


def plot_targets_overlap(a, b, title: str = "Targets Overlap") -> None:
    plt.figure()
    plt.title(title)
    plt.axis("equal")

    projection = MercatorProjection()

    def plot_target(target) -> None:
        ra, dec = (np.radians(np.array(x)) for x in target.center)
        center = projection.project(ra, dec)

        draw_spherical_polygon(target.footprint, projection, color="blue")
        plt.scatter([center[0]], [center[1]], color="black")

    plot_target(a)
    plot_target(b)

    plt.show()


def plot_mosaic(targets: TargetsDict, mosaic: Mosaic, title: str = "Mosaic") -> None:
    plot_targets_and_footprint(targets, mosaic.target_ids, mosaic.footprint, title)


def plot_targets_and_footprint(
    targets: TargetsDict,
    target_ids: list[TargetHash],
    footprint: SphericalPolygon | None,
    title: str = "Mosaic",
) -> None:
    import ray
    from mosaics.identification import targets_overlap

    plt.figure()
    plt.title(title)
    plt.axis("equal")

    projection = MercatorProjection()

    for i, target_id in enumerate(target_ids):
        target = targets[target_id]

        ra, dec = (np.radians(np.array(x)) for x in target.center)
        center = projection.project(ra, dec)

        print(f"### center is {center}")

        overlapping_targets = [
            tid
            for tid in target_ids
            if ray.get(targets_overlap.remote(targets[target_id], targets[tid]))
        ]
        print(
            f"### overlapping_targets for target {target_id} are {overlapping_targets}"
        )

        draw_spherical_polygon(target.footprint, projection, color="blue")
        plt.text(
            center[0],
            center[1],
            f"{target_id}: {len(overlapping_targets)}",
            fontsize=12,
            color="blue",
        )

        plt.scatter([center[0]], [center[1]], color="black")

    if footprint is not None:
        draw_spherical_polygon(footprint, projection, color="red")

    plt.show()
