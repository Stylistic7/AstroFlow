import math
import numpy as np
import ray
from typing import Callable, Any, TypeVar

import metadata.image_types as image_types
from geometry import footprints
import plate_solving
from clustering.continuous import cluster_continuous
from decorators import node
from cache.context_ref import ContextRef
from metadata.classes import MetadataDict
import structlog

log = structlog.get_logger()

T = TypeVar("T")
ComparisonFn = Callable[[T, T], bool]


def find_groups(
    values: dict[T, Any],
    comparison_fn: ComparisonFn,
    min_component_members: int,
) -> list[list[T]]:
    print(f"Finding groups among {len(values)} values")
    ids = values.keys()
    groups: list[list[T]] = []
    last_printed_percent_done: float = 0
    for i, id in enumerate(ids):
        found = False
        value = values[id]

        for group in groups:
            group_value = values[group[0]]

            if comparison_fn(value, group_value):
                group.append(id)
                found = True
                break

        if not found:
            groups.append([id])

        percent_done = 100.0 * i / len(ids)
        percent_done = 10.0 * round(percent_done / 10.0)
        if percent_done > last_printed_percent_done:
            print(f"Finding groups is {percent_done:.0f}% done.")
            last_printed_percent_done = percent_done

    num_groups_of_all_sizes = len(groups)
    groups = [t for t in groups if len(t) >= min_component_members]

    print(
        f"found {len(groups)} groups, discarded "
        f"{num_groups_of_all_sizes - len(groups)} groups that were too small"
    )
    return groups


def _min_fov_deg(metadata: MetadataDict, frame_hash: str) -> float:
    """Minimum FOV dimension in degrees for a plate-solved frame."""
    m = metadata[frame_hash]
    return m.plate_solution.image_scale * min(m.width_pixels, m.height_pixels) / 3600.0


def pointing(metadata: MetadataDict, frame_hash: str, overlap_fraction: float) -> tuple:
    """The pointing of one solved image for `same_pointing`: its centre as a
    unit vector and the cosine of the offset it tolerates — a fraction
    `1 - overlap_fraction` of its own narrower field, at most 2 degrees.
    One definition for raw frames and for masters (`stacks.foreign`)."""
    m = metadata[frame_hash]
    max_offset_deg = (1.0 - overlap_fraction) * _min_fov_deg(metadata, frame_hash) / 2.0
    return (np.asarray(footprints.radec_to_vector(*m.plate_solution.center)),
            math.cos(math.radians(min(max_offset_deg, 2.0))))


def same_pointing(a: tuple, b: tuple) -> bool:
    """Two `pointing`s frame the same field: closer than the stricter of
    their tolerances."""
    (va, ta), (vb, tb) = a, b
    return bool(np.dot(va, vb) > max(ta, tb))


@node
def cluster_by_image_center(
    metadata_ref: ContextRef[MetadataDict],
    overlap_fraction: float,
    min_component_members: int,
) -> list[tuple[str, ...]]:
    """Cluster light frames by sky position, adaptive to each sensor and
    focal length.

    Frames are partitioned by sensor geometry first — different pixel counts
    mean different cameras, whose frames can never share a stack, so they are
    never compared for grouping at all. (Before this rule, one small-chip
    frame in a same-arcsec/px group shrank the offset budget for every
    full-frame pair: a 5.5 arcmin meridian-flip re-center split one pointing
    into two stacks.) Within a sensor group, frames split by image scale
    (proxy for focal length), then cluster spatially; each frame carries a
    cosine threshold from its own FOV and a pair must satisfy the stricter.
    """
    metadata = metadata_ref.get()

    plate_solved = {
        h: i
        for h, i in metadata.items()
        if image_types.is_light_frame(i.header)
        and not image_types.is_master(i.header)
        and type(i.plate_solution) is plate_solving.PlateSolution
    }

    by_sensor: dict[tuple[int, int], list[str]] = {}
    for h, i in plate_solved.items():
        by_sensor.setdefault((i.width_pixels, i.height_pixels), []).append(h)
    log.info(
        "Sensor groups",
        sizes={f"{w}x{h}": len(hs) for (w, h), hs in sorted(by_sensor.items())},
    )

    all_spatial_groups: list[tuple[str, ...]] = []
    for dims, sensor_hashes in sorted(by_sensor.items()):
        scale_items = [
            (h, plate_solved[h].plate_solution.image_scale) for h in sensor_hashes
        ]
        scale_groups = cluster_continuous(scale_items, epsilon=0.2)

        for scale_group in scale_groups:
            frame_hashes = [h for h, _ in scale_group]
            values = {h: pointing(metadata, h, overlap_fraction) for h in frame_hashes}

            log.info(
                "Spatial clustering",
                sensor=dims,
                num_frames=len(frame_hashes),
                min_fov_deg=round(min(_min_fov_deg(metadata, h) for h in frame_hashes), 3),
            )

            groups = find_groups(values, same_pointing, min_component_members)
            all_spatial_groups.extend(tuple(sorted(g)) for g in groups)

    log.info("Total spatial groups", num_groups=len(all_spatial_groups))
    return all_spatial_groups
