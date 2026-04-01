import math
import numpy as np
import ray
from typing import Sequence, Callable, Any, TypeVar

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


@node
def cluster_by_image_center(
    metadata_ref: ContextRef[MetadataDict],
    overlap_fraction: float,
    min_component_members: int,
) -> list[tuple[str, ...]]:
    """Cluster light frames by sky position, adaptive to each focal length.

    First groups frames by image scale (proxy for focal length), then
    spatially clusters within each scale group using a cosine similarity
    threshold derived from the group's FOV and the requested overlap fraction.
    """
    metadata = metadata_ref.get()

    plate_solved = {
        h: i
        for h, i in metadata.items()
        if image_types.is_light_frame(i.header)
        and not image_types.is_master(i.header)
        and type(i.plate_solution) is plate_solving.PlateSolution
    }

    # Split by image scale first so each spatial group has uniform FOV.
    scale_items = [(h, i.plate_solution.image_scale) for h, i in plate_solved.items()]
    scale_groups = cluster_continuous(scale_items, epsilon=0.2)
    log.info("Scale groups", num_groups=len(scale_groups), sizes=[len(g) for g in scale_groups])

    def get_centroid(center: Sequence[float]) -> np.ndarray:
        return np.asarray(footprints.radec_to_vector(*center))

    all_spatial_groups: list[tuple[str, ...]] = []
    for scale_group in scale_groups:
        frame_hashes = [h for h, _ in scale_group]
        values = {h: get_centroid(plate_solved[h].plate_solution.center) for h in frame_hashes}

        min_fov = _min_fov_deg(metadata, frame_hashes[0])
        max_offset_deg = (1.0 - overlap_fraction) * min_fov / 2.0
        max_offset_deg = min(max_offset_deg, 2.0)
        threshold = math.cos(math.radians(max_offset_deg))

        log.info(
            "Spatial clustering for scale group",
            num_frames=len(frame_hashes),
            min_fov_deg=round(min_fov, 3),
            max_offset_deg=round(max_offset_deg, 4),
            threshold=round(threshold, 8),
        )

        def comparison_fn(a, b, t=threshold):
            return bool(np.dot(a, b) > t)

        groups = find_groups(values, comparison_fn, min_component_members)
        all_spatial_groups.extend(tuple(sorted(g)) for g in groups)

    log.info("Total spatial groups", num_groups=len(all_spatial_groups))
    return all_spatial_groups
