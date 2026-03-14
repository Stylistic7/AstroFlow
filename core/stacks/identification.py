import ray
import numpy as np
from beartype import beartype

from context import context
from typing import Iterator
import clustering
from stacks.composite_stack import CompositeStack, CompositeStacksDict
from metadata.classes import FrameHash, MetadataDict
from decorators import cached
from hashing.types import LargeObject


@cached.remote
@beartype
def identify_stacks(
    metadata_ref: LargeObject,
    angular_tolerance: float,
    light_frames: tuple[FrameHash, ...],
) -> CompositeStacksDict:
    print(
        f"### identify_stacks running for spatial group of length {len(light_frames)}"
    )
    metadata_in = metadata_ref.get()

    # TODO: Maybe implement a version of cluster_continuous() that repeatedly
    # subdivides by independent 1D variables.
    metadata = MetadataDict(
        {h: i for h, i in metadata_in.items() if h in light_frames}
    )

    @beartype
    def scale_clusters() -> Iterator[tuple[FrameHash, ...]]:
        scales = [
            (metadata[k].content_hash, metadata[k].plate_solution.image_scale)
            for k in metadata
        ]
        for cluster in clustering.cluster_continuous(scales, epsilon=0.2):
            scale_frames, _ = zip(*cluster)
            yield scale_frames

    @beartype
    def camera_clusters(
        frames: tuple[FrameHash, ...],
    ) -> Iterator[tuple[FrameHash, ...]]:
        cameras = [
            (
                metadata[k].content_hash,
                (
                    metadata[k].header["INSTRUME"],
                    metadata[k].header["NAXIS1"],
                    metadata[k].header["NAXIS2"],
                    metadata[k].header["XPIXSZ"],
                    metadata[k].header["YPIXSZ"],
                ),
            )
            for k in frames
        ]
        for cluster in clustering.cluster_discrete(cameras):
            frames, cameras = zip(*cluster)
            assert len(set(cameras)) == 1
            yield frames

    @beartype
    def rotation_clusters(
        frames: tuple[FrameHash, ...],
    ) -> Iterator[tuple[FrameHash, ...]]:
        rotation_angles = [
            (
                metadata[k].content_hash,
                metadata[k].plate_solution.rotation_angle,
            )
            for k in frames
        ]
        for cluster in clustering.cluster_angles_mod_pi(
            rotation_angles, angular_tolerance
        ):
            frames, _ = zip(*cluster)
            yield frames

    @beartype
    def filter_clusters(
        frames: tuple[FrameHash, ...],
    ) -> Iterator[tuple[FrameHash, ...]]:
        filters = [
            (metadata[k].content_hash, metadata[k].header["FILTER"])
            for k in frames
        ]
        for cluster in clustering.cluster_discrete(filters):
            filter_frames, filters = zip(*cluster)
            assert len(set(filters)) == 1
            yield filter_frames

    new_stacks = CompositeStacksDict()
    for scale_frames in scale_clusters():
        for camera_frames in camera_clusters(scale_frames):
            for rotation_frames in rotation_clusters(camera_frames):
                for filter_frames in filter_clusters(rotation_frames):
                    if len(filter_frames) >= context.min_frames_in_stack:
                        new_stack = CompositeStack.from_frames(
                            filter_frames, metadata=metadata
                        )
                        if new_stack is not None:
                            new_stacks[new_stack.stable_hash] = new_stack

    assert None not in new_stacks.values()

    print(f"### identify_stacks done, returning {len(new_stacks)} stacks")
    return new_stacks
