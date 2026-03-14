import numpy as np
from typing import Iterator
from beartype import beartype

import clustering
from targets.types import Target, TargetsDict
from stacks.composite_stack import CompositeStacksDict
from stacks.types import StackHash
from decorators import cached


@cached.remote
def identify_targets(
    angular_tolerance: float, stacks: CompositeStacksDict
) -> TargetsDict:
    print(f"### identify_targets stacks {list(stacks.keys())}")

    @beartype
    def scale_clusters() -> Iterator[tuple[StackHash, ...]]:
        scales = [
            (stack.stable_hash, stack.image_scale) for stack in stacks.values()
        ]
        for cluster in clustering.cluster_continuous(scales, epsilon=0.2):
            scale_stacks, _ = zip(*cluster)
            yield scale_stacks

    @beartype
    def camera_clusters(
        stack_hashes: tuple[StackHash, ...],
    ) -> Iterator[tuple[StackHash, ...]]:
        cameras = [
            (
                stacks[s].stable_hash,
                (
                    stacks[s].camera,
                    stacks[s].width_pixels,
                    stacks[s].height_pixels,
                ),
            )
            for s in stack_hashes
        ]
        for cluster in clustering.cluster_discrete(cameras):
            stack_hashes, cameras = zip(*cluster)
            assert len(set(cameras)) == 1
            yield stack_hashes

    @beartype
    def rotation_clusters(
        stack_hashes: tuple[StackHash, ...],
    ) -> Iterator[tuple[StackHash, ...]]:
        rotation_angles = [
            (
                stacks[s].stable_hash,
                stacks[s].rotation_angle,
            )
            for s in stack_hashes
        ]
        for cluster in clustering.cluster_angles_mod_pi(
            rotation_angles, angular_tolerance
        ):
            stack_hashes, _ = zip(*cluster)
            yield stack_hashes

    new_targets = TargetsDict()
    for scale_stacks in scale_clusters():
        for camera_stacks in camera_clusters(scale_stacks):
            for rotation_stacks in rotation_clusters(camera_stacks):
                try:
                    new_target = Target.from_stacks(
                        [stacks[s] for s in rotation_stacks]
                    )
                    new_targets[new_target.stable_hash] = new_target
                except ValueError:
                    print(
                        f"### WARNING: Failed to create target from "
                        f"stacks {rotation_stacks}"
                    )

    print(f"### identify_targets done, returning {len(new_targets)} targets")
    return new_targets
