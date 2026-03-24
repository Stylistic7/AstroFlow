import numpy as np
import structlog
from typing import Iterator
from beartype import beartype

import clustering
from targets.types import Target, TargetsDict
from customization.types import CustomizedStacksDict
from stacks.types import StackHash
from decorators import node
from cache.context_ref import ContextRef

log = structlog.get_logger()


@node
def identify_targets(
    angular_tolerance: float, stacks: CustomizedStacksDict
) -> TargetsDict:
    log.info("Identifying targets", num_stacks=len(stacks))

    # Resolve all ContextRefs once up-front for efficient access
    resolved = {k: ref.get() for k, ref in stacks.items()}

    @beartype
    def scale_clusters() -> Iterator[tuple[StackHash, ...]]:
        scales = [
            (stack.stable_hash, stack.image_scale) for stack in resolved.values()
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
                resolved[s].stable_hash,
                (
                    resolved[s].camera,
                    resolved[s].width_pixels,
                    resolved[s].height_pixels,
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
                resolved[s].stable_hash,
                resolved[s].rotation_angle,
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
                        [resolved[s] for s in rotation_stacks]
                    )
                    new_targets[new_target.stable_hash] = ContextRef.store(
                        ("targets", new_target.stable_hash), new_target
                    )
                except ValueError:
                    log.warning("Failed to create target", stacks=rotation_stacks)

    log.info("Identified targets", num_targets=len(new_targets))
    return new_targets
