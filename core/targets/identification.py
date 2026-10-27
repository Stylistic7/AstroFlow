import numpy as np
import structlog
from typing import Iterator
from beartype import beartype

import clustering
from targets.types import Target, TargetsDict
from stacks.composite_stack import StacksDict
from stacks.types import StackHash
from decorators import node
from cache.context_ref import ContextRef

log = structlog.get_logger()


@node
def identify_targets(
    angular_tolerance: float, stacks: StacksDict
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
        # Sensor GEOMETRY, not the camera string. `INSTRUME` is free text a
        # driver can rewrite between sessions, and splitting Targets on it
        # scatters one night's data across two Targets. Two stacks agreeing
        # on geometry, image scale, pointing and rotation are framing the
        # same field, which is what a Target is.
        sensors = [
            (
                resolved[s].stable_hash,
                (
                    resolved[s].width_pixels,
                    resolved[s].height_pixels,
                ),
            )
            for s in stack_hashes
        ]
        for cluster in clustering.cluster_discrete(sensors):
            stack_hashes, sensors = zip(*cluster)
            assert len(set(sensors)) == 1
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
                    # The sensor keeps cameras apart in mosaics. Checked HERE,
                    # in the node body: this node's key hashes only its own
                    # source, so the field's arrival in `from_stacks`
                    # (2026-10-07) would otherwise cache-hit sensorless
                    # Targets forever.
                    assert new_target.sensor != (0, 0), "Target without a sensor"
                    assert new_target.stable_center, "Target without a stable centre"   # same reason, 2026-10-27
                    new_targets[new_target.stable_hash] = ContextRef.store(
                        ("targets", new_target.stable_hash), new_target
                    )
                except ValueError:
                    log.warning("Failed to create target", stacks=rotation_stacks)

    log.info("Identified targets", num_targets=len(new_targets))
    return new_targets
