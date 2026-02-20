from dataclasses import replace
from beartype import beartype

from decorators import cached
from targets.types import Target
from targets.types import TargetHash
from targets.types import TargetsDict
from stacks.composite_stack import CompositeStacksDict, SimpleStacksDict
from stacks.types import StackHash
from mosaics.types import MosaicHash, MosaicsDict
from mosaics.types import Mosaic


@beartype
@cached.remote
def quarantine_stacks(
    quarantined_stacks: list[StackHash], stacks: CompositeStacksDict
) -> CompositeStacksDict:
    return {
        stack_id: stack
        for stack_id, stack in stacks.items()
        if stack_id not in quarantined_stacks
    }


@beartype
@cached.remote
def quarantine_targets(
    quarantined_stacks: list[StackHash],
    quarantined_targets: list[TargetHash],
    stacks: CompositeStacksDict,
    targets: TargetsDict,
) -> TargetsDict:
    print(f"### quarantine_targets {len(targets)} targets")

    output_targets = {}
    for target_id, target in targets.items():
        if target_id in quarantined_targets:
            print(f"### removing quarantined target {target_id}")
            continue

        stack_ids = [
            s
            for s in target.stack_ids
            if s in stacks and s not in quarantined_stacks
        ]
        if not stack_ids:
            continue

        target = replace(target, stack_ids=stack_ids)
        output_targets[target_id] = target

    print(f"### quarantine_targets returning {len(output_targets)} targets")
    return output_targets


@cached.remote
def quarantine_mosaics(
    quarantined_targets: list[TargetHash],
    quarantined_mosaics: list[MosaicHash],
    targets: TargetsDict,
    mosaics: MosaicsDict,
) -> MosaicsDict:
    output_mosaics = {}
    for mosaic_id, mosaic in mosaics.items():
        if mosaic_id in quarantined_mosaics:
            continue

        new_target_ids = []
        for target_id in mosaic.target_ids:
            if target_id not in targets or target_id in quarantined_targets:
                continue

            new_target_ids.append(target_id)

        # TODO: After removing targets, strips have to be recomputed.
        mosaic = replace(mosaic, target_ids=new_target_ids)
        output_mosaics[mosaic_id] = mosaic

    return output_mosaics


@cached.remote
def quarantine_calibration_stacks(
    quarantined_calibration_stacks: list[StackHash],
    cal_stacks: SimpleStacksDict,
) -> SimpleStacksDict:
    return SimpleStacksDict(
        {
            h: s
            for h, s in cal_stacks.items()
            if h not in quarantined_calibration_stacks
        }
    )
