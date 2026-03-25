import ray

from mosaics.mosaic import MosaicsDict
from targets.types import TargetsDict
from decorators import node


@node
def validate_mosaics(mosaics: MosaicsDict, targets: TargetsDict) -> MosaicsDict:
    print(f"### Entering validate_mosaics")
    for mosaic_id, mosaic in mosaics.items():
        assert all(t in targets for t in mosaic.target_ids)

    return mosaics
