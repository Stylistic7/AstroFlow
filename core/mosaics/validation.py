import ray
import structlog

from mosaics.mosaic import MosaicsDict
from targets.types import TargetsDict
from decorators import node

log = structlog.get_logger()


@node
def validate_mosaics(mosaics: MosaicsDict, targets: TargetsDict) -> MosaicsDict:
    log.info("Entering validate_mosaics")
    for mosaic_id, mosaic in mosaics.items():
        assert all(t in targets for t in mosaic.target_ids)

    return mosaics
