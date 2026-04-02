import structlog
from frozendict import frozendict

from decorators import node
from decisions.decisions import (
    DecisionType, DecisionOverride, DecisionKey, ObjectType, BinaryChoice,
)
from decisions.helpers import quarantine_key
from mosaics.types import MosaicsDict
from targets.types import TargetsDict

log = structlog.get_logger()

# All exclusion decision types, applied in precedence order by resolve_decisions.
ALL_EXCLUSION_TYPES = (
    DecisionType.QUARANTINE,
    DecisionType.ROTATION_OUTLIER,
    DecisionType.SUBFRAME_QUALITY,
)


def _is_quarantined(
    decision_overrides: frozendict[DecisionKey, DecisionOverride],
    object_type: ObjectType,
    object_id: str,
) -> bool:
    override = decision_overrides.get(quarantine_key(object_type, object_id))
    return override is not None and override.choice == BinaryChoice.EXCLUDE


@node
def scatter_customize_targets(
    decision_overrides: frozendict, targets: TargetsDict
) -> TargetsDict:
    return targets


@node
def scatter_customize_mosaics(
    decision_overrides: frozendict, mosaics: MosaicsDict
) -> MosaicsDict:
    return mosaics
