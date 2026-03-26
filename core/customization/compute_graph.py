import structlog
from frozendict import frozendict

from decorators import node
from customization.settings import TargetSettings
from customization.decisions import DecisionType
from mosaics.types import MosaicsDict
from targets.types import TargetsDict

log = structlog.get_logger()

# All exclusion decision types, applied in precedence order by resolve_decisions.
ALL_EXCLUSION_TYPES = (
    DecisionType.QUARANTINE,
    DecisionType.ROTATION_OUTLIER,
    DecisionType.SUBFRAME_QUALITY,
)


@node
def scatter_customize_targets(
    target_settings: frozendict, targets: TargetsDict
) -> TargetsDict:
    return TargetsDict(
        {
            h: ref
            for h, ref in targets.items()
            if not target_settings.get(h, TargetSettings()).is_quarantined
        }
    )


# TODO: Implement real customization features.
@node
def scatter_customize_mosaics(
    mosaic_settings: frozendict, mosaics: MosaicsDict
) -> MosaicsDict:
    return mosaics
