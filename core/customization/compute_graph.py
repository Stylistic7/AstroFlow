import structlog
from frozendict import frozendict

from decorators import node
from customization.decisions import (
    DecisionType, DecisionOverride, DecisionKey, ObjectType, BinaryChoice,
)
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
    key = (DecisionType.QUARANTINE, object_type, object_id, object_id)
    override = decision_overrides.get(key)
    return override is not None and override.choice == BinaryChoice.EXCLUDE


def _all_stacks_quarantined(
    decision_overrides: frozendict[DecisionKey, DecisionOverride],
    target,
) -> bool:
    return all(
        _is_quarantined(decision_overrides, ObjectType.STACK, stack_id)
        for stack_id in target.stack_ids
    )


@node
def scatter_customize_targets(
    decision_overrides: frozendict, targets: TargetsDict
) -> TargetsDict:
    from cache.context_ref import ContextRef
    return TargetsDict(
        {
            h: ref
            for h, ref in targets.items()
            if not _is_quarantined(decision_overrides, ObjectType.TARGET, h)
            and not _all_stacks_quarantined(
                decision_overrides,
                ref.get() if isinstance(ref, ContextRef) else ref,
            )
        }
    )


@node
def scatter_customize_mosaics(
    decision_overrides: frozendict, mosaics: MosaicsDict
) -> MosaicsDict:
    return MosaicsDict(
        {
            h: ref
            for h, ref in mosaics.items()
            if not _is_quarantined(decision_overrides, ObjectType.MOSAIC, h)
        }
    )
