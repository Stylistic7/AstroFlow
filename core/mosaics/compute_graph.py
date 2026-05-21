"""Mosaic helpers + the UI-facing `process_mosaics_for_ui` @node.

The legacy `build_mosaic_subgraph` / `build_mosaic_filter_subgraph` /
`build_placement_subgraph` graph builders that used to live here are
gone — each `@pipeline` in `core/pipelines/*.py` now expresses its
own placement + merge directly. The Mosaic helpers in this module
(`targets_for_mosaic`, `_unwrap`, `get_targets_for_filter`) survive
because they're useful idioms for pipeline bodies and tests.
"""
import numpy as np
import structlog
from spherical_geometry.polygon import SingleSphericalPolygon

from decorators import node
from mosaics.types import Mosaic, MosaicsDict
from stacks.convert_to_dict import convert_to_dict
from targets.target_channel import filters_for_target
from targets.types import TargetsDict, Target, TargetHash

log = structlog.get_logger()


def targets_for_mosaic(mosaic: Mosaic, targets: TargetsDict) -> TargetsDict:
    """Returns only the targets included in the mosaic."""
    return TargetsDict(
        {target_id: targets[target_id] for target_id in mosaic.target_ids}
    )


def _unwrap(ref):
    """Unwrap a ContextRef to its value, or return plain objects as-is."""
    from cache.context_ref import ContextRef
    return ref.get() if isinstance(ref, ContextRef) else ref


def get_targets_for_filter(
    mosaic: Mosaic,
    filter: str,
    targets: TargetsDict,
) -> tuple[tuple[Target | None, ...], ...]:
    """Replace each TargetHash in mosaic.strips with the resolved Target,
    or None if the target has no stack matching this filter.
    """
    def target_with_filter(target_id: TargetHash) -> Target | None:
        target = _unwrap(targets[target_id])
        return target if filter in filters_for_target(target) else None

    return tuple(
        tuple(target_with_filter(tid) for tid in strip)
        for strip in mosaic.strips
    )


@node
def process_mosaics_for_ui(mosaics: MosaicsDict) -> dict:
    def my_transform(v):
        if type(v) is np.ndarray:
            return v.tolist()
        if type(v) is SingleSphericalPolygon:
            return v.points.tolist()
        return v

    def prepare_mosaic(mosaic: Mosaic):
        return convert_to_dict(mosaic, transform=my_transform)

    # TODO: Parallelize this.
    ui_mosaics = {h: prepare_mosaic(ref.get()) for h, ref in mosaics.items()}
    return ui_mosaics
