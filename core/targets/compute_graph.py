import structlog
import numpy as np
from spherical_geometry.polygon import SingleSphericalPolygon

from decorators import node
from targets.types import Target
from stacks.convert_to_dict import convert_to_dict

log = structlog.get_logger()


@node
def process_targets_for_ui(targets: dict) -> dict:
    def my_transform(v):
        if type(v) is np.ndarray:
            return v.tolist()
        if type(v) is SingleSphericalPolygon:
            return v.points.tolist()
        return v

    def prepare_target(target: Target):
        return convert_to_dict(target, transform=my_transform)

    # TODO: Parallelize this.
    from cache.context_ref import ContextRef
    ui_targets = {
        h: prepare_target(ref.get() if isinstance(ref, ContextRef) else ref)
        for h, ref in targets.items()
    }
    return ui_targets
