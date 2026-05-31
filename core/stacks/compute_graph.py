from pathlib import Path
import ray
import numpy as np
from spherical_geometry.polygon import SingleSphericalPolygon
from beartype import beartype
from decorators import node
from stacks.composite_stack import Run, Stack, StacksDict, CalibrationStacksDict

from stacks.convert_to_dict import convert_to_dict
from decisions.decisions import decisions_for_stack
from decisions.subframe_selection import (
    effective_integration,
    get_light_frames_sorted_by_stars,
)
from dependency_graph import Graph
from decisions.settings import ProcessingConfig
from context import context
import structlog

log = structlog.get_logger()


# TODO: Calibrate the subframe returned here.
@node
def get_stack_subframe_image(stack: Stack) -> Path:
    assert stack is not None
    light_frames_by_stars = get_light_frames_sorted_by_stars(stack)
    best_light_frame = stack.all_metadata()[light_frames_by_stars[0]]
    return Path(best_light_frame.filename)


def _my_transform(v):
    if type(v) is np.ndarray:
        return v.tolist()
    if type(v) is SingleSphericalPolygon:
        return v.points.tolist()
    return v



@node
def prepare_stack_for_ui(
    stack: Stack,
    cal_stacks: CalibrationStacksDict,
    decisions,
    min_frames: int,
) -> dict:
    from decisions.resolve import resolve_calibration, resolve_decisions
    from decisions.compute_graph import ALL_EXCLUSION_TYPES

    stack = resolve_calibration(stack, cal_stacks, decisions)
    resolve_decisions(stack, ALL_EXCLUSION_TYPES, decisions, min_frames)
    return {
        "stable_hash": stack.stable_hash,
        "content_hash": stack.content_hash,
        "name": stack.name,
        "image_type": stack.image_type,
        "camera": stack.camera,
        "filter": stack.filter,
        "exposure": stack.exposure,
        # `integration` is the *effective* (post-subframe-selection)
        # signal — what the viewer's exposure clock should reflect.
        # Run-level views still derive raw per-frame data themselves.
        "integration": effective_integration(stack),
        # `raw_integration_s` is the raw sum of EXPOSURE across every
        # subframe, with no quality filter applied. Used as the upper
        # bound (blue polygon) in the dual exposure widget on the
        # mosaic view, against which the WBPP-effective value
        # (`wbpp_exposure_s` carried in the /api/metadata output
        # polygon) is compared.
        "raw_integration_s": stack.integration,
        "mean_temperature": stack.mean_temperature,
        "image_scale": stack.image_scale,
        "rotation_angle": stack.rotation_angle,
        "width_pixels": stack.width_pixels,
        "height_pixels": stack.height_pixels,
        "center": list(stack.center) if stack.center else None,
        "oldest_date": str(stack.oldest_date) if stack.oldest_date else None,
        "newest_date": str(stack.newest_date) if stack.newest_date else None,
        "state": dict(stack.state),
        "filenames": [
            m.filename for s in stack.contents.values() for m in s.metadata.values()
        ],
    }


@ray.remote(num_cpus=0.1)
def gather_ui_stacks(*stack_dicts) -> dict:
    """Combine per-stack UI dicts into a single {hash: dict} mapping."""
    return {d["stable_hash"]: d for d in stack_dicts}


@node
def process_stacks_for_ui(
    decisions,
    min_frames: int,
    *ancestor_results,
) -> Graph:
    from cache.context_ref import ContextRef

    # Ancestor results arrive in non-deterministic set order; unpack by type.
    stacks = next(r for r in ancestor_results if isinstance(r, StacksDict))
    cal_stacks = next(r for r in ancestor_results if isinstance(r, CalibrationStacksDict))

    per_stack_tasks = []
    for h, ref in stacks.items():
        stack = ref.get() if isinstance(ref, ContextRef) else ref
        stack_decisions = decisions_for_stack(decisions, stack.stable_hash)
        per_stack_tasks.append(
            (prepare_stack_for_ui, stack, cal_stacks, stack_decisions, min_frames)
        )

    return Graph({gather_ui_stacks: per_stack_tasks})


@node
def process_calibration_stacks_for_ui(cal_stacks: dict) -> dict:
    assert None not in cal_stacks.values()

    def prepare_stack(stack):
        result = convert_to_dict(stack, [["contents", "*", "metadata"]])
        result["filenames"] = [m.filename for m in stack.all_metadata().values()]

        return result

    # TODO: Parallelize this.
    from cache.context_ref import ContextRef
    ui_cal_stacks = {
        h: prepare_stack(ref.get() if isinstance(ref, ContextRef) else ref)
        for h, ref in cal_stacks.items()
    }
    return ui_cal_stacks
