from pathlib import Path
import ray
import numpy as np
from spherical_geometry.polygon import SingleSphericalPolygon
from beartype import beartype
from decorators import node
from stacks.composite_stack import Run, Stack, StacksDict, CalibrationStacksDict
from stacks.types import StackIdDependency

from stacks.convert_to_dict import convert_to_dict
from dataclasses import replace
from decisions.decisions import decisions_for_stack
from decisions.subframe_selection import (
    effective_integration,
    get_light_frames_sorted_by_stars,
)
from dependency_graph import Graph


@ray.remote(num_cpus=0.1)
@beartype
def calibration_done(stack: Stack) -> None:
    pass


from decisions.settings import ProcessingConfig
from context import context
import structlog

log = structlog.get_logger()


@node
def calibrate(
    stack: Stack,
    config: ProcessingConfig,
    stacker=None,
) -> Graph:
    """Build the per-(bias, dark, flat) stacking sub-graph this
    `stack`'s calibration depends on, terminating in a single
    `calibration_done` sentinel that downstream nodes can `_after=`
    on.

    `stacker` selects the @node op that stacks each cal stack —
    `stack_files` (full-resolution) or `stack_files_downscaled`
    (frames block-mean-downscaled 8x before WBPP). Pipelines pass
    whichever stacker matches their light-stack op; that way cal
    masters come out at the same resolution the light frames will
    be calibrated against. Default is full-resolution PI WBPP.
    """
    from decisions.resolve import resolve_calibration
    from frozendict import frozendict as fd
    if stacker is None:
        from stacks.stacking.pixinsight import stack_files as stacker

    enriched = resolve_calibration(
        stack, config.get_calibration_stacks(), config.decisions,
    )
    cal_config = replace(config, decisions=fd())

    def visit(run: Run) -> dict:
        """Recursively build stacking tasks for one run's cal deps."""
        subgraph = {}
        for dep in run.dependencies.values():
            if not isinstance(dep, StackIdDependency):
                continue
            cal_stack = config.get_calibration_stacks()[dep.stack_id]
            task = (stacker, cal_stack, cal_config)
            cal_run = next(iter(cal_stack.contents.values()))
            inner = visit(cal_run)
            subgraph[task] = list(inner) if inner else []
            subgraph |= inner
        return subgraph

    cal_tasks = {}
    for run in enriched.contents.values():
        cal_tasks |= visit(run)

    if not cal_tasks:
        return Graph({})

    # Terminal sentinel: every leaf cal task funnels into one Node so
    # the caller can wait on a single handle.
    graph = Graph(cal_tasks)
    has_dep = {d for ds in cal_tasks.values() for d in ds}
    graph[(calibration_done, stack)] = set(cal_tasks) - has_dep
    return graph


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
