from pathlib import Path
import ray
import numpy as np
from spherical_geometry.polygon import SingleSphericalPolygon
from functools import partial
from beartype import beartype
from decorators import node
from stacks.stacking import stack_files
from stacks.composite_stack import Run, Stack, RunsDict, StacksDict, CalibrationStacksDict
from stacks.types import StackIdDependency
from stacks.post_process import post_process_stack
from stacks.convert_to_dict import convert_to_dict
from metadata.image_metadata import compute_image_metadata
from decisions.settings import DevelopmentSettings
from decisions.subframe_selection import get_light_frames_sorted_by_stars
from previews.types import QualityLevel
from dependency_graph import Graph, Node, chain
from dependency_graph.workflow_graph_adapter import (
    store_workflow_graph,
    store_node_status,
)


@ray.remote
@beartype
def calibration_done(stack: Stack) -> None:
    pass


from decisions.settings import ProcessingConfig

@node
def build_calibration_stack_subgraph(
    stack: Stack,
    config: ProcessingConfig,
) -> Graph:
    from decisions.resolve import resolve_calibration

    # Resolve calibration to discover which cal stacks this stack depends on.
    enriched = resolve_calibration(stack, config.calibration_stacks, config.decisions)

    def visit_cal_dependencies(run: Run) -> Graph:
        """Build tasks to stack all calibration dependencies of a run."""
        subgraph = {}
        for dep in run.dependencies.values():
            if isinstance(dep, StackIdDependency):
                cal_stack = config.calibration_stacks[dep.stack_id]
                task = (stack_files, cal_stack, config)
                cal_run = next(iter(cal_stack.contents.values()))
                inner = visit_cal_dependencies(cal_run)
                subgraph[task] = list(inner.keys()) if inner else []
                subgraph |= inner
        return subgraph

    graph = Graph({})
    for run in enriched.contents.values():
        graph |= visit_cal_dependencies(run)

    return graph


@node
def build_stacking_subgraph(
    stack: Stack,
    config: ProcessingConfig,
) -> Graph:
    graph_update_fn = partial(
        store_workflow_graph, ("stack", stack.stable_hash)
    )
    node_status_fn = store_node_status

    return Graph(
        {
            **chain(
                (build_calibration_stack_subgraph, stack, config),
                (stack_files, stack, config),
                Node(
                    compute_image_metadata, config.development_settings,
                    name=f"stacking_{stack.stable_hash}",
                ),
                (post_process_stack, stack, config.development_settings),
                Node(
                    compute_image_metadata, config.development_settings,
                    name=f"post_process_{stack.stable_hash}",
                ),
            )
        },
        graph_update_fn=graph_update_fn,
        node_status_fn=node_status_fn,
    )


# TODO: Calibrate the subframe returned here.
@node
def get_stack_subframe_image(stack: Stack) -> Path:
    assert stack is not None
    light_frames_by_stars = get_light_frames_sorted_by_stars(stack)
    best_light_frame = stack.all_metadata()[light_frames_by_stars[0]]
    return Path(best_light_frame.filename)


@ray.remote
def get_stack_image(
    mode: QualityLevel,
    stack: Stack,
    config: ProcessingConfig,
) -> Graph:
    assert stack is not None
    if mode == QualityLevel.SUBFRAME:
        return Graph({
            **chain(
                (get_stack_subframe_image, stack),
                Node(compute_image_metadata, config.development_settings, name=f"subframe_{stack.stable_hash}"),
            )
        })
    else:
        return Graph({(build_stacking_subgraph, stack, config): {}})


def _my_transform(v):
    if type(v) is np.ndarray:
        return v.tolist()
    if type(v) is SingleSphericalPolygon:
        return v.points.tolist()
    return v


@ray.remote
def _prepare_stack_for_ui(
    stack: Stack,
    cal_stacks: CalibrationStacksDict,
    decisions,
    min_frames: int,
) -> dict:
    from decisions.resolve import resolve_calibration, resolve_decisions
    from decisions.compute_graph import ALL_EXCLUSION_TYPES

    # Resolve calibration so the UI shows calibration deps per run.
    stack = resolve_calibration(stack, cal_stacks, decisions)
    # Publish quality/rotation decisions as side effects (for UI visibility).
    resolve_decisions(stack, ALL_EXCLUSION_TYPES, decisions, min_frames)

    # Emit only the scalar fields the UI needs for starmap, cards, and search.
    # Heavy fields (contents, projected_footprints, footprint) are fetched
    # on demand via /stacks/<id>/detail.
    return {
        "stable_hash": stack.stable_hash,
        "content_hash": stack.content_hash,
        "name": stack.name,
        "image_type": stack.image_type,
        "camera": stack.camera,
        "filter": stack.filter,
        "exposure": stack.exposure,
        "integration": stack.integration,
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


@ray.remote
def process_stacks_for_ui(*_ancestor_results) -> dict:
    from context import context
    from cache.context_ref import ContextRef

    stacks = context.stacks or {}
    cal_stacks = context.calibration_stacks or CalibrationStacksDict()
    decisions = context.settings.decision_overrides
    min_frames = context.settings.clustering_settings.min_frames_in_stack

    # Fire all stack preparations in parallel.
    futures = {
        h: _prepare_stack_for_ui.remote(
            ref.get() if isinstance(ref, ContextRef) else ref,
            cal_stacks, decisions, min_frames,
        )
        for h, ref in stacks.items()
    }

    return {h: ray.get(ref) for h, ref in futures.items()}


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
