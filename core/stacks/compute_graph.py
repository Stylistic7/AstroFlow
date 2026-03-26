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
from metadata.image_metadata import compute_image_metadata_no_solve
from customization.settings import DevelopmentSettings
from customization.subframe_selection import get_light_frames_sorted_by_stars
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


@node
def build_calibration_stack_subgraph(
    stack: Stack,
    calibration_stacks: CalibrationStacksDict,
    development_settings: DevelopmentSettings,
    min_frames_in_stack: int,
) -> Graph:
    def visit_cal_dependencies(run: Run) -> Graph:
        """Build tasks to stack all calibration dependencies of a run."""
        subgraph = {}
        for dep in run.dependencies.values():
            if isinstance(dep, StackIdDependency):
                cal_stack = calibration_stacks[dep.stack_id]
                task = (stack_files, cal_stack, calibration_stacks, development_settings, min_frames_in_stack)
                # Recurse: this cal stack's run may depend on other cal stacks.
                cal_run = next(iter(cal_stack.contents.values()))
                inner = visit_cal_dependencies(cal_run)
                subgraph[task] = list(inner.keys()) if inner else []
                subgraph |= inner
        return subgraph

    graph = Graph({})
    for run in stack.contents.values():
        graph |= visit_cal_dependencies(run)

    return graph


@ray.remote
@beartype
def stack_runs_done(stack: Stack) -> None:
    pass


from frozendict import frozendict

# This function accepts an enriched stack (all frames with calibration).
# stack_files resolves decisions internally to filter before stacking.
@node
def build_stacking_subgraph(
    stack: Stack,
    calibration_stacks: CalibrationStacksDict,
    development_settings: DevelopmentSettings,
    min_frames_in_stack: int,
    decisions: frozendict = frozendict(),
) -> Graph:
    graph_update_fn = partial(
        store_workflow_graph, ("stack", stack.stable_hash)
    )
    node_status_fn = store_node_status

    return Graph(
        {
            **chain(
                (build_calibration_stack_subgraph, stack, calibration_stacks, development_settings, min_frames_in_stack),
                (stack_files, stack, calibration_stacks, development_settings, min_frames_in_stack, decisions),
                Node(
                    compute_image_metadata_no_solve, name=f"stacking_{stack.stable_hash}"
                ),
                (post_process_stack, stack, development_settings),
                Node(
                    compute_image_metadata_no_solve,
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
    calibration_stacks: CalibrationStacksDict,
    development_settings: DevelopmentSettings,
    min_frames_in_stack: int,
    decisions: frozendict = frozendict(),
) -> Graph:
    assert stack is not None
    if mode == QualityLevel.SUBFRAME:
        return Graph({
            **chain(
                (get_stack_subframe_image, stack),
                Node(compute_image_metadata_no_solve, name=f"subframe_{stack.stable_hash}"),
            )
        })
    else:
        return Graph({(build_stacking_subgraph, stack, calibration_stacks, development_settings, min_frames_in_stack, decisions): {}})


@ray.remote
def process_stacks_for_ui(*_ancestor_results) -> dict:
    from context import context
    from cache.context_ref import ContextRef
    from customization.resolve import resolve_calibration, resolve_decisions
    from customization.compute_graph import ALL_EXCLUSION_TYPES
    from frozendict import frozendict

    stacks = context.stacks or {}
    cal_stacks = context.calibration_stacks or CalibrationStacksDict()
    decisions = context.settings.decision_overrides

    def my_transform(v):
        if type(v) is np.ndarray:
            return v.tolist()
        if type(v) is SingleSphericalPolygon:
            return v.points.tolist()
        return v

    min_frames = context.settings.clustering_settings.min_frames_in_stack

    def prepare_stack(stack: Stack):
        # Resolve calibration so the UI shows calibration deps per run.
        stack = resolve_calibration(stack, cal_stacks, decisions)
        # Publish quality/rotation decisions as side effects (for UI visibility).
        resolve_decisions(stack, ALL_EXCLUSION_TYPES, decisions, min_frames)
        result = convert_to_dict(
            stack, [["contents", "*", "metadata"]], transform=my_transform
        )

        filenames = []
        for s in stack.contents.values():
            filenames += [m.filename for m in s.metadata.values()]

        result["filenames"] = filenames

        return result

    ui_stacks = {
        h: prepare_stack(ref.get() if isinstance(ref, ContextRef) else ref)
        for h, ref in stacks.items()
    }
    return ui_stacks


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
