from pathlib import Path
import ray
import numpy as np
from spherical_geometry.polygon import SingleSphericalPolygon
from functools import partial
from beartype import beartype

from decorators import cached
from stacks.stacking import stack_files
from stacks.composite_stack import SimpleStack, CompositeStack, SimpleStacksDict
from stacks.types import StackIdDependency
from stacks.post_process import post_process_stack
from stacks.convert_to_dict import convert_to_dict
from stacks.composite_stack import CompositeStacksDict
from metadata.image_metadata import compute_image_metadata_no_solve
from customization.types import CustomizedStack
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
def calibration_done(stack: CustomizedStack) -> None:
    pass


@cached.remote
def build_calibration_stack_subgraph(
    stack: CustomizedStack,
    calibration_stacks: SimpleStacksDict,
    development_settings: DevelopmentSettings,
    min_frames_in_stack: int,
) -> Graph:
    def visit_all_dependencies(stack: SimpleStack, depth=0) -> Graph:
        assert isinstance(stack, SimpleStack)

        tasks = []
        subgraphs = {}
        for t, dep in stack.dependencies.items():
            if isinstance(dep, StackIdDependency):
                substack = calibration_stacks[dep.stack_id]
                tasks.append((stack_files, substack, calibration_stacks, development_settings, min_frames_in_stack))
                subgraphs |= visit_all_dependencies(substack, depth=depth + 2)

        if depth == 0:
            return subgraphs
        else:
            return {(stack_files, stack, calibration_stacks, development_settings, min_frames_in_stack): tasks} | subgraphs

    graph = Graph({})
    for substack in stack.contents.values():
        graph |= visit_all_dependencies(substack)

    return graph


@ray.remote
@beartype
def stack_runs_done(stack: CustomizedStack) -> None:
    pass


# This function accepts a fully customized stack, after settings and calibration
# frames have been associated.
@cached.remote
def build_stacking_subgraph(
    stack: CustomizedStack,
    calibration_stacks: SimpleStacksDict,
    development_settings: DevelopmentSettings,
    min_frames_in_stack: int,
) -> Graph:
    graph_update_fn = partial(
        store_workflow_graph, ("stack", stack.stable_hash)
    )
    node_status_fn = store_node_status

    return Graph(
        {
            **chain(
                (build_calibration_stack_subgraph, stack, calibration_stacks, development_settings, min_frames_in_stack),
                (stack_files, stack, calibration_stacks, development_settings, min_frames_in_stack),
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
@cached.remote
def get_stack_subframe_image(stack: CompositeStack) -> Path:
    assert stack is not None
    light_frames_by_stars = get_light_frames_sorted_by_stars(stack)
    best_light_frame = stack.all_metadata()[light_frames_by_stars[0]]
    return Path(best_light_frame.filename)


@cached.remote
@beartype
def get_stack_image(
    mode: QualityLevel,
    stack: CompositeStack,
    calibration_stacks: SimpleStacksDict,
    development_settings: DevelopmentSettings,
    min_frames_in_stack: int,
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
        return Graph({(build_stacking_subgraph, stack, calibration_stacks, development_settings, min_frames_in_stack): {}})


@cached.remote
@beartype
def process_stacks_for_ui(stacks: dict) -> dict:
    assert None not in stacks.values()

    def my_transform(v):
        if type(v) is np.ndarray:
            return v.tolist()
        if type(v) is SingleSphericalPolygon:
            return v.points.tolist()
        return v

    def prepare_stack(stack: CustomizedStack):
        result = convert_to_dict(
            stack, [["contents", "*", "metadata"]], transform=my_transform
        )

        filenames = []
        for s in stack.contents.values():
            filenames += [m.filename for m in s.metadata.values()]

        result["filenames"] = filenames

        return result

    # TODO: Parallelize this.
    from cache.context_ref import ContextRef
    ui_stacks = {
        h: prepare_stack(ref.get() if isinstance(ref, ContextRef) else ref)
        for h, ref in stacks.items()
    }
    return ui_stacks


@cached.remote
@beartype
def process_calibration_stacks_for_ui(cal_stacks: dict) -> dict:
    assert None not in cal_stacks.values()

    def prepare_stack(stack: SimpleStack):
        result = convert_to_dict(stack, [["metadata"]])
        result["filenames"] = [m.filename for m in stack.metadata.values()]

        return result

    # TODO: Parallelize this.
    from cache.context_ref import ContextRef
    ui_cal_stacks = {
        h: prepare_stack(ref.get() if isinstance(ref, ContextRef) else ref)
        for h, ref in cal_stacks.items()
    }
    return ui_cal_stacks
