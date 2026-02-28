from pathlib import Path
import ray
import numpy as np
from spherical_geometry.polygon import SingleSphericalPolygon
from functools import partial

from context import context
from decorators import cached
from stacks.stacking import stack_files
from stacks.composite_stack import SimpleStack, CompositeStack
from stacks.post_process import post_process_stack
from stacks.convert_to_dict import convert_to_dict
from stacks.composite_stack import CompositeStacksDict
from metadata.image_metadata import compute_image_metadata
from customization.types import CustomizedStack
from customization.subframe_selection import get_light_frames_sorted_by_stars
from previews.types import QualityLevel
from dependency_graph import Graph, Node, chain
from dependency_graph.workflow_graph_adapter import (
    store_workflow_graph,
    store_node_status,
)


@ray.remote
def calibration_done(stack: CustomizedStack) -> None:
    pass


@cached.remote
def build_calibration_stack_subgraph(
    mode: QualityLevel, stack: CustomizedStack
) -> Graph:
    calibration_stacks = context.calibration_stacks

    def visit_all_dependencies(stack: SimpleStack, depth=0) -> Graph:
        assert isinstance(stack, SimpleStack)

        tasks = []
        subgraphs = {}
        for t, dep in stack.dependencies.items():
            if (stack_id := dep.get("stack_id")) is not None:
                substack = calibration_stacks[stack_id]
                tasks.append((stack_files, mode, substack))
                subgraphs |= visit_all_dependencies(substack, depth=depth + 2)

        if depth == 0:
            return subgraphs
        else:
            return {(stack_files, mode, stack): tasks} | subgraphs

    graph = Graph({})
    for substack in stack.contents.values():
        graph |= visit_all_dependencies(substack)

    return graph


@ray.remote
def stack_runs_done(stack: CustomizedStack) -> None:
    pass


# This function accepts a fully customized stack, after settings and calibration
# frames have been associated.
@cached.remote
def build_stacking_subgraph(
    mode: QualityLevel, stack: CustomizedStack
) -> Graph:
    graph_update_fn, node_status_fn = None, None
    if mode == QualityLevel.MASTER:
        graph_update_fn = partial(
            store_workflow_graph, ("stack", stack.stable_hash)
        )
        node_status_fn = store_node_status

    return Graph(
        {
            **chain(
                (build_calibration_stack_subgraph, mode, stack),
                (stack_files, mode, stack),
                Node(
                    compute_image_metadata, name=f"stacking_{stack.stable_hash}"
                ),
                # TODO: Put the post-processing back into the graph somewhere
                # else.
                # (post_process_stack, mode, stack),
                # Node(
                #     compute_image_metadata,
                #     name=f"post_process_{stack.stable_hash}",
                # ),
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
def get_stack_image(mode: QualityLevel, stack: CompositeStack) -> Graph:
    assert stack is not None
    if mode == QualityLevel.SUBFRAME:
        return Graph({(get_stack_subframe_image, stack): {}})
    else:
        return Graph({(build_stacking_subgraph, mode, stack): {}})


@cached.remote
def process_stacks_for_ui(stacks: CompositeStacksDict) -> None:
    print(f"### process_stacks_for_ui type(stacks) {type(stacks)}")
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
    ui_stacks = {h: prepare_stack(s) for h, s in stacks.items()}
    context.ui_stacks = ui_stacks
    return ui_stacks
