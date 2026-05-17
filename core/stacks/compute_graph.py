from pathlib import Path
import ray
import numpy as np
from spherical_geometry.polygon import SingleSphericalPolygon
from functools import partial
from beartype import beartype
from decorators import node
from stacks.composite_stack import Run, Stack, RunsDict, StacksDict, CalibrationStacksDict
from stacks.types import StackIdDependency
from products import get_producer, MediaType
from products.identity import StackId, identity_for_stack

from stacks.convert_to_dict import convert_to_dict
from metadata.image_metadata import compute_image_metadata
from dataclasses import replace
from decisions.settings import DevelopmentSettings
from decisions.decisions import decisions_for_stack
from decisions.subframe_selection import (
    effective_integration,
    get_light_frames_sorted_by_stars,
)
from previews.types import QualityLevel
from dependency_graph import Graph, Node, chain
from dependency_graph.workflow_graph_adapter import (
    store_workflow_graph,
    store_node_status,
)


@ray.remote(num_cpus=0.1)
@beartype
def calibration_done(stack: Stack) -> None:
    pass


from decisions.settings import ProcessingConfig
from context import context
import structlog

log = structlog.get_logger()


@node
def build_calibration_stack_subgraph(
    stack: Stack,
    config: ProcessingConfig,
) -> Graph:
    from decisions.resolve import resolve_calibration

    # Resolve calibration to discover which cal stacks this stack depends on.
    enriched = resolve_calibration(stack, config.get_calibration_stacks(), config.decisions)

    # Calibration stacks have no per-stack decisions — give them a clean
    # config so that light-stack quarantine changes don't invalidate their cache.
    from frozendict import frozendict as fd
    cal_config = replace(config, decisions=fd())

    def visit_cal_dependencies(run: Run) -> Graph:
        """Build tasks to stack all calibration dependencies of a run."""
        subgraph = {}
        for dep in run.dependencies.values():
            if isinstance(dep, StackIdDependency):
                cal_stack = config.get_calibration_stacks()[dep.stack_id]
                stack_fn = get_producer(MediaType.IMAGE, identity_for_stack(cal_stack), config.quality, config.implementation)
                task = (stack_fn, cal_stack, cal_config)
                cal_run = next(iter(cal_stack.contents.values()))
                inner = visit_cal_dependencies(cal_run)
                subgraph[task] = list(inner.keys()) if inner else []
                subgraph |= inner
        return subgraph

    cal_tasks = {}
    for run in enriched.contents.values():
        cal_tasks |= visit_cal_dependencies(run)

    if not cal_tasks:
        return Graph({})

    # Single terminal node gathering all calibration stacking tasks.
    graph = Graph(cal_tasks)
    all_cal = set(cal_tasks.keys())
    depended_on = {d for deps in cal_tasks.values() for d in deps}
    leaf_tasks = all_cal - depended_on
    graph[(calibration_done, stack)] = leaf_tasks
    return graph


@node
def build_stacking_subgraph(
    stack: Stack,
    config: ProcessingConfig,
) -> Graph:
    graph_update_fn = partial(
        store_workflow_graph, ("stack", stack.stable_hash, config.quality.value)
    )
    node_status_fn = store_node_status

    stack_fn = get_producer(MediaType.IMAGE, identity_for_stack(stack), config.quality, config.implementation)

    if config.quality == QualityLevel.MASTER:
        # Full: calibrate → stack → metadata → post-process → metadata
        pp_fn = get_producer(MediaType.IMAGE, StackId, QualityLevel.POST_PROCESSED, config.implementation)
        steps = chain(
            (build_calibration_stack_subgraph, stack, config),
            (stack_fn, stack, config),
            Node(compute_image_metadata, name=f"stacking_{stack.stable_hash}"),
            (pp_fn, stack, config.development_settings.mock_pixinsight),
            Node(compute_image_metadata, name=f"post_process_{stack.stable_hash}"),
        )
    elif config.quality == QualityLevel.DOWNSCALED_MASTER:
        # Downscaled: calibrate → stack → metadata → gradient-correct
        # (on the small image) → metadata. Mirrors MASTER so a
        # DownscaledMaster mosaic gets gradient removal too.
        pp_fn = get_producer(MediaType.IMAGE, StackId, QualityLevel.DOWNSCALED_POST_PROCESSED, config.implementation)
        steps = chain(
            (build_calibration_stack_subgraph, stack, config),
            (stack_fn, stack, config),
            Node(compute_image_metadata, name=f"stacking_{stack.stable_hash}"),
            (pp_fn, stack, config.development_settings.mock_pixinsight),
            Node(compute_image_metadata, name=f"post_process_downscaled_{stack.stable_hash}"),
        )
    else:
        # SUBFRAME or other: calibrate → stack → metadata (no post-process)
        steps = chain(
            (build_calibration_stack_subgraph, stack, config),
            (stack_fn, stack, config),
            Node(compute_image_metadata, name=f"stacking_{stack.stable_hash}"),
        )

    return Graph(
        {**steps},
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
                Node(compute_image_metadata, name=f"subframe_{stack.stable_hash}"),
            )
        })
    else:
        filtered_config = replace(config, decisions=decisions_for_stack(
            config.decisions, stack.stable_hash))
        return Graph({(build_stacking_subgraph, stack, filtered_config): {}})


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
