import ray
from beartype import beartype

from decorators import node
from file_io.types import FitsFilePath, WithSidecar, XisfFilePath
from stacks.composite_stack import CalibrationStack, StacksDict
from cache.context_ref import ContextRef
from dependency_graph import Graph, TaskDescriptor
from stacks.composite_stack import Stack
from decisions.subframe_selection import get_light_frames_sorted_by_stars


@ray.remote(num_cpus=0.1)
@beartype
def scatter_stack_thumbnails(
    cal_stacks_ref: ContextRef,
    stacks: StacksDict,
) -> Graph:
    """Kept as a no-op so the preview graph's structure and ordering are
    unchanged. Thumbnails are rendered lazily on first view by the web
    server (`web_server/backend/thumbnails`); the eager per-subframe videos
    this used to schedule were deleted on 2026-10-01 (never generated since
    they dominated cold start, and never served: key mismatch)."""
    tasks: list[TaskDescriptor] = list()
    return Graph({gather_stack_thumbnails: tasks})


def get_first_frame(stack: CalibrationStack) -> WithSidecar:
    # A STAR flat has no frames to integrate: its members are the light subs
    # whose sidecar photometry the solve reads, so `contents` is empty and
    # `all_frames()[0]` raises IndexError. `get_thumbnail` swallows that and
    # falls through to 404, which left every star flat's detail page blank
    # (2026-09-09). Its picture is the built master itself — the same source
    # `previews.calibration.get_subframe_for_calibration_stack` already uses
    # for the Run carousel. Never trigger a build from a preview: an unbuilt
    # flat simply has no thumbnail yet.
    if getattr(stack, "kind", None) == "star":
        from stacks.star_flat import star_flat_master_path
        master = star_flat_master_path(stack)
        if master is None:
            raise FileNotFoundError(
                f"star flat {stack.stable_hash} has no built master yet"
            )
        return WithSidecar(XisfFilePath(str(master)))
    # Use a sidecar file to avoid ever modifying the user's raw data files.
    metadata = stack.all_metadata()
    first_frame = stack.all_frames()[0]
    return WithSidecar(FitsFilePath(metadata[first_frame].filename))


def get_best_light_frame(stack: Stack) -> WithSidecar:
    metadata = stack.all_metadata()
    light_frames_by_stars = get_light_frames_sorted_by_stars(stack)

    # Use a sidecar file to avoid ever modifying the user's raw data files.
    return WithSidecar(
        FitsFilePath(metadata[light_frames_by_stars[0]].filename)
    )


@node(filename_fn=get_best_light_frame)
def stack_thumbnail(stack: Stack) -> WithSidecar:
    return get_best_light_frame(stack)


@node(filename_fn=get_first_frame)
def calibration_stack_thumbnail(stack: CalibrationStack) -> WithSidecar:
    return get_first_frame(stack)


@ray.remote(num_cpus=0.1)
@beartype
def gather_stack_thumbnails() -> None:
    pass
