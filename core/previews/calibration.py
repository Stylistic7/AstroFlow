import numpy as np
from functools import cache

from context import context
from metadata.classes import ImageMetadata
from stacks.types import StackHash, DependenciesDict, StackIdDependency
from previews.operations import resize_array
from file_io.generic import read_image


# TODO: Average flat frames before applying them.
def calibrate_image(
    light_metadata: ImageMetadata | None,
    light_image: np.ndarray | None,
    bias_metadata: ImageMetadata | None,
    bias_image: np.ndarray | None,
    dark_metadata: ImageMetadata | None,
    dark_image: np.ndarray | None,
    flat_metadata: ImageMetadata | None,
    flat_image: np.ndarray | None,
    rotate: bool,
) -> np.ndarray:
    assert light_image is not None

    if bias_image is None or dark_image is None or flat_image is None:
        if rotate:
            light_image = np.rot90(light_image, 2)
        return light_image

    assert (
        light_metadata is not None
        and dark_metadata is not None
        and flat_metadata is not None
    )
    assert light_image is not None and dark_image is not None and flat_image is not None

    image_sizes = {
        light_image.shape,
        dark_image.shape,
        bias_image.shape,
        flat_image.shape,
    }
    assert len(image_sizes) == 1, (
        f"Shapes don't match: light {light_image.shape}, "
        f"bias {bias_image.shape}, "
        f"dark {dark_image.shape}, "
        f"flat {flat_image.shape}"
    )

    assert "EXPOSURE" in light_metadata.header, (
        f"### file {light_metadata.filename} does not have EXPOSURE tag"
    )
    assert "EXPOSURE" in dark_metadata.header, (
        f"### file {dark_metadata.filename} does not have EXPOSURE tag"
    )
    assert "EXPOSURE" in flat_metadata.header, (
        f"### file {flat_metadata.filename} does not have EXPOSURE tag"
    )

    light_dark_scale = (
        light_metadata.header["EXPOSURE"] / dark_metadata.header["EXPOSURE"]
    )
    flat_dark_scale = (
        flat_metadata.header["EXPOSURE"] / dark_metadata.header["EXPOSURE"]
    )

    image = (light_image - bias_image) - light_dark_scale * (dark_image - bias_image)
    flat_image = (flat_image - bias_image) - flat_dark_scale * (dark_image - bias_image)
    assert flat_image is not None
    image = np.divide(
        image, flat_image, out=np.zeros_like(image), where=flat_image != 0
    )

    if rotate:
        image = np.rot90(image, 2)

    return image


@cache
def get_subframe_for_calibration_stack(
    stack_id: StackHash, width: int
) -> tuple[ImageMetadata, np.ndarray]:
    if stack_id == None:
        return None, None

    stack = context.calibration_stacks[stack_id]
    assert stack is not None
    frame = stack.all_metadata()[stack.all_frames()[0]]
    image = resize_array(read_image(frame.filename), width).astype(np.float32)

    return frame, image


def calibrate_subframe(
    light_frame: ImageMetadata, dependencies: DependenciesDict, width: int = 2048,
) -> np.ndarray:
    assert light_frame.plate_solution is not None
    rotate = light_frame.plate_solution.rotation_angle < 0.0

    light_image = resize_array(read_image(light_frame.filename), width).astype(
        np.float32
    )

    def get_stack_id(key) -> StackHash | None:
        dep = dependencies.get(key)
        if isinstance(dep, StackIdDependency):
            return dep.stack_id
        return None

    bias_frame, bias_image = get_subframe_for_calibration_stack(
        get_stack_id("bias"), width
    )
    dark_frame, dark_image = get_subframe_for_calibration_stack(
        get_stack_id("dark"), width
    )
    flat_frame, flat_image = get_subframe_for_calibration_stack(
        get_stack_id("flat"), width
    )

    return calibrate_image(
        light_frame,
        light_image,
        bias_frame,
        bias_image,
        dark_frame,
        dark_image,
        flat_frame,
        flat_image,
        rotate,
    )
