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
    """Preview-grade calibration: apply WHATEVER components are present.

    This used to require bias AND dark AND flat, silently returning the
    raw light otherwise. The unified CALIBRATION_TOPOLOGY (2026-06) gives
    lights `("dark", "flat")` and NEVER a bias (Landmann §8.2 — the dark
    carries the bias signal), so after that change the all-or-nothing
    guard made every flat-corrected Run preview serve the raw frame:
    byte-identical carousel entries for every candidate flat, a no-op
    picker that looked like a working one (found 2026-09-04).

    Semantics per component, preview-grade (thumbnails choose FLATS —
    vignetting and dust — not photometry):
      * bias absent: treated as zero; the dark then carries bias+current,
        correct for exposure-MATCHED darks. Exposure scaling of a
        bias-inclusive dark is approximate; fine at preview scale.
      * dark absent: no dark subtraction (and no dark term in the flat).
      * flat absent: no division — the honest "uncorrected" preview.
    """
    assert light_image is not None

    frames = [f for f in (light_image, bias_image, dark_image, flat_image)
              if f is not None]
    # Independent resize-to-width can leave light and cal frames with
    # heights that differ by ±1 row when their native heights straddle an
    # integer-truncation boundary (e.g. a sensor crop of 2 rows pushes
    # 6388 → 6386 native, which rounds to 1365 vs. 1366 at width 2048).
    # Crop everything to the minimum common shape — this is a thumbnail
    # preview, sub-pixel registration is irrelevant.
    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    light = light_image[:h, :w].astype(np.float32)
    bias = bias_image[:h, :w] if bias_image is not None else np.float32(0.0)
    dark = dark_image[:h, :w] if dark_image is not None else None
    flat = flat_image[:h, :w] if flat_image is not None else None

    def _exposure(md):
        assert md is not None and "EXPOSURE" in md.header, (
            f"### file {md and md.filename} does not have EXPOSURE tag"
        )
        return md.header["EXPOSURE"]

    image = light - bias
    if dark is not None:
        scale = _exposure(light_metadata) / _exposure(dark_metadata)
        image = image - scale * (dark - bias)

    if flat is not None:
        flat_signal = flat - bias
        if dark is not None:
            scale = _exposure(flat_metadata) / _exposure(dark_metadata)
            flat_signal = flat_signal - scale * (dark - bias)
        # Normalize so division rescales SHAPE (vignetting/dust) without
        # moving the preview's overall level.
        med = float(np.median(flat_signal[flat_signal > 0])) if np.any(flat_signal > 0) else 1.0
        flat_signal = flat_signal / max(med, 1e-6)
        image = np.divide(
            image, flat_signal, out=np.zeros_like(image),
            where=flat_signal != 0,
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
