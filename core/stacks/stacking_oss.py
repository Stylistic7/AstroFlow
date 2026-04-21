"""OSS stacking: calibration + cosmic ray removal + astroalign + sigma-clipped combine.

Replaces PixInsight's WBPP.

Memory is bounded by a constant (a few same-shape float32 arrays),
independent of stack size. The combine is a two-pass streaming
algorithm:
  * Pass 1: preprocess + register each frame, write it to a temp file,
            and update per-pixel running sum and sum-of-squares.
  * Pass 2: re-read each temp file, apply a 3σ mask against the mean
            and std computed in pass 1, and accumulate the masked sum.
  * Final: masked_sum / count.

This preserves the sigma-clipped-mean behavior of the old in-memory
version while dropping peak RSS from O(N_frames) to O(1).
"""
import os
import shutil
import tempfile
import numpy as np
import structlog

import astropy.io.fits as pyfits
from astroalign import register as aa_register
from astroscrappy import detect_cosmics

from decorators import node, create_thumbnail, create_tile_pyramid
from file_io.generic import read_image, read_header
from file_io.types import FitsFilePath, WithSidecar
from previews.types import OSS, MASTER, DOWNSCALED_MASTER, DOWNSCALE_FACTORS
from products import makes_image
from products.registry import get_filename_fn, MediaType
from stacks.artifacts import stack_filename_fits
from stacks.composite_stack import BaseStack, Stack, CalibrationStack
from stacks.types import StackIdDependency
from decisions.resolve import resolve_calibration, resolve_decisions
from decisions.compute_graph import ALL_EXCLUSION_TYPES
from decisions.settings import ProcessingConfig

log = structlog.get_logger()


def _load_calibration(cal_stack, quality, implementation):
    fn = get_filename_fn(MediaType.IMAGE, type(cal_stack), quality, implementation)
    path = fn(cal_stack) if fn else None
    return read_image(path) if path and os.path.exists(path) else None


def _calibrate(data, bias, dark, flat):
    if bias is not None:
        data = data - bias
    if dark is not None:
        data = data - (dark - bias if bias is not None else dark)
    if flat is not None:
        flat_norm = flat / np.mean(flat)
        flat_norm[flat_norm == 0] = 1.0
        data = data / flat_norm
    return data


def _downscale(data, K):
    if K <= 1:
        return data
    h, w = data.shape
    data = data[: h - h % K, : w - w % K]
    h, w = data.shape
    return data.reshape(h // K, K, w // K, K).mean(axis=(1, 3))


def _preprocess_frame(frame_meta, bias, dark, flat, downscale_factor, is_light):
    """Load one raw frame, downscale, calibrate, and (for lights) remove
    cosmic rays. Returns a float32 ndarray or None if the file is missing."""
    if not os.path.exists(frame_meta.filename):
        log.warning("Frame not found, skipping", filename=str(frame_meta.filename))
        return None
    data = read_image(frame_meta.filename).astype(np.float32)
    data = _downscale(data, downscale_factor)
    data = _calibrate(data, bias, dark, flat)
    if is_light:
        header = read_header(frame_meta.filename)
        try:
            _, data = detect_cosmics(
                data,
                gain=float(header.get("EGAIN", header.get("GAIN", 1.0))),
                readnoise=float(header.get("RDNOISE", header.get("READNOIS", 10.0))),
                sigclip=4.5, niter=3,
            )
        except Exception as e:
            log.warning("Cosmic ray removal failed", error=str(e))
    return data


def _streaming_sigma_clipped_mean(preprocessed_frames, is_light):
    """Two-pass sigma-clipped mean with streaming registration.

    ``preprocessed_frames`` is an iterator of (index, ndarray) pairs — each
    already calibrated, downscaled, and cosmic-ray-cleaned. For light
    frames, all frames after the first are registered to the first.

    Pass 1 writes each (registered) frame to a temp .npy, accumulating
    sum / sum-of-squares. Pass 2 reloads each temp, applies the 3σ
    mask, and accumulates the masked sum + per-pixel surviving count.

    Returns (result, num_combined, reference_shape).
    """
    temp_dir = tempfile.mkdtemp(prefix="stack_oss_")
    try:
        reference = None
        sum_arr = None
        sum_sq = None
        n = 0
        for i, frame in preprocessed_frames:
            if is_light and reference is not None:
                try:
                    aligned, _ = aa_register(frame, reference)
                    frame = np.nan_to_num(aligned, nan=0.0).astype(np.float32, copy=False)
                except Exception as e:
                    log.warning("Failed to register frame, skipping", index=i, error=str(e))
                    continue
            else:
                frame = frame.astype(np.float32, copy=False)
                if is_light:
                    reference = frame

            if sum_arr is None:
                sum_arr = frame.astype(np.float32, copy=True)
                sum_sq = frame * frame
            else:
                sum_arr += frame
                sum_sq += frame * frame

            np.save(os.path.join(temp_dir, f"{n}.npy"), frame, allow_pickle=False)
            n += 1

        if n == 0:
            return None, 0, None
        if n == 1:
            one = np.load(os.path.join(temp_dir, "0.npy"))
            return one, 1, one.shape

        # Pass 1 statistics. var is clamped at 0 to absorb floating-point
        # negatives (sum_sq/n can fall a hair below mean² when the samples
        # are nearly identical).
        mean = sum_arr / n
        var = np.maximum(sum_sq / n - mean * mean, 0.0)
        std = np.sqrt(var, dtype=np.float32)
        std[std == 0] = 1.0

        # Release pass-1 scratch before pass 2 allocates its own.
        del sum_arr, sum_sq, var

        # Pass 2: masked combine.
        masked_sum = np.zeros_like(mean)
        count = np.zeros(mean.shape, dtype=np.float32)
        for i in range(n):
            frame = np.load(os.path.join(temp_dir, f"{i}.npy"))
            mask = np.abs(frame - mean) < 3.0 * std
            masked_sum += frame * mask
            count += mask

        count[count == 0] = 1.0
        return masked_sum / count, n, mean.shape
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@node(auto_memory=True)
@makes_image(CalibrationStack, OSS, [MASTER, DOWNSCALED_MASTER], stack_filename_fits)
@makes_image(Stack, OSS, [MASTER, DOWNSCALED_MASTER], stack_filename_fits)
@create_thumbnail
@create_tile_pyramid
def stack_files_oss(
    stack: BaseStack,
    config: ProcessingConfig,
) -> WithSidecar[FitsFilePath]:
    from products import ensure_producers_registered
    ensure_producers_registered()

    stack = resolve_calibration(stack, config.get_calibration_stacks(), config.decisions)
    if config.decisions and isinstance(stack, Stack):
        filtered = resolve_decisions(stack, ALL_EXCLUSION_TYPES, config.decisions, config.min_frames_in_stack)
        if filtered is None:
            raise ValueError(f"Stack {stack} has no frames after applying decisions")
        stack = filtered

    dest_filename = get_filename_fn(MediaType.IMAGE, type(stack), config.quality, OSS)(stack)
    os.makedirs(os.path.dirname(dest_filename), exist_ok=True)
    log.info("OSS stacking", stack=str(stack), dest=str(dest_filename))

    # Load master calibration frames from whichever implementation has them.
    bias = dark = flat = None
    for run in stack.contents.values():
        for dep in run.dependencies.values():
            if not isinstance(dep, StackIdDependency):
                continue
            cal_stack = config.get_calibration_stacks()[dep.stack_id]
            cal_data = _load_calibration(cal_stack, config.quality, OSS)
            if cal_data is None:
                cal_data = _load_calibration(cal_stack, config.quality, config.implementation)
            if cal_data is None:
                continue
            if cal_stack.image_type == "bias": bias = cal_data
            elif cal_stack.image_type == "dark": dark = cal_data
            elif cal_stack.image_type == "flat": flat = cal_data

    downscale_factor = DOWNSCALE_FACTORS[config.quality]
    is_light = isinstance(stack, Stack)

    # Pre-count frames so we can enforce min_frames_in_stack before
    # starting the expensive combine. The generator below is what actually
    # drives the streaming combine.
    frame_metas = [
        fm
        for run in stack.contents.values()
        for fm in run.metadata.values()
    ]
    usable = [fm for fm in frame_metas if os.path.exists(fm.filename)]
    if len(usable) < config.min_frames_in_stack:
        raise ValueError(
            f"Stack {stack} has only {len(usable)} frames "
            f"(minimum {config.min_frames_in_stack})"
        )

    def frame_iter():
        for i, fm in enumerate(usable):
            data = _preprocess_frame(fm, bias, dark, flat, downscale_factor, is_light)
            if data is not None:
                yield i, data

    result, num_combined, _ = _streaming_sigma_clipped_mean(frame_iter(), is_light)
    if result is None:
        raise ValueError(f"Stack {stack} produced no usable frames")

    header = pyfits.Header()
    header["IMAGETYP"] = f"Master {stack.image_type}"
    header["STBLHASH"] = stack.stable_hash
    header["NCOMBINE"] = num_combined
    header["PIPELINE"] = "AstroFlow-OSS"
    if isinstance(stack, Stack):
        header["FILTER"] = stack.filter
        header["RA"] = stack.center[0]
        header["DEC"] = stack.center[1]

    pyfits.PrimaryHDU(data=result.astype(np.float32), header=header).writeto(
        str(dest_filename), overwrite=True,
    )
    log.info("OSS stacking complete", output=str(dest_filename), frames=num_combined)
    return WithSidecar(FitsFilePath(dest_filename))
