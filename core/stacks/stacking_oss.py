"""OSS stacking: calibration + cosmic ray removal + astroalign + sigma-clipped combine.

Replaces PixInsight's WBPP.
"""
import os
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


def _sigma_clipped_mean(frames):
    """Stack frames with 3-sigma rejection, averaging the survivors."""
    if len(frames) == 1:
        return frames[0]
    stack = np.array(frames, dtype=np.float32)
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0)
    std[std == 0] = 1.0
    mask = np.abs(stack - mean) < 3.0 * std
    count = np.sum(mask, axis=0).astype(np.float32)
    count[count == 0] = 1.0
    return np.sum(stack * mask, axis=0) / count


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

    stack = resolve_calibration(stack, config.calibration_stacks, config.decisions)
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
            cal_stack = config.calibration_stacks[dep.stack_id]
            cal_data = _load_calibration(cal_stack, config.quality, OSS)
            if cal_data is None:
                cal_data = _load_calibration(cal_stack, config.quality, config.implementation)
            if cal_data is None:
                continue
            if cal_stack.image_type == "bias": bias = cal_data
            elif cal_stack.image_type == "dark": dark = cal_data
            elif cal_stack.image_type == "flat": flat = cal_data

    # Collect, calibrate, and optionally clean each frame.
    downscale_factor = DOWNSCALE_FACTORS[config.quality]
    is_light = isinstance(stack, Stack)
    frames = []
    for run in stack.contents.values():
        for frame_meta in run.metadata.values():
            if not os.path.exists(frame_meta.filename):
                log.warning("Frame not found, skipping", filename=str(frame_meta.filename))
                continue
            data = read_image(frame_meta.filename).astype(np.float32)
            if downscale_factor > 1:
                K = downscale_factor
                h, w = data.shape
                data = data[: h - h % K, : w - w % K]
                h, w = data.shape
                data = data.reshape(h // K, K, w // K, K).mean(axis=(1, 3))
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
            frames.append(data)

    if len(frames) < config.min_frames_in_stack:
        raise ValueError(f"Stack {stack} has only {len(frames)} frames (minimum {config.min_frames_in_stack})")

    # Register light frames to the first one. Skip for calibration stacks
    # (already aligned — same sensor, same session).
    if is_light and len(frames) > 1:
        reference = frames[0]
        registered = [reference]
        for i, frame in enumerate(frames[1:], start=1):
            try:
                aligned, _ = aa_register(frame, reference)
                registered.append(np.nan_to_num(aligned, nan=0.0))
            except Exception as e:
                log.warning("Failed to register frame, skipping", index=i, error=str(e))
        frames = registered

    result = _sigma_clipped_mean(frames)

    header = pyfits.Header()
    header["IMAGETYP"] = f"Master {stack.image_type}"
    header["STBLHASH"] = stack.stable_hash
    header["NCOMBINE"] = len(frames)
    header["PIPELINE"] = "AstroFlow-OSS"
    if isinstance(stack, Stack):
        header["FILTER"] = stack.filter
        header["RA"] = stack.center[0]
        header["DEC"] = stack.center[1]

    pyfits.PrimaryHDU(data=result.astype(np.float32), header=header).writeto(
        str(dest_filename), overwrite=True,
    )
    log.info("OSS stacking complete", output=str(dest_filename), frames=len(frames))
    return WithSidecar(FitsFilePath(dest_filename))
