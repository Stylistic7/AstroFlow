"""OSS stacking: ccdproc calibration + astroalign registration + sigma-clipped combine.

Replaces PixInsight's WBPP with professional astronomy Python libraries.
"""
import os
import tempfile
import numpy as np
import structlog
from pathlib import Path

import astropy.io.fits as pyfits
import astropy.units as u
from ccdproc import CCDData, subtract_bias, subtract_dark, flat_correct, combine
from astroalign import register as aa_register
from astroscrappy import detect_cosmics

from decorators import node, create_thumbnail, create_tile_pyramid
from file_io.generic import read_image, read_header
from file_io.types import FitsFilePath, WithSidecar
from previews.types import QualityLevel, OSS, MASTER, DOWNSCALED_MASTER, DOWNSCALE_FACTORS
from products import makes_image
from products.registry import get_filename_fn, MediaType
from stacks.artifacts import stack_filename_fits
from stacks.composite_stack import BaseStack, Stack, CalibrationStack
from stacks.run import Run
from stacks.types import StackIdDependency
from decisions.resolve import resolve_calibration, resolve_decisions
from decisions.compute_graph import ALL_EXCLUSION_TYPES
from decisions.settings import ProcessingConfig

log = structlog.get_logger()


def _load_master_calibration(cal_stack, quality, implementation):
    """Load a master calibration frame (bias/dark/flat) as a numpy array."""
    filename_fn = get_filename_fn(MediaType.IMAGE, type(cal_stack), quality, implementation)
    if filename_fn is None:
        return None
    path = filename_fn(cal_stack)
    if path is None or not os.path.exists(path):
        return None
    return read_image(path)


def _calibrate_frame(frame_path, master_bias, master_dark, master_flat, exposure):
    """Apply bias/dark/flat calibration to a single frame."""
    data = read_image(frame_path).astype(np.float32)

    if master_bias is not None:
        data = data - master_bias

    if master_dark is not None:
        # Scale dark by exposure ratio if we have exposure info
        dark_corrected = master_dark
        if master_bias is not None:
            dark_corrected = master_dark - master_bias
        data = data - dark_corrected

    if master_flat is not None:
        flat_norm = master_flat / np.mean(master_flat)
        flat_norm[flat_norm == 0] = 1.0
        data = data / flat_norm

    return data


def _remove_cosmic_rays(data, header):
    """Remove cosmic rays using LA Cosmic algorithm."""
    gain = float(header.get("EGAIN", header.get("GAIN", 1.0)))
    readnoise = float(header.get("RDNOISE", header.get("READNOIS", 10.0)))
    mask, cleaned = detect_cosmics(
        data.astype(np.float32),
        gain=gain,
        readnoise=readnoise,
        sigclip=4.5,
        niter=3,
    )
    return cleaned


def _register_frames(frames, reference_idx=0):
    """Register all frames to a reference using star pattern matching."""
    reference = frames[reference_idx]
    registered = [reference]

    for i, frame in enumerate(frames):
        if i == reference_idx:
            continue
        try:
            aligned, footprint = aa_register(frame, reference)
            # Replace NaN/invalid pixels from transform with zeros
            aligned = np.nan_to_num(aligned, nan=0.0)
            registered.append(aligned)
        except Exception as e:
            log.warning("Failed to register frame, skipping", index=i, error=str(e))
            continue

    return registered


def _combine_frames(registered_frames):
    """Sigma-clipped average combination."""
    if len(registered_frames) == 1:
        return registered_frames[0]

    # Stack into 3D array and do sigma-clipped mean
    stack = np.array(registered_frames, dtype=np.float32)

    # Simple sigma clipping: compute mean and std, reject outliers, re-average
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0)
    std[std == 0] = 1.0  # avoid div by zero

    mask = np.abs(stack - mean) < 3.0 * std  # 3-sigma clip
    # Masked average
    masked_sum = np.sum(stack * mask, axis=0)
    masked_count = np.sum(mask, axis=0).astype(np.float32)
    masked_count[masked_count == 0] = 1.0
    result = masked_sum / masked_count

    return result


@node
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

    # Resolve calibration (same as PixInsight path).
    stack = resolve_calibration(stack, config.calibration_stacks, config.decisions)

    # Apply exclusion decisions for light stacks.
    if config.decisions and isinstance(stack, Stack):
        filtered = resolve_decisions(stack, ALL_EXCLUSION_TYPES, config.decisions, config.min_frames_in_stack)
        if filtered is None:
            raise ValueError(f"Stack {stack} has no frames after applying decisions")
        stack = filtered

    # Determine output file.
    filename_fn = get_filename_fn(MediaType.IMAGE, type(stack), config.quality, OSS) or stack_files_oss.filename_fn
    dest_filename = filename_fn(stack)
    os.makedirs(os.path.dirname(dest_filename), exist_ok=True)
    log.info("OSS stacking", stack=str(stack), dest=str(dest_filename))

    # Load master calibration frames.
    # Try OSS calibration first, fall back to PixInsight calibration.
    master_bias = None
    master_dark = None
    master_flat = None

    for run in stack.contents.values():
        for dep_name, dep in run.dependencies.items():
            if isinstance(dep, StackIdDependency):
                cal_stack = config.calibration_stacks[dep.stack_id]
                for impl in [OSS, config.implementation]:
                    cal_data = _load_master_calibration(cal_stack, config.quality, impl)
                    if cal_data is not None:
                        if cal_stack.image_type == "bias":
                            master_bias = cal_data
                        elif cal_stack.image_type == "dark":
                            master_dark = cal_data
                        elif cal_stack.image_type == "flat":
                            master_flat = cal_data
                        break

    # Collect all light frames.
    all_frames = []
    for run in stack.contents.values():
        for frame_meta in run.metadata.values():
            filename = frame_meta.filename
            if not os.path.exists(filename):
                log.warning("Frame not found, skipping", filename=str(filename))
                continue

            header = read_header(filename)

            # Downscale if needed.
            downscale_factor = DOWNSCALE_FACTORS[config.quality]
            if downscale_factor > 1:
                from stacks.downscale import downscale_array
                data = read_image(filename)
                data = downscale_array(data, downscale_factor)
            else:
                data = read_image(filename).astype(np.float32)

            # Calibrate.
            if master_bias is not None:
                data = data - master_bias
            if master_dark is not None:
                dark = master_dark - master_bias if master_bias is not None else master_dark
                data = data - dark
            if master_flat is not None:
                flat_norm = master_flat / np.mean(master_flat)
                flat_norm[flat_norm == 0] = 1.0
                data = data / flat_norm

            # Cosmic ray removal (skip for calibration stacks).
            if isinstance(stack, Stack):
                try:
                    data = _remove_cosmic_rays(data, header)
                except Exception as e:
                    log.warning("Cosmic ray removal failed, using uncleaned", error=str(e))

            all_frames.append(data)

    if len(all_frames) < config.min_frames_in_stack:
        raise ValueError(
            f"Stack {stack} contains only {len(all_frames)} frames "
            f"(minimum {config.min_frames_in_stack})"
        )

    # Register.
    log.info("Registering frames", count=len(all_frames))
    if isinstance(stack, Stack) and len(all_frames) > 1:
        registered = _register_frames(all_frames)
    else:
        registered = all_frames

    # Combine.
    log.info("Combining frames", count=len(registered))
    result = _combine_frames(registered)

    # Write output as FITS.
    header = pyfits.Header()
    header["IMAGETYP"] = f"Master {stack.image_type}"
    header["STBLHASH"] = stack.stable_hash
    header["NCOMBINE"] = len(registered)
    header["PIPELINE"] = "AstroFlow-OSS"

    if isinstance(stack, Stack):
        header["FILTER"] = stack.filter
        header["RA"] = stack.center[0]
        header["DEC"] = stack.center[1]

    hdu = pyfits.PrimaryHDU(data=result.astype(np.float32), header=header)
    hdu.writeto(str(dest_filename), overwrite=True)

    log.info("OSS stacking complete", output=str(dest_filename), frames=len(registered))
    return WithSidecar(FitsFilePath(dest_filename))
