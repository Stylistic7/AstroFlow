from __future__ import annotations

import dataclasses
from contextvars import ContextVar
from pathlib import Path
import numpy as np
from astropy.io import fits
from frozendict import frozendict

from hashing import hash_list
from metadata.classes import ImageMetadata, MetadataDict
from stacks.composite_stack import BaseStack, Stack, CalibrationStack
from stacks.run import BaseRun, Run, CalibrationRun, RunsDict
from units import ArcsecondsPerPixel


# When True, `downscale_stack` skips file existence checks and
# file writes — it just constructs the new Stack metadata. Used by
# the web server's `_ensure_pipelines_traced` HTTP path to avoid
# blocking the request on per-frame `stat()` over network-mounted
# storage (~1 ms × ~tens of thousands of frames adds up to ~100 s).
# The background Ray dispatch leaves this False; that trace does
# the actual file work as part of pipeline execution.
TRACE_ONLY = ContextVar("downscale_stack_trace_only", default=False)


def downscale_fits(
    in_path: str | Path, out_path: str | Path, *, K: int, overwrite: bool = True
) -> None:
    """
    Power-of-two downscale that mimics *in-camera binning* for WBPP:

    - Reads the frame in the camera's physical uint16 ADU domain (Astropy applies BZERO/BSCALE).
    - Downscales by block-mean (so output stays in the same ADU range as the input).
    - Writes a FITS that looks like a native camera frame:
        * stored dtype: int16
        * BSCALE=1, BZERO=32768  (unsigned16 convention)
    - Drops SIP/WCS by construction (minimal safe header copy).
    """
    if K <= 1 or (K & (K - 1)) != 0:
        raise ValueError(f"K must be a power of two >= 1, got {K}")

    in_path = Path(in_path)
    out_path = Path(out_path)

    with fits.open(in_path) as hdul:
        src = hdul[0]
        h0 = src.header
        data = src.data

    if data is None or data.ndim != 2:
        raise ValueError("Expected 2D primary image in HDU[0]")

    h, w = data.shape

    # Trim edges so dimensions are divisible by K (at most K-1 pixels lost).
    h_trim = h - (h % K)
    w_trim = w - (w % K)
    data = data[:h_trim, :w_trim]
    h, w = data.shape

    # Astropy applies BZERO/BSCALE and unsigned16 convention automatically:
    # img is in "physical ADU" space, typically [0..65535] for these ZWO frames.
    img = np.asarray(data, dtype=np.float32)
    assert np.isfinite(img).all()
    assert img.min() >= -1e-3  # should be >= 0 in practice
    assert img.max() <= 65535 + 1e-3

    y = (
        img.reshape(h // K, K, w // K, K)
        .mean(axis=(1, 3))
        .astype(np.float32, copy=False)
    )

    # Range invariants for mean downscale in ADU space
    assert np.isfinite(y).all()
    assert y.min() >= img.min() - 1e-3
    assert y.max() <= img.max() + 1e-3

    # Clamp to legal uint16 physical range, then encode as FITS unsigned16 convention:
    # stored int16 with BZERO=32768, BSCALE=1
    y = np.clip(y, 0.0, 65535.0, out=y)
    q = np.rint(y - 32768.0).astype(np.int32, copy=False)
    q = np.clip(q, -32768, 32767).astype(np.int16, copy=False)

    # Fresh primary HDU to avoid header/data mismatches.
    dst = fits.PrimaryHDU(data=q)
    hdr = dst.header
    hdr["BSCALE"] = 1.0
    hdr["BZERO"] = 32768.0

    # Copy only safe metadata (no structural/scaling/WCS/SIP families).
    keep = (
        "CREATOR",
        "INSTRUME",
        "GUIDECAM",
        "TELESCOP",
        "RIG",
        "IMAGETYP",
        "FILTER",
        "EXPOSURE",
        "EXPTIME",
        "DATE-OBS",
        "DATE-END",
        "TIMESYS",
        "GAIN",
        "OFFSET",
        "EGAIN",
        "CCD-TEMP",
        "SET-TEMP",
        "FOCALLEN",
        "FOCUSPOS",
        "DONTSOLV",
    )
    for k in keep:
        if k in h0:
            hdr[k] = h0[k]

    # Update camera-geometry metadata to match "binned" appearance.
    for k in ("XBINNING", "YBINNING", "CCDXBIN", "CCDYBIN"):
        if k in h0:
            try:
                hdr[k] = int(round(float(h0[k]) * K))
            except (KeyError, ValueError, TypeError):
                pass

    for k in ("XPIXSZ", "YPIXSZ"):
        if k in h0:
            try:
                hdr[k] = float(h0[k]) * K
            except (KeyError, ValueError, TypeError):
                pass

    for k in ("XORGSUBF", "YORGSUBF"):
        if k in h0:
            try:
                hdr[k] = float(h0[k]) / K
            except (KeyError, ValueError, TypeError):
                pass

    dst.writeto(out_path, overwrite=overwrite, output_verify="exception")


def _downscaled_frame_content_hash(orig_content_hash: str, factor: int) -> str:
    return hash_list([orig_content_hash, f"downscale-K{factor}"])


def _downscale_frame(m: ImageMetadata, factor: int, dest_dir: Path) -> ImageMetadata:
    """Build an ImageMetadata pointing at the downscaled FITS for
    frame `m`. The on-disk write itself is done in parallel by
    `downscale_stack`'s ThreadPool — this function only constructs
    the metadata. `stable_hash` is preserved; `content_hash` is
    derived from `(orig.content_hash, factor)` so it's stable
    across runs."""
    new_h = _downscaled_frame_content_hash(m.content_hash, factor)
    new_path = dest_dir / f"{new_h}.fits"
    return ImageMetadata(
        content_hash=new_h,
        stable_hash=m.stable_hash,
        filename=new_path,
        header=m.header,
        width_pixels=m.width_pixels // factor,
        height_pixels=m.height_pixels // factor,
        channels=m.channels,
        dtype=m.dtype,
        # Plate solution is in sky coords; CRPIX/CD scaling for the
        # smaller pixel grid would be more accurate, but WBPP only
        # uses RA/DEC as a hint for re-solving on the downscaled
        # frame, so the body in `_copy_frames_with_plate_solution`
        # writes RA/DEC hints only when frame width disagrees with
        # the plate-solution header's NAXIS1.
        plate_solution=m.plate_solution,
    )


def _downscale_run(run: BaseRun, factor: int, dest_dir: Path) -> BaseRun:
    """Build a new Run whose frames point at downscaled files.
    `stable_hash` preserved (identity); `content_hash` derived
    from the new frame hashes (so the run is materially different
    for any downstream op that includes it in a call_hash)."""
    new_meta = MetadataDict({})
    new_frames = []
    for orig_h, m in run.metadata.items():
        new_m = _downscale_frame(m, factor, dest_dir)
        new_meta[new_m.content_hash] = new_m
        new_frames.append(new_m.content_hash)

    overrides = dict(
        frames=tuple(new_frames),
        metadata=new_meta,
        content_hash=hash_list(sorted(new_frames)),
        width_pixels=run.width_pixels // factor,
        height_pixels=run.height_pixels // factor,
    )
    # Light Runs carry AstrometricAttributes — scale image_scale.
    if hasattr(run, "image_scale"):
        overrides["image_scale"] = ArcsecondsPerPixel(
            float(run.image_scale) * factor
        )
    return dataclasses.replace(run, **overrides)


def downscale_stack(stack: BaseStack, factor: int) -> BaseStack:
    """Block-mean-downscale every frame in `stack` by `factor`.

    Returns a new Stack with the SAME `stable_hash` as the input
    (preserves the user-facing identity used by tap routing —
    `/stack/<original_hash>/...` still surfaces taps from
    pipelines that downscale before stacking) and a NEW
    `content_hash` derived from the downscaled frames' content
    (so downstream ops that include the stack in their call_hash
    produce distinct outputs).

    REGULAR FUNCTION, NOT @node. Runs at trace time so the
    returned Stack is a literal value — the pipeline's downstream
    `stack_files(stack, ...)` call then sees `stack` as a literal
    arg, satisfying the tracer's "literals before deps" contract
    in `dependency_graph.tracing.record()`. (Making this @node
    would put the downscaled Stack into the data_deps list, and
    the tracer reorders deps to the END of `make_fn_args`, so
    `stack` would land in the wrong positional slot at runtime.)

    Frame-level downscaling is parallelized via a ThreadPool;
    per-frame output paths are content-addressed
    (`<output_dir>/downscale_stack/<derived_hash>.fits`), so
    re-runs skip already-written files and re-traces are cheap.

    Pipeline-agnostic — works on `Stack` (light frames) and
    `CalibrationStack` (bias/dark/flat) the same way."""
    from context import context
    from concurrent.futures import ThreadPoolExecutor

    if factor == 1:
        return stack

    dest_dir = Path(context.output_dir) / "downscale_stack"
    if not TRACE_ONLY.get():
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Pre-compute every frame's downscale target path so the
        # ThreadPool can run them in parallel. Skip already-written.
        # Existence checks over network storage are the load-bearing
        # cost here; TRACE_ONLY mode skips this loop entirely for
        # the HTTP `pipeline_outputs` path, which only needs the
        # constructed Stack metadata to enumerate tap points.
        jobs = []
        for run in stack.contents.values():
            for m in run.metadata.values():
                new_h = _downscaled_frame_content_hash(m.content_hash, factor)
                new_path = dest_dir / f"{new_h}.fits"
                if not new_path.exists():
                    jobs.append((m.filename, new_path))
        if jobs:
            with ThreadPoolExecutor(max_workers=16) as ex:
                list(ex.map(
                    lambda j: downscale_fits(j[0], j[1], K=factor),
                    jobs,
                ))

    new_runs = RunsDict()
    for run_key, run in stack.contents.items():
        new_runs[run_key] = _downscale_run(run, factor, dest_dir)

    overrides = dict(
        contents=new_runs,
        content_hash=hash_list(sorted(r.content_hash for r in new_runs.values())),
        width_pixels=stack.width_pixels // factor,
        height_pixels=stack.height_pixels // factor,
    )
    if hasattr(stack, "image_scale"):
        overrides["image_scale"] = ArcsecondsPerPixel(
            float(stack.image_scale) * factor
        )
    return dataclasses.replace(stack, **overrides)
