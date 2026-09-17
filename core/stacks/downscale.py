
import dataclasses
from pathlib import Path
import numpy as np
from astropy.io import fits
from frozendict import frozendict

from beartype import beartype

from decorators import node
from file_io.generic import read_header, read_image
from file_io.types import WithSidecar, XisfFilePath
from file_io.xisf_file_io import write_xisf_image
from hashing import hash_list
from metadata.classes import ImageMetadata, MetadataDict
from stacks.composite_stack import Stack
from stacks.run import BaseRun, RunsDict
from units import ArcsecondsPerPixel


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


def downscale_dest_dir() -> Path:
    """Where `downscale_stack` scatters its derived frames.

    Lives OUTSIDE the node on purpose. `downscale_stack` used to read
    `context.output_dir` in its own body, which made its cached result
    depend on a value its cache key could not see — the same shape of bug
    that cost 2026-08-18 (a relocated library invalidated nothing). Callers
    are pipelines and builders, which may read context; the resolved path
    is then an ordinary literal argument and enters the call hash, so
    moving `output_dir` correctly re-keys the derived frames.
    """
    from context import context
    return Path(context.output_dir) / "downscale_stack"


@beartype
@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class DownscaledStack(Stack):
    """A stack seen at 1/`factor` resolution — the SAME entity as `source`
    (`stable_hash` inherited, so `/stack/<hash>/…` surfaces this pipeline's
    taps too) whose frames are block-mean copies that `downscale_stack`
    writes. Every field is derived from the source in the pipeline body, at
    trace time, so `stack_files` receives a real stack with both hashes and
    keys on it exactly as it keys on any stack; nothing downstream ever
    sees a handle for the stack itself.

    `content_hash` is `hash(source.content_hash, factor, downscale_stack's
    source)`: the frames' content is a function of those three and nothing
    else — in particular NOT of where they land on disk — and editing the
    downscaler re-keys every master built on its frames."""
    factor: int
    source: Stack

    @classmethod
    def of(cls, stack: Stack, factor: int, dest_dir: Path) -> "DownscaledStack":
        if factor < 1 or (factor & (factor - 1)) != 0:
            raise ValueError(f"factor must be a power of two >= 1, got {factor}")
        dest_dir = Path(dest_dir)
        new_runs = RunsDict()
        for run_key, run in stack.contents.items():
            new_runs[run_key] = _downscale_run(run, factor, dest_dir)
        values = {f.name: getattr(stack, f.name)
                  for f in dataclasses.fields(stack) if f.init}
        values.update(
            contents=new_runs,
            content_hash=hash_list([stack.content_hash, factor, downscale_stack._fn_hash]),
            width_pixels=stack.width_pixels // factor,
            height_pixels=stack.height_pixels // factor,
            image_scale=ArcsecondsPerPixel(float(stack.image_scale) * factor),
        )
        return cls(**values, factor=factor, source=stack)


@node
def downscale_stack(stack: DownscaledStack) -> tuple[Path, ...]:
    """Write the frames a `DownscaledStack` names: each source frame
    block-mean downscaled by `stack.factor` to the content-addressed path
    its downscaled metadata already carries
    (`<output_dir>/downscale_stack/<derived_hash>.fits`). Returns those
    paths. Consumers of the frames (`stack_files`) order after this node
    with `_after=`: the stack they read is the literal view, the files are
    this node's side effect.

    Idempotent per file (an existing frame is not rewritten); frames are
    written in parallel. Works on light stacks; calibration MASTERS are
    downscaled as images by `downscale_master` instead of re-integrating
    downscaled cal frames, so the full-resolution masters are shared with
    the full-resolution pipeline."""
    from concurrent.futures import ThreadPoolExecutor

    jobs = []
    for run_key, run in stack.contents.items():
        source_run = stack.source.contents[run_key]
        for m, src in zip(run.metadata.values(), source_run.metadata.values()):
            if not Path(m.filename).exists():
                jobs.append((src.filename, m.filename))
    if jobs:
        Path(jobs[0][1]).parent.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(lambda j: downscale_fits(j[0], j[1], K=stack.factor), jobs))
    return tuple(Path(m.filename) for run in stack.contents.values()
                 for m in run.metadata.values())


# Header keywords a downscaled master keeps verbatim: identity + what WBPP
# matches on. Geometry keys are regenerated by the writer; binning and pixel
# size scale with the factor below (same rule as `downscale_fits`).
_MASTER_KEEP = (
    "IMAGETYP", "STBLHASH", "FILTER", "EXPOSURE", "EXPTIME", "INSTRUME",
    "TELESCOP", "GAIN", "OFFSET", "EGAIN", "CCD-TEMP", "SET-TEMP", "FOCALLEN",
)


@node(plate_solve="none")
def downscale_master(factor: int, master: ImageMetadata) -> WithSidecar[XisfFilePath]:
    """A calibration master at 1/`factor` resolution: the block mean of the
    full-resolution master, header identity kept (`STBLHASH` is how
    `stack_files` maps a master to the cal stack a run depends on).

    Calibration is linear, so the mean of a dark over a block IS the dark of
    the block-meaned light; a flat's block mean is the flat at that scale to
    within the smoothness of the pattern. Building the masters once at full
    resolution and downscaling them at the boundary is what makes the
    downscaled pipeline "the full pipeline plus downscale calls" rather than
    a second calibration stack of its own."""
    if factor < 1 or (factor & (factor - 1)) != 0:
        raise ValueError(f"factor must be a power of two >= 1, got {factor}")
    img = np.asarray(read_image(Path(master.filename)), dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"expected a mono master, got shape {img.shape}")
    h, w = img.shape
    img = img[: h - h % factor, : w - w % factor]
    h, w = img.shape
    out = img.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))

    src = read_header(Path(master.filename))
    header = {k: src[k] for k in _MASTER_KEEP if k in src}
    for k in ("XBINNING", "YBINNING"):
        if k in src:
            header[k] = int(round(float(src[k]) * factor))
    for k in ("XPIXSZ", "YPIXSZ"):
        if k in src:
            header[k] = float(src[k]) * factor
    output = downscale_master.filename_fn(factor, master)
    # `XISF.write` wants (H, W, C); a master is mono.
    write_xisf_image(Path(output), out.astype(np.float32, copy=False)[:, :, None], header)
    return WithSidecar(XisfFilePath(output))
