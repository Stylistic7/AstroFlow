import tempfile
import os
from pathlib import Path
import errno
import astropy.io.fits as pyfits  # type: ignore

import file_io
from file_io.types import XisfFilePath, WithSidecar
from file_io.generic import update_header, copy_image
from file_io.xisf_file_io import read_xisf_total_exposure_time
from stacks.run import Run
from stacks.types import StackIdDependency, FilenameDependency
from stacks.composite_stack import BaseStack, Stack, CalibrationStack, CalibrationStacksDict
from pix.tasks import run_wbpp
from metadata.classes import MetadataDict
from decorators import node
from decisions.settings import DevelopmentSettings, ProcessingConfig
from decisions.resolve import resolve_calibration, resolve_decisions
from decisions.compute_graph import ALL_EXCLUSION_TYPES
from previews.types import PIXINSIGHT, MASTER
from products import makes_image
from products.identity import (
    StackId, CalibrationStackId, identity_for_stack, PresentationType,
)
import structlog

THUMBNAIL = PresentationType.THUMBNAIL
TILE_PYRAMID = PresentationType.TILE_PYRAMID

log = structlog.get_logger()


def _symlink_frames(filenames: list[str], dest_dir: str) -> None:
    for filename in filenames:
        assert filename, f"bad filename {filename}"
        if not os.path.exists(filename):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), filename
            )

        dest_filename = os.path.join(dest_dir, os.path.basename(filename))
        print(f"symlinking {filename} -> {dest_filename}")
        os.symlink(filename, dest_filename)


def _copy_frames_with_plate_solution(
    metadata: MetadataDict, dest_dir: str,
) -> None:
    """Copy input frames into the WBPP working dir at full resolution.
    Downscaling lives in a separate op (`downscale_master`) that runs
    after `stack_files`; the legacy in-frame downscale is gone."""
    for frame in metadata.values():
        filename = frame.filename
        assert filename
        if not os.path.exists(filename):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), filename
            )

        dest_filename = os.path.join(dest_dir, os.path.basename(filename))
        os.makedirs(dest_dir, exist_ok=True)
        print(f"copying {filename} -> {dest_filename}")
        copy_image(filename, dest_filename)

        with pyfits.open(str(dest_filename), mode="update", mmap=False) as fits:
            if "invalid" in fits[0].header.get("BAYERPAT", "").lower():
                del fits[0].header["BAYERPAT"]

            if frame.plate_solution:
                fits[0].header["RA"] = frame.plate_solution.header["CRVAL1"]
                fits[0].header["DEC"] = frame.plate_solution.header["CRVAL2"]
                for k, v in frame.plate_solution.header.items():
                    fits[0].header[k] = v


def _setup_run(
    run: Run,
    calibration_stacks: CalibrationStacksDict,
    dest_dir: str,
) -> list[str]:
    filenames = [i.filename for i in run.metadata.values()]
    assert len(run.frames) == len(filenames)

    _copy_frames_with_plate_solution(run.metadata, dest_dir)

    def get_dependency_filename(d):
        if isinstance(d, StackIdDependency):
            from products.registry import get_filename_fn, MediaType
            cal_stack = calibration_stacks[d.stack_id]
            filename_fn = get_filename_fn(
                MediaType.IMAGE, identity_for_stack(cal_stack),
                MASTER,
            )
            return filename_fn(cal_stack)
        elif isinstance(d, FilenameDependency):
            return d.filename

    dependency_filenames = [
        get_dependency_filename(d) for d in run.dependencies.values()
    ]
    _symlink_frames(dependency_filenames, dest_dir)

    return filenames


def _setup_stack(
    stack: BaseStack,
    calibration_stacks: CalibrationStacksDict,
    dest_dir: str,
) -> list[str]:
    filenames = []
    for i, run in enumerate(stack.contents.values()):
        run_dir = os.path.join(dest_dir, f"RUN_{i}")
        os.makedirs(run_dir, exist_ok=True)
        filenames.extend(_setup_run(run, calibration_stacks, run_dir))

    return filenames


def _mock_stacking(
    src_filename: str, dest_filename: str, metadata: dict[str, str]
) -> WithSidecar[XisfFilePath]:
    assert src_filename is not None

    print(
        f"### Mocked stacking, converted filename {src_filename} to "
        f"filename {dest_filename} with metadata {metadata}"
    )
    file_io.conversion.convert_fits_to_xisf(
        src_filename, dest_filename, add_noise=True, metadata=metadata
    )
    return WithSidecar(XisfFilePath(dest_filename))


def _copy_wbpp_result(
    stacking_dir: str, dest_filename: str, metadata: dict[str, str]
):
    # The 'master' dir indicates success.
    master_dir = os.path.join(stacking_dir, "master")
    if not os.path.exists(master_dir):
        raise Exception("WBPP did not produce an output master dir")

    print("WBPP master dir contains:")
    for f in os.listdir(master_dir):
        print(f"  {f}")

    # Copy the last file written, and update its metadata.
    wbpp_masters = list(
        str(f.absolute()) for f in Path(master_dir).rglob("master*.xisf")
    )
    wbpp_masters.sort(key=lambda x: os.path.getmtime(x))
    filename = wbpp_masters[-1]
    print(f"last modified WBPP master {filename} -> {dest_filename}")
    copy_image(filename, dest_filename)
    update_header(dest_filename, metadata)

    return WithSidecar(XisfFilePath(dest_filename))


def narrow_stack_config(stack: BaseStack, config: ProcessingConfig) -> ProcessingConfig:
    """Narrow `config` for cache-key hashing: keep only this stack's
    decision overrides; drop `calibration_stacks` from the key entirely.

    Why no cal_stacks filtering:
    - `resolve_decision` caches the default in the context actor on the
      first ask. Once it picks a bias / dark / flat, the choice is sticky
      — new candidates don't re-bid. So the resolved deps for this stack
      are fully determined by `(stack, this-stack's-decisions, context)`.
    - The candidate set isn't an input to the cached result; only the
      sticky pick is.
    - Therefore `cal_stacks` doesn't need to be in the cache key at all.
      Adding a bias for any camera no longer invalidates this stack's
      cached master.

    The function still receives the full `ProcessingConfig` at runtime —
    `resolve_calibration` needs the full set to look up the picked dep
    by id. This narrowing affects only the cache key.
    """
    from dataclasses import replace
    from decisions.decisions import decisions_for_stack

    return replace(
        config,
        # Constant marker — drops cal_stacks from the hash without
        # changing the field's presence on the dataclass.
        calibration_stacks=None,
        decisions=decisions_for_stack(config.decisions, stack.stable_hash),
    )


def _stack_files_narrow(args, kwargs):
    """`stack_files` cache-key hook: drop the global cal_stacks and
    other-stack decisions so a new bias for an unrelated camera doesn't
    invalidate this stack's cached master."""
    stack, config, *rest = args
    return (stack, narrow_stack_config(stack, config), *rest), kwargs


@node(resources={"PixInsight": 1}, scheduling_strategy="SPREAD", auto_memory=True,
      cache_key_narrow=_stack_files_narrow)
@makes_image(PIXINSIGHT, [MASTER],
             identity=CalibrationStackId, presentations=[THUMBNAIL, TILE_PYRAMID],
             file_type=XisfFilePath, display="Master")
@makes_image(PIXINSIGHT, [MASTER],
             identity=StackId, presentations=[THUMBNAIL, TILE_PYRAMID],
             file_type=XisfFilePath, display="Master")
def stack_files(
    stack: BaseStack,
    config: ProcessingConfig,
) -> WithSidecar[XisfFilePath]:
    # Resolve calibration (lazy, cached via decision framework).
    stack = resolve_calibration(stack, config.get_calibration_stacks(), config.decisions)
    # Apply exclusion decisions to filter frames before stacking (light stacks only).
    if config.decisions and isinstance(stack, Stack):
        filtered = resolve_decisions(stack, ALL_EXCLUSION_TYPES, config.decisions, config.min_frames_in_stack)
        if filtered is None:
            raise ValueError(f"Stack {stack} has no frames after applying decisions")
        stack = filtered

    development_settings = config.development_settings

    # Determine the output file, and the metadata it should have.
    from products.registry import get_filename_fn, MediaType
    filename_fn = get_filename_fn(MediaType.IMAGE, identity_for_stack(stack), MASTER) or stack_files.filename_fn
    dest_filename = filename_fn(stack)
    dest_filename_exists = os.path.exists(dest_filename)
    os.makedirs(os.path.dirname(dest_filename), exist_ok=True)
    log.info("Entering stack_files", stack=str(stack), dest_filename=dest_filename, dest_filename_exists=dest_filename_exists)

    metadata = {
        "IMAGETYP": f"Master {stack.image_type}",
        "STBLHASH": stack.stable_hash,
    }

    # Maybe skip stacking if the output file already exists.
    # TODO: This is kind of dangerous, because the output filename only depends
    # on the stack's own content hash -- the hash used by AstroFlowNode is more
    # complete and includes this function's code, for example.
    if development_settings.dangerously_accept_existing_files and dest_filename_exists:
        log.info("Dangerously skipping stacking because file already exists", dest_filename=dest_filename)
        update_header(dest_filename, metadata)
        return WithSidecar(dest_filename)

    calibration_stacks = config.get_calibration_stacks()
    min_frames_in_stack = config.min_frames_in_stack
    log.info("Calibration stacks", keys=list(calibration_stacks.keys()))

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        log.debug("Using temporary directory", tmp_dir=tmp_dir)

        # `_setup_stack` defaults to full-resolution. The downscaled
        # output is produced by a separate `downscale_master` op (see
        # `stacks/downscale_master.py`), not by re-stacking at low res.
        filenames = _setup_stack(stack, calibration_stacks, tmp_dir)

        if len(filenames) < min_frames_in_stack:
            log.info("Not stacking, too few files", stack=str(stack), num_files=len(filenames))
            raise ValueError(
                f"Stack {stack} contains only {len(filenames)} files"
            )

        if development_settings.mock_pixinsight:
            return _mock_stacking(filenames[0], dest_filename, metadata)

        # WBPP groups by EXPOSURE and emits one master per group; if a
        # stale stack ever sneaks through with mixed exposures, fail
        # loud here rather than letting WBPP produce duplicate masters
        # downstream. Stack identification should make this unreachable.
        # Tolerance matches ImagingAttributes — header float jitter is
        # not a real mismatch.
        from attributes.imaging_attributes import EXPOSURE_TOLERANCE_SECONDS
        exposures = [
            float(m.header["EXPOSURE"])
            for m in stack.all_metadata().values()
            if "EXPOSURE" in m.header
        ]
        if exposures:
            spread = max(exposures) - min(exposures)
            assert spread < EXPOSURE_TOLERANCE_SECONDS, (
                f"Stack {stack.stable_hash} has mixed exposures "
                f"(spread={spread:.4f}s, distinct={sorted(set(exposures))}); "
                "WBPP would produce one master per exposure."
            )

        log.info("Running WBPP", stack=str(stack))
        run_wbpp(Path(tmp_dir), Path(tmp_dir))
        result = _copy_wbpp_result(tmp_dir, dest_filename, metadata)

    _check_integration_time(stack, dest_filename)
    return result


def _check_integration_time(stack: BaseStack, master_path) -> None:
    """Compare WBPP's recorded integration time against the input frames'
    sum of EXPOSURE; flag a concern if WBPP integrated materially less.

    A common WBPP failure mode is silent rejection of frames whose
    StarAlignment couldn't register them — the master gets produced
    but contains a fraction of the expected signal. The XISF stores
    `PCL:TotalExposureTime` (sum of EXPTIME for frames with weight > 0);
    we compare to `sum(EXPOSURE × frames)` from AstroFlow's view.
    Calibration stacks skipped — TotalExposureTime semantics differ
    for non-light masters.
    """
    if not isinstance(stack, Stack):
        return

    actual = read_xisf_total_exposure_time(master_path)
    if actual is None:
        return

    metadata = stack.all_metadata()
    expected = sum(
        m.header.get("EXPOSURE", m.header.get("EXPTIME", 0) or 0) or 0
        for m in metadata.values()
    )
    if expected <= 0:
        return

    ratio = actual / expected
    if ratio >= 0.95:
        return

    from concerns import warn as _concern_warn

    n_frames = len(metadata)
    n_eq = actual / max(expected / max(n_frames, 1), 1)  # equivalent frames at avg exposure
    _concern_warn(
        title=(
            f"WBPP integrated {actual / 3600:.2f} h of "
            f"{expected / 3600:.2f} h ({ratio:.0%})"
        ),
        detail=(
            f"The master integrates ~{n_eq:.0f} frames-worth of signal but "
            f"{n_frames} frames were submitted to WBPP. The captured WBPP log "
            f"for this run lists each rejection — most often a StarAlignment "
            f"failure against a poorly-chosen reference. Re-stacking with the "
            f"reference manually pinned often resolves it."
        ),
        actual_seconds=actual,
        expected_seconds=expected,
        ratio=round(ratio, 4),
        frame_count=n_frames,
    )


