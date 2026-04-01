import ray
import tempfile
import os
from pathlib import Path
import errno
import os
import astropy.io.fits as pyfits  # type: ignore
from typing import Any

import file_io
from file_io.types import XisfFilePath, WithSidecar
from file_io.generic import update_header, copy_image
from stacks.artifacts import stack_filename
from stacks.run import Run, RunsDict
from stacks.types import StackIdDependency, FilenameDependency
from stacks.composite_stack import BaseStack, Stack, CalibrationStack, CalibrationStacksDict
from pix.tasks import run_wbpp
from metadata.classes import MetadataDict
from frozendict import frozendict
from decorators import node, create_thumbnail
from customization.settings import DevelopmentSettings, ProcessingConfig
from customization.resolve import resolve_calibration, resolve_decisions
from customization.compute_graph import ALL_EXCLUSION_TYPES
from products import makes_image
import structlog

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
    metadata: MetadataDict, dest_dir: str
) -> None:
    # TODO: Parallelize this.
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

        # TODO: Reinstate the DOWNSCALED_MASTER mode.
        # downscale_fits(filename, dest_filename, K=4)

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
            return stack_files.filename_fn(calibration_stacks[d.stack_id])
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


# TODO: add an `as_downscaled_master` quality level.
@makes_image(Stack).as_master(stack_filename)
@node(resources={"PixInsight": 1}, scheduling_strategy="SPREAD")
@create_thumbnail
def stack_files(
    stack: BaseStack,
    config: ProcessingConfig,
) -> WithSidecar[XisfFilePath]:
    # Resolve calibration (lazy, cached via decision framework).
    stack = resolve_calibration(stack, config.calibration_stacks, config.decisions)
    # Apply exclusion decisions to filter frames before stacking (light stacks only).
    if config.decisions and isinstance(stack, Stack):
        filtered = resolve_decisions(stack, ALL_EXCLUSION_TYPES, config.decisions, config.min_frames_in_stack)
        if filtered is None:
            raise ValueError(f"Stack {stack} has no frames after applying decisions")
        stack = filtered

    development_settings = config.development_settings

    # Determine the output file, and the metadata it should have.
    dest_filename = stack_files.filename_fn(stack)
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

    calibration_stacks = config.calibration_stacks
    min_frames_in_stack = config.min_frames_in_stack
    log.info("Calibration stacks", keys=list(calibration_stacks.keys()))

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        log.debug("Using temporary directory", tmp_dir=tmp_dir)

        filenames = _setup_stack(stack, calibration_stacks, tmp_dir)

        if len(filenames) < min_frames_in_stack:
            log.info("Not stacking, too few files", stack=str(stack), num_files=len(filenames))
            raise ValueError(
                f"Stack {stack} contains only {len(filenames)} files"
            )

        if development_settings.mock_pixinsight:
            return _mock_stacking(filenames[0], dest_filename, metadata)

        log.info("Running WBPP", stack=str(stack))
        run_wbpp(Path(tmp_dir), Path(tmp_dir))
        return _copy_wbpp_result(tmp_dir, dest_filename, metadata)


# Register stack_files as a producer for CalibrationStack too.
from products.registry import register_producer, MediaType
from previews.types import QualityLevel
register_producer(MediaType.IMAGE, CalibrationStack, QualityLevel.MASTER, stack_files)
