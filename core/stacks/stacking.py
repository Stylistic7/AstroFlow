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
from stacks.simple_stack import SimpleStack, SimpleStacksDict
from stacks.types import StackIdDependency, FilenameDependency
from stacks.composite_stack import Stack, CompositeStack
from pix.tasks import run_wbpp
from metadata.classes import MetadataDict
from frozendict import frozendict
from decorators import node, create_thumbnail
from customization.settings import DevelopmentSettings
from customization.resolve import resolve_decisions
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


def _setup_simple_stack(
    stack: SimpleStack,
    calibration_stacks: SimpleStacksDict,
    dest_dir: str,
) -> None:
    filenames = [i.filename for i in stack.metadata.values()]
    assert len(stack.frames) == len(filenames)

    _copy_frames_with_plate_solution(stack.metadata, dest_dir)

    def get_dependency_filename(d):
        if isinstance(d, StackIdDependency):
            return stack_files.filename_fn(calibration_stacks[d.stack_id])
        elif isinstance(d, FilenameDependency):
            return d.filename

    # Add any dependent stacks (e.g. calibration stacks).
    dependendency_filenames = [
        get_dependency_filename(d) for d in stack.dependencies.values()
    ]
    _symlink_frames(dependendency_filenames, dest_dir)

    return filenames


def _setup_composite_stack(
    stack: CompositeStack,
    calibration_stacks: SimpleStacksDict,
    dest_dir: str,
) -> tuple[str | None, int]:
    filenames = []
    for i, substack in enumerate(stack.contents.values()):
        run_dir = os.path.join(dest_dir, f"RUN_{i}")
        os.makedirs(run_dir, exist_ok=True)
        result = _setup_simple_stack(
            substack, calibration_stacks, run_dir
        )
        filenames.extend(result)

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
@makes_image(CompositeStack).as_master(stack_filename)
@node(resources={"PixInsight": 1}, scheduling_strategy="SPREAD")
@create_thumbnail
def stack_files(
    stack: Stack,
    calibration_stacks: SimpleStacksDict,
    development_settings: DevelopmentSettings,
    min_frames_in_stack: int,
    decisions: frozendict = frozendict(),
) -> WithSidecar[XisfFilePath]:
    # Apply exclusion decisions to filter frames before stacking.
    if isinstance(stack, CompositeStack) and decisions:
        filtered = resolve_decisions(stack, ALL_EXCLUSION_TYPES, decisions, min_frames_in_stack)
        if filtered is None:
            raise ValueError(f"Stack {stack} has no frames after applying decisions")
        stack = filtered

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

    log.info("Calibration stacks", keys=list(calibration_stacks.keys()))

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        log.debug("Using temporary directory", tmp_dir=tmp_dir)

        setup_fn = (
            _setup_simple_stack
            if isinstance(stack, SimpleStack)
            else _setup_composite_stack
        )
        filenames = setup_fn(stack, calibration_stacks, tmp_dir)

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
