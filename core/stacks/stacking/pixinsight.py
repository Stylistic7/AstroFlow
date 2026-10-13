import tempfile
import os
from pathlib import Path
import errno
from dataclasses import dataclass
from typing import Any
from frozendict import frozendict
import astropy.io.fits as pyfits  # type: ignore

import file_io
from file_io.types import XisfFilePath, WithSidecar
from file_io.generic import update_header, copy_image
from file_io.xisf_file_io import read_xisf_total_exposure_time
from stacks.run import Run
from stacks.types import StackIdDependency, FilenameDependency
from stacks.composite_stack import BaseStack, Stack
from pix.tasks import run_wbpp, solver_geometry
from metadata.classes import ImageMetadata, MetadataDict
from stacks.dust_epochs import MoteModel
from stacks.star_flat_gain import GAIN_SECTION, FlatGain, with_scaled_flats
from decorators import node
from decisions.settings import ProcessingConfig, DevelopmentSettings
from decisions.resolve import resolve_calibration, resolve_decisions
from decisions.compute_graph import ALL_EXCLUSION_TYPES
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
    metadata: MetadataDict, dest_dir: str,
) -> None:
    """Stage input frames into the WBPP working dir.

    Reads the frames the stack names, as they are. For each frame: copy the file, drop a stale BAYERPAT, and write
    plate-solution header info — full WCS when the frame's
    `width_pixels` matches the plate-solution NAXIS1 (lossless
    re-use); otherwise just RA/DEC hints (the frame was
    downscaled after solving, so the WCS pixel grid no longer
    matches, but RA/DEC still anchor WBPP's re-solve)."""
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
                ps_header = frame.plate_solution.header
                fits[0].header["RA"] = ps_header["CRVAL1"]
                fits[0].header["DEC"] = ps_header["CRVAL2"]
                ps_naxis = ps_header.get("NAXIS1") or ps_header.get("IMAGEW")
                if ps_naxis == frame.width_pixels:
                    for k, v in ps_header.items():
                        fits[0].header[k] = v

def _setup_run(
    run: Run,
    cal_master_by_id: dict,
    dest_dir: str,
) -> list[str]:
    """Stage one run's lights + symlink its resolved cal masters.

    `cal_master_by_id` maps each `StackIdDependency.stack_id` to
    the on-disk path of that cal master (a value supplied by the
    pipeline as an upstream `stack_files` result, NOT recomputed
    from `filename_fn`). FilenameDependencies (external
    pre-stacked masters) still resolve inline."""
    filenames = [i.filename for i in run.metadata.values()]
    assert len(run.frames) == len(filenames)

    _copy_frames_with_plate_solution(run.metadata, dest_dir)

    def get_dependency_filename(d):
        if isinstance(d, StackIdDependency):
            return cal_master_by_id[str(d.stack_id)]
        elif isinstance(d, FilenameDependency):
            return d.filename

    dependency_filenames = [
        p for d in run.dependencies.values()
        if (p := get_dependency_filename(d)) is not None
    ]
    _symlink_frames(dependency_filenames, dest_dir)

    return filenames

def _setup_stack(
    stack: BaseStack,
    cal_master_by_id: dict,
    dest_dir: str,
) -> list[str]:
    filenames = []
    for i, run in enumerate(stack.contents.values()):
        run_dir = os.path.join(dest_dir, f"RUN_{i}")
        os.makedirs(run_dir, exist_ok=True)
        filenames.extend(_setup_run(run, cal_master_by_id, run_dir))

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

@dataclass(frozen=True)
class _StackKey:
    """Cache-key projection of `ProcessingConfig` — exactly the fields a
    stacked/merged master's *output* depends on.

    Field names, order, and values mirror the historical narrowed
    `ProcessingConfig` (calibration_stacks=None, development_settings,
    min_frames_in_stack, decisions). `make_hashable` hashes dataclasses
    structurally (by field name + value, ignoring the class), so this
    yields a byte-identical cache key to the old narrowing — no re-stack.
    And because it is its own type, new `ProcessingConfig` fields can never
    perturb stacking's key (protobuf-style: unknown fields are ignored).

    `calibration_stacks` is carried as `None` — sticky resolution means
    the candidate set isn't an output dependency; the `None` both
    preserves the historical key and documents the deliberate exclusion.
    """
    calibration_stacks: Any
    development_settings: DevelopmentSettings
    min_frames_in_stack: int
    decisions: frozendict


def narrow_stack_config(stack: BaseStack, config: ProcessingConfig) -> _StackKey:
    """Project `config` onto stacking's cache-dependency surface
    (`_StackKey`). Key-only: callers still receive the full
    `ProcessingConfig` at runtime (`resolve_calibration` needs
    `calibration_stacks`); only the hash surrogate is narrowed.
    Returning a dedicated projection type — rather than a narrowed
    `ProcessingConfig` — is what lets `ProcessingConfig` gain fields
    without re-keying every cached master.
    """
    if not isinstance(config, ProcessingConfig):
        raise TypeError(
            f"narrow_stack_config got config={type(config).__name__!r}, "
            f"expected ProcessingConfig — stack_files args are almost certainly "
            f"swapped. The signature is stack_files(stack, config, *cal_masters): "
            f"stack first, config second."
        )
    from decisions.decisions import decisions_for_stack

    return _StackKey(
        calibration_stacks=None,
        development_settings=config.development_settings,
        min_frames_in_stack=config.min_frames_in_stack,
        decisions=decisions_for_stack(config.decisions, stack.stable_hash),
    )

def _stack_files_narrow(args, kwargs):
    """`stack_files` cache-key hook: drop the global cal_stacks and
    other-stack decisions so a new bias for an unrelated camera doesn't
    invalidate this stack's cached master. cal_stack_ids and the
    `*cal_masters` data-deps that follow are passed through — their
    content_hashes flow into the call_hash, which is what makes
    different upstream cal masters produce different output files."""
    stack, config, *rest = args
    return (stack, narrow_stack_config(stack, config), *rest), kwargs

def _wbpp(stack, tmp_dir, dest_filename, metadata):
    """Run WBPP on the stack set up in `tmp_dir` and copy its master to `dest_filename`."""
    # WBPP groups by EXPOSURE and emits one master per group; if a
    # stale stack ever sneaks through with mixed exposures, fail
    # loud here rather than letting WBPP produce duplicate masters
    # downstream. Stack identification should make this unreachable.
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

    # Tell WBPP's image solver the camera's real geometry. Left to its
    # defaults it assumes a 5.4 um pixel and writes that back into the
    # master's XPIXSZ/FOCALLEN (see `solver_geometry`).
    geometry = solver_geometry(
        [m.header for m in stack.all_metadata().values()]
    )
    log.info("Running WBPP", stack=str(stack), **geometry)
    run_wbpp(Path(tmp_dir), Path(tmp_dir), **geometry)
    result = _copy_wbpp_result(tmp_dir, dest_filename, metadata)
    _check_integration_time(stack, dest_filename)
    return result


def _run_wbpp_stacking(
    stack: BaseStack, config: ProcessingConfig,
    cal_masters: tuple,
    dest_filename: Path,
) -> WithSidecar[XisfFilePath]:
    """Shared body for `stack_files`. Resolution-agnostic — reads
    `stack.contents` frame paths AS-IS. Cal
    master file paths come from `cal_masters`; each master's
    `STBLHASH` header (written by `_run_wbpp_stacking` on the
    upstream call) names the cal stack it was built from, so the
    body builds the `dep.stack_id → file` map internally."""
    # Enrich the stack's runs with their cal dependencies, so
    # `_setup_run`'s `run.dependencies.values()` walk can match
    # `dep.stack_id` to the cal master file from `cal_masters`.
    # Idempotent if already enriched.
    stack = resolve_calibration(
        stack, config.get_calibration_stacks(), config.decisions,
    )
    if config.decisions and isinstance(stack, Stack):
        filtered = resolve_decisions(stack, ALL_EXCLUSION_TYPES, config.decisions, config.min_frames_in_stack)
        if filtered is None:
            raise ValueError(f"Stack {stack} has no frames after applying decisions")
        stack = filtered

    development_settings = config.development_settings

    dest_filename_exists = os.path.exists(dest_filename)
    os.makedirs(os.path.dirname(dest_filename), exist_ok=True)
    log.info("Entering stack_files", stack=str(stack),
             dest_filename=dest_filename, dest_filename_exists=dest_filename_exists)

    metadata = {
        "IMAGETYP": f"Master {stack.image_type}",
        "STBLHASH": stack.stable_hash,
    }

    if development_settings.dangerously_accept_existing_files and dest_filename_exists:
        log.info("Dangerously skipping stacking because file already exists", dest_filename=dest_filename)
        update_header(dest_filename, metadata)
        return WithSidecar(dest_filename)

    min_frames_in_stack = config.min_frames_in_stack
    # Each cal master file has STBLHASH (= the cal stack's stable_hash)
    # in its FITS/XISF header, so read it back to build the
    # dependency-id → file-path map without the pipeline having to
    # thread parallel id/master lists through the call.
    from file_io.generic import read_header
    cal_master_by_id = {
        # STBLHASH is a hash identifier, but the XISF reader runs
        # ast.literal_eval on keyword values, so an all-digit stable_hash
        # round-trips as an int. Force str so the str-keyed lookup matches
        # (see get_dependency_filename, which also str()s the dep id).
        str(read_header(m.filename)["STBLHASH"]): m.filename
        for m in cal_masters if not isinstance(m, (FlatGain, MoteModel))
    }
    gains = [m for m in cal_masters if isinstance(m, FlatGain)]
    models = [m for m in cal_masters if isinstance(m, MoteModel)]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        log.debug("Using temporary directory", tmp_dir=tmp_dir)
        # A star / night-sky flat comes with this stack's gain: its scaled
        # copy lives in the scratch and goes with it.
        cal_master_by_id = with_scaled_flats(cal_master_by_id, gains, tmp_dir, models[:1] if len(models) == 1 else ())
        if len(models) == 1:
            # WBPP attaches one flat per filter group by keyword, so one
            # epoch's mote map is applied whole; a stack spanning epochs
            # is calibrated without motes (Siril calibrates run by run).
            for g in gains:
                cal_master_by_id[g.flat_id] = cal_master_by_id[f"{g.flat_id}:{models[0].content_hash}"]
        elif models:
            from concerns import warn
            warn("Dust motes not corrected: the stack spans several dust epochs",
                 "WBPP takes one flat per filter, so a stack whose runs saw different dust is "
                 "calibrated without the mote correction. The Siril pipeline calibrates run by run.",
                 stack=stack.stable_hash, epochs=[m.content_hash for m in models])

        filenames = _setup_stack(stack, cal_master_by_id, tmp_dir)

        if len(filenames) < min_frames_in_stack:
            log.info("Not stacking, too few files", stack=str(stack), num_files=len(filenames))
            raise ValueError(
                f"Stack {stack} contains only {len(filenames)} files"
            )

        if development_settings.mock_pixinsight:
            result = _mock_stacking(filenames[0], dest_filename, metadata)
        else:
            result = _wbpp(stack, tmp_dir, dest_filename, metadata)

    if gains:
        import json
        from file_io.sidecar import set_section
        set_section(Path(dest_filename), GAIN_SECTION, {g.flat_id: json.loads(g.fit_json) for g in gains})
    return result

@node(
    display="Master",
    final=True,
    cache_key_narrow=_stack_files_narrow,
    resources={"PixInsight": 1},
    scheduling_strategy="SPREAD",
    auto_memory=True,
)
def stack_files(
    stack: BaseStack,
    config: ProcessingConfig,
    *cal_masters: ImageMetadata | FlatGain | MoteModel,
) -> WithSidecar[XisfFilePath]:
    """Resolution-agnostic PixInsight WBPP master.

    Frame paths are read from `stack.contents` AS-IS.

    Signature follows the tracer's "literals first, deps trailing"
    convention: `stack` and `config` are literals, and `*cal_masters`
    are always Node handles from upstream producers.

    `*cal_masters` is the cal masters this stack's runs depend on
    (any order; one per (bias, dark, flat) the stack references).
    Each master file carries its source cal stack's `stable_hash`
    in its FITS/XISF `STBLHASH` header — the body reads that to
    map masters to dependencies internally.

    The masters' content_hashes flow into this op's call_hash via
    `_make_call_hash_filename_fn`, so calling this op with the
    same light stack but different cal masters produces distinct
    output files."""
    return _run_wbpp_stacking(
        stack, config, cal_masters,
        stack_files.filename_fn(stack, config, *cal_masters),
    )

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

