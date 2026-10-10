"""Siril as a stacker: `integrate_with_siril`, the peer of WBPP's `stack_files`.

Same contract — same signature (`stack, config, *cal_masters`), same
cache-key narrowing, same calibration resolution and decision filtering, a
master stamped with `STBLHASH` — but the integration runs in Siril's
headless `siril-cli`:

    per run      link the run's lights, calibrate with ITS dark/flat (runs
                 can differ, as in WBPP's grouping)
    all runs     merge into one sequence, global star registration
                 (homography), average with linear fit clipping, additive +
                 scale normalisation, noise weighting

Measured on 45 x 61 MP OIII subs (2026-10-08): 35 s, against ~29 min for the
app's WBPP on the same stack. What it does NOT do that WBPP does: local
normalisation and PSF-based weights (see the stacking A/B).

Siril writes no `NOISE00`, so the master gets one here — the robust noise
`metadata.visibility` uses, NOT PixInsight's MRS estimate. Within one
pipeline every master is measured the same way; do not compare the number
across stackers. Its frame count comes from Siril's own `STACKCNT`
(`file_io.master_integration`).
"""
# NO `from __future__ import annotations`: `@node` resolves the return
# annotation at decoration time to give the op its `filename_fn`; as a string
# it silently doesn't (see stacks/sky_flat.py).

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import structlog

from decisions.compute_graph import ALL_EXCLUSION_TYPES
from decisions.resolve import resolve_calibration, resolve_decisions
from decisions.settings import ProcessingConfig
from decorators import node
from file_io.types import FitsFilePath, WithSidecar
from metadata.classes import ImageMetadata
from stacks.composite_stack import BaseStack, Stack
from stacks.stacking.pixinsight import _stack_files_narrow
from stacks.star_flat_gain import GAIN_SECTION, FlatGain, with_scaled_flats
from stacks.types import FilenameDependency, StackIdDependency

log = structlog.get_logger()

SIRIL_APP = "/Applications/Siril.app/Contents/MacOS/siril-cli"
# Threads per Siril instance. Siril's file I/O is serial, so one instance
# leaves cores idle; but two instances at 24 threads each spin-wait in
# OpenMP. Measured on 45 x 61 MP subs with the app running (2026-10-08),
# seconds per stack: one at a time x 24 threads 38.5, two at once x 12
# threads 32.5, three x 8 38, two x 24 43.
SIRIL_THREADS = 8
# The CPUs its Ray task reserves: fewer than its threads, because an
# instance spends much of a stack in serial phases (sampled 1-4 cores,
# bursting to 14). Reserving all 8 left the Studio ~50% busy with two
# instances admitted (2026-10-09); at 6 it admits a third or fourth, whose
# bursts fill the others' serial gaps, and the 10-CPU MacBook Pro one.
SIRIL_CPUS = 6
# The integration recipe, chosen by measurement (2026-10-08, panel 6 OIII,
# 45 subs, uncalibrated; "revealed" = % of the field at >=3 sigma after
# GraXpert): winsorized 3/3 sigma clipping, additive + scaling normalisation,
# weights from star-weighted FWHM. It revealed 21.0% against WBPP's 19.9%, in
# 26 s against WBPP's minutes. Linear-fit clipping (WBPP's choice above 15
# frames) revealed 20.0%; noise weighting gave fatter stars; dropping the
# worst 20% by wFWHM (-filter-wfwhm=80%) lost depth; two-pass registration
# gave sharper stars (2.09 px vs 2.19) but revealed less (18.7%).
STACK_OPTIONS = "rej w 3 3 -norm=addscale -output_norm -weight=wfwhm"


def siril_cli() -> str:
    """`$ASTROFLOW_SIRIL_BIN`, else the official app bundle (Homebrew's 1.4.4
    links OpenCV 5 and crashes in every rejection stack, 2026-10-08)."""
    path = os.environ.get("ASTROFLOW_SIRIL_BIN") or SIRIL_APP
    if not os.path.exists(path):
        raise FileNotFoundError(f"siril-cli not found at {path}: install Siril on this node (siril.org)")
    return path


def _as_fits(src: Path, dst: Path) -> Path:
    """A calibration master as 32-bit FITS at `dst` (Siril does not read
    XISF). A FITS master is symlinked, because the script names it by `dst`:
    library paths contain spaces, which Siril splits arguments on."""
    from astropy.io import fits
    from file_io.generic import read_image
    if dst.exists():
        return dst
    if src.suffix.lower() in (".fit", ".fits"):
        os.symlink(src, dst)
    else:
        fits.writeto(dst, np.asarray(read_image(src), np.float32).squeeze(), overwrite=True)
    return dst


def _calibration_script(stack: BaseStack, cal_by_id: dict[str, Path], tmp: Path) -> str | None:
    """Phase 1: calibrate each run that has masters into `tmp/proc` as
    pp_r{i}_NNNNN.fit. None when no run has any. Paths are written bare:
    Siril splits arguments on spaces (masters are linked under `tmp`)."""
    assert " " not in str(tmp), tmp
    lines = ["requires 1.4.0", "setext fit", f"setcpu {SIRIL_THREADS}"]
    for i, run in enumerate(stack.contents.values()):
        cal = {}
        for kind, dep in run.dependencies.items():
            if isinstance(dep, StackIdDependency):
                src = cal_by_id.get(str(dep.stack_id))
            elif isinstance(dep, FilenameDependency):
                src = Path(dep.filename) if dep.filename else None
            else:
                src = None
            if src is not None and kind in ("bias", "dark", "flat"):
                cal[kind] = _as_fits(Path(src), tmp / f"r{i}_{kind}.fit")
        if not cal:
            continue
        args = " ".join(f"-{k}={v}" for k, v in sorted(cal.items()))
        if len(run.metadata) == 1:
            # Siril won't load a one-image sequence ("invalid input
            # sequence"); calibrate the lone frame directly, under the name
            # the sequence path would have given it.
            (m,) = run.metadata.values()
            os.symlink(m.filename, tmp / "proc" / f"r{i}_00001.fit")
            lines += [f"cd {tmp}/proc", f"calibrate_single r{i}_00001.fit {args} -prefix=pp_"]
            continue
        run_dir = tmp / f"run_{i}"
        run_dir.mkdir()
        for m in run.metadata.values():
            os.symlink(m.filename, run_dir / Path(m.filename).name)
        lines += [f"cd {run_dir}", f"link r{i} -out={tmp}/proc", f"cd {tmp}/proc",   # Siril names it r{i}_
                  f"calibrate r{i} {args} -prefix=pp_"]
    return "\n".join(lines) + "\n" if len(lines) > 3 else None


def _gather(stack: BaseStack, tmp: Path) -> Path:
    """Every frame to stack, calibrated where phase 1 calibrated it, as
    symlinks in ONE folder: one sequence without Siril's `merge`, which
    would copy every calibrated frame again."""
    all_dir = tmp / "all"
    all_dir.mkdir()
    n = 0
    for i, run in enumerate(stack.contents.values()):
        calibrated = sorted((tmp / "proc").glob(f"pp_r{i}_*.fit"))
        frames = calibrated or [Path(m.filename) for m in run.metadata.values()]
        assert len(frames) == len(run.metadata), (i, len(frames), len(run.metadata))
        for f in frames:
            n += 1
            os.symlink(f, all_dir / f"{n:05d}{f.suffix.lower()}")
    return all_dir


def _stack_script(all_dir: Path, tmp: Path, mem_gb: float, image_type: str = "light") -> str:
    """Phase 2: integrate the gathered frames within `mem_gb` (Siril's
    default, 0.9 of free memory, would let concurrent instances each claim
    the machine). Lights are registered first; calibration frames are
    integrated as they lie, flats scaled to a common level, bias and darks
    left unnormalised (Siril's own master recipes)."""
    assert " " not in str(tmp), tmp
    head = ["requires 1.4.0", "setext fit", f"setcpu {SIRIL_THREADS}", "set core.mem_mode=1",
            f"set core.mem_amount={mem_gb:.1f}", f"cd {all_dir}"]
    if image_type != "light":
        norm = "-norm=mul" if image_type == "flat" else "-nonorm"
        return "\n".join(head + [
            f"link cal -out={tmp}/reg", f"cd {tmp}/reg",
            f"stack cal rej w 3 3 {norm} -out={tmp}/master", "close"]) + "\n"
    return "\n".join(head + [
        f"link light -out={tmp}/reg", f"cd {tmp}/reg",
        # Register, then re-apply cropped to the area EVERY frame covers, so
        # no consumer can include ragged, partly covered edges. -2pass would
        # skip the first write but picks its own reference, which cost
        # depth (revealed 19.3% vs 20.6%, panel 6 OIII, 2026-10-08).
        # Siril's default homography, deliberately (2026-10-10, three stacks
        # spanning both meridian sides: similarity left edge stars 3-9% fatter
        # and photometric scatter 20-45% worse). A stack whose bright, starved
        # pre-dawn frames derail it shows as an SPFC refusal; quarantine that
        # run (Fish on the Platter B: dropping one spring night, scatter 0.29 -> 0.09).
        "register light", "seqapplyreg light -framing=min -interp=la",
        f"stack r_light {STACK_OPTIONS} -out={tmp}/master", "close"]) + "\n"


def _run_siril(tmp: Path, name: str, script: str) -> None:
    (tmp / name).write_text(script)
    r = subprocess.run([siril_cli(), "-d", str(tmp), "-s", str(tmp / name)],
                       capture_output=True, text=True, timeout=6 * 3600)
    if r.returncode != 0:
        raise RuntimeError(f"Siril {name} exited {r.returncode}.\n{r.stdout[-3000:]}\n{r.stderr[-1500:]}")


def _memory_budget_gb() -> float:
    """What this task may give Siril: its Ray memory reservation (fixed in
    profiling.memory_estimate._FIXED_MEMORY_BY_OP) less what the worker
    itself already holds."""
    import psutil
    import ray
    reserved = ray.get_runtime_context().get_assigned_resources().get("memory")
    assert reserved, "integrate_with_siril runs without a Ray memory reservation"
    return max(1.0, (reserved - psutil.Process().memory_info().rss) / 1024 ** 3)


def _robust_noise(path: Path) -> float:
    from file_io.generic import read_image
    from metadata.visibility import _noise
    a = np.asarray(read_image(path), np.float64).squeeze()
    return _noise(np.where(np.isfinite(a) & (a > 0), a, np.nan))



_NETWORK_FS = {"smbfs", "nfs", "afpfs", "webdav"}
_HEADROOM = 20 * 2**30


def _scratch_parent(dest: Path, need_bytes: int) -> Path:
    """Where a stack's scratch goes. Beside the output when that volume is
    local: the big processed volume, one rename to publish. On a network
    volume every calibrated and registered copy would cross the wire twice
    (~4-5x the raw data), so this node's own disk takes the scratch when it
    has room: one stack peaks at 2 x frames x 4 bytes x pixels, 170 GB for
    the largest."""
    import psutil
    mount = max((p for p in psutil.disk_partitions(all=True) if dest.is_relative_to(p.mountpoint)),
                key=lambda p: len(p.mountpoint))
    local = Path(tempfile.gettempdir())
    if mount.fstype in _NETWORK_FS and shutil.disk_usage(local).free > need_bytes + _HEADROOM:
        return local
    return dest.parent

@node(display="Master", final=True, cache_key_narrow=_stack_files_narrow,
      num_cpus=SIRIL_CPUS, resources={"Siril": 0.001}, scheduling_strategy="SPREAD",
      auto_memory=True)
def integrate_with_siril(
    stack: BaseStack,
    config: ProcessingConfig,
    *cal_masters: ImageMetadata | FlatGain,
) -> WithSidecar[FitsFilePath]:
    """Calibrate, register and integrate `stack` with Siril. The peer of
    `stack_files`: pipelines call it with the same arguments."""
    from file_io.generic import read_header, update_header
    stack = resolve_calibration(stack, config.get_calibration_stacks(), config.decisions)
    if config.decisions and isinstance(stack, Stack):
        stack = resolve_decisions(stack, ALL_EXCLUSION_TYPES, config.decisions, config.min_frames_in_stack)
        if stack is None:
            raise ValueError("stack has no frames after applying decisions")
    n = sum(len(r.frames) for r in stack.contents.values())
    if n < config.min_frames_in_stack:
        raise ValueError(f"Stack {stack} contains only {n} files")
    dest = Path(integrate_with_siril.filename_fn(stack, config, *cal_masters))
    dest.parent.mkdir(parents=True, exist_ok=True)
    cal_by_id = {str(read_header(m.filename)["STBLHASH"]): Path(m.filename)
                 for m in cal_masters if not isinstance(m, FlatGain)}
    gains = [m for m in cal_masters if isinstance(m, FlatGain)]
    need = sum(2 * 4 * int(m.header["NAXIS1"]) * int(m.header["NAXIS2"])      # calibrated + registered float32
               for m in stack.all_metadata().values())
    with tempfile.TemporaryDirectory(prefix=".siril-", dir=_scratch_parent(dest, need)) as d:
        tmp = Path(d)
        (tmp / "proc").mkdir()
        # A star / night-sky flat comes with this stack's gain: its scaled
        # copy lives in the scratch and goes with it.
        cal_by_id = with_scaled_flats(cal_by_id, gains, tmp)
        log.info("Running Siril", stack=str(stack), frames=n)
        script = _calibration_script(stack, cal_by_id, tmp)
        if script:
            _run_siril(tmp, "calibrate.ssf", script)
        _run_siril(tmp, "stack.ssf", _stack_script(_gather(stack, tmp), tmp, _memory_budget_gb(), stack.image_type))
        out = tmp / "master.fit"
        assert out.exists(), f"Siril wrote no master for {stack}"
        part = dest.with_name(f".siril-{dest.name}.part")
        shutil.move(out, part)                  # a copy when the scratch is on another volume
        os.replace(part, dest)
    if gains:
        import json
        from file_io.sidecar import set_section
        set_section(dest, GAIN_SECTION, {g.flat_id: json.loads(g.fit_json) for g in gains})
    update_header(dest, {"IMAGETYP": f"Master {stack.image_type}", "STBLHASH": stack.stable_hash,
                         "NOISE00": _robust_noise(dest), "NOISEEST": "robust high-pass (AstroFlow)"})
    return WithSidecar(FitsFilePath(dest))
