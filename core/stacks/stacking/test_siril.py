"""The Siril stacker's script and its master's integration time, without
running Siril. Runs as a script too."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits

from file_io.master_integration import read_master_integration_seconds
from stacks.stacking.siril import STACK_OPTIONS, _calibration_script, _gather, _stack_script
from stacks.types import StackIdDependency


@dataclass
class _Meta:
    filename: Path


@dataclass
class _Run:
    metadata: dict
    dependencies: dict = field(default_factory=dict)


@dataclass
class _Stack:
    contents: dict


def _fits(path: Path) -> Path:
    fits.writeto(path, np.zeros((4, 4), np.float32), overwrite=True)
    return path


def test_runs_calibrate_with_their_own_masters_then_stack_as_one_sequence():
    tmp = Path(tempfile.mkdtemp(prefix="sirtest"))
    (tmp / "proc").mkdir()
    lights = tmp / "lights"; lights.mkdir()
    library = tmp / "NGC 7000 masters"; library.mkdir()     # library paths have spaces
    dark, flat = _fits(library / "dark.fits"), _fits(library / "flat.fits")
    run0 = _Run({str(k): _Meta(_fits(lights / f"a{k}.fit")) for k in range(2)},
                {"dark": StackIdDependency("d1"), "flat": StackIdDependency("f1")})
    run1 = _Run({str(k): _Meta(_fits(lights / f"b{k}.fit")) for k in range(2)})       # no calibration
    stack = _Stack({"r0": run0, "r1": run1})
    script = _calibration_script(stack, {"d1": dark, "f1": flat}, tmp)
    assert f"calibrate r0 -dark={tmp}/r0_dark.fit -flat={tmp}/r0_flat.fit -prefix=pp_" in script
    assert "r1" not in script                                   # nothing to calibrate
    assert (tmp / "r0_dark.fit").resolve() == dark.resolve()
    for k in (1, 2):                                            # what Siril's calibrate writes
        _fits(tmp / "proc" / f"pp_r0_{k:05d}.fit")
    all_dir = _gather(stack, tmp)
    targets = sorted(os.readlink(f) for f in all_dir.iterdir())
    assert targets == sorted([str(tmp / "proc/pp_r0_00001.fit"), str(tmp / "proc/pp_r0_00002.fit"),
                              str(lights / "b0.fit"), str(lights / "b1.fit")])
    script = _stack_script(all_dir, tmp, 7.25)
    assert "setcpu 8\nset core.mem_mode=1\nset core.mem_amount=7.2\n" in script and "merge" not in script
    assert f"stack r_light {STACK_OPTIONS} -out={tmp}/master" in script


def test_uncalibrated_stack_needs_no_calibration_phase():
    tmp = Path(tempfile.mkdtemp(prefix="sirtest"))
    (tmp / "proc").mkdir()
    run = _Run({"0": _Meta(_fits(tmp / "x.fit"))})
    assert _calibration_script(_Stack({"r": run}), {}, tmp) is None
    assert len(list(_gather(_Stack({"r": run}), tmp).iterdir())) == 1


def test_siril_master_integration_is_frame_count_times_exposure():
    tmp = Path(tempfile.mkdtemp())
    p = tmp / "m.fits"
    hdr = fits.Header({"STACKCNT": 45, "EXPTIME": 300.0, "LIVETIME": 13500.0})
    fits.writeto(p, np.zeros((2, 2), np.float32), hdr)
    assert read_master_integration_seconds(p, dict(hdr)) == 13500.0
    assert read_master_integration_seconds(p, {"LIVETIME": 600.0}) == 600.0
    assert read_master_integration_seconds(p, {"EXPTIME": 300.0}) is None      # a raw sub


def test_scratch_is_local_only_when_the_output_is_on_a_network_volume(monkeypatch):
    """A network output volume would carry every intermediate copy twice;
    a local one keeps the scratch beside the output, and so does a node
    whose own disk is too small for the stack."""
    from collections import namedtuple

    import psutil
    from stacks.stacking import siril
    Part = namedtuple("Part", "mountpoint fstype")
    Usage = namedtuple("Usage", "free")
    dest = Path("/Volumes/AstroProcessed/Processed/integrate_with_siril/m.fits")
    local = Path(tempfile.gettempdir())

    def where(fstype, free):
        monkeypatch.setattr(psutil, "disk_partitions", lambda all: [
            Part("/", "apfs"), Part("/Volumes/AstroProcessed", fstype)])
        monkeypatch.setattr(siril.shutil, "disk_usage", lambda p: Usage(free))
        return siril._scratch_parent(dest, 100 * 2**30)

    assert where("apfs", 10**15) == dest.parent
    assert where("smbfs", 10**15) == local
    assert where("smbfs", 110 * 2**30) == dest.parent       # no room for the stack + headroom


def test_calibration_frames_are_integrated_unregistered_by_their_kind():
    """Bias and darks integrate unnormalised, flats scaled to a common level;
    neither is registered (no stars) nor wFWHM-weighted."""
    tmp = Path(tempfile.mkdtemp(prefix="sirtest"))
    dark = _stack_script(tmp / "all", tmp, 4.0, "dark")
    flat = _stack_script(tmp / "all", tmp, 4.0, "flat")
    assert f"stack cal rej w 3 3 -nonorm -out={tmp}/master" in dark
    assert f"stack cal rej w 3 3 -norm=mul -out={tmp}/master" in flat
    for script in (dark, flat):
        assert "register" not in script and "wfwhm" not in script


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)


def test_integrate_with_siril_is_an_artifact_producer():
    """The body names its own output via `filename_fn`; postponed annotations
    silently drop it (boot141: every integration failed on it)."""
    from stacks.stacking.siril import integrate_with_siril
    assert callable(getattr(integrate_with_siril, "filename_fn", None))


def test_a_single_frame_run_is_calibrated_alone():
    """Siril won't load a one-image sequence; the lone frame goes through
    calibrate_single under the name the sequence path would give it."""
    tmp = Path(tempfile.mkdtemp(prefix="sirtest"))
    (tmp / "proc").mkdir()
    dark = _fits(tmp / "dark.fits")
    run = _Run({"0": _Meta(_fits(tmp / "only.fit"))}, {"dark": StackIdDependency("d1")})
    script = _calibration_script(_Stack({"r": run}), {"d1": dark}, tmp)
    assert f"calibrate_single r0_00001.fit -dark={tmp}/r0_dark.fit -prefix=pp_" in script
    assert "link" not in script and (tmp / "proc" / "r0_00001.fit").resolve() == (tmp / "only.fit").resolve()
