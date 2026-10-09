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
from stacks.stacking.siril import STACK_OPTIONS, _calibration_script, _gather, _reference_frame, _stack_script
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
    assert "setcpu 12\nset core.mem_mode=1\nset core.mem_amount=7.2\n" in script and "merge" not in script
    assert f"stack r_light {STACK_OPTIONS} -out={tmp}/master" in script


def test_uncalibrated_stack_needs_no_calibration_phase():
    tmp = Path(tempfile.mkdtemp(prefix="sirtest"))
    (tmp / "proc").mkdir()
    run = _Run({"0": _Meta(_fits(tmp / "x.fit"))})
    assert _calibration_script(_Stack({"r": run}), {}, tmp) is None
    assert len(list(_gather(_Stack({"r": run}), tmp).iterdir())) == 1


def test_reference_frame_follows_links_back_to_the_raw_sub():
    """Sequence header says image index 1 is the reference: file number 2,
    a calibrated frame whose uncalibrated link points at the raw sub."""
    tmp = Path(tempfile.mkdtemp(prefix="sirtest"))
    for d in ("proc", "all", "reg", "raw"):
        (tmp / d).mkdir()
    raw = [_fits(tmp / "raw" / f"sub{k}.fit") for k in (1, 2)]
    for k, r in enumerate(raw, 1):
        os.symlink(r, tmp / "proc" / f"r0_{k:05d}.fit")              # Siril's link
        _fits(tmp / "proc" / f"pp_r0_{k:05d}.fit")                    # Siril's calibrate
        os.symlink(tmp / "proc" / f"pp_r0_{k:05d}.fit", tmp / "all" / f"{k:05d}.fit")
        os.symlink(tmp / "all" / f"{k:05d}.fit", tmp / "reg" / f"light_{k:05d}.fit")
    (tmp / "reg" / "light_.seq").write_text(
        "#Siril sequence file.\n"
        "#S 'sequence_name' start_index nb_images nb_selected fixed_len reference_image version\n"
        "S 'light_' 1 2 2 5 1 6 0 0 0\nL 1\nI 1 1\nI 2 1\n")
    assert _reference_frame(tmp) == Path(os.path.realpath(raw[1]))


def test_siril_master_integration_is_frame_count_times_exposure():
    tmp = Path(tempfile.mkdtemp())
    p = tmp / "m.fits"
    hdr = fits.Header({"STACKCNT": 45, "EXPTIME": 300.0, "LIVETIME": 13500.0})
    fits.writeto(p, np.zeros((2, 2), np.float32), hdr)
    assert read_master_integration_seconds(p, dict(hdr)) == 13500.0
    assert read_master_integration_seconds(p, {"LIVETIME": 600.0}) == 600.0
    assert read_master_integration_seconds(p, {"EXPTIME": 300.0}) is None      # a raw sub


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)


def test_integrate_with_siril_is_an_artifact_producer():
    """The body names its own output via `filename_fn`; postponed annotations
    silently drop it (boot141: every integration failed on it)."""
    from stacks.stacking.siril import integrate_with_siril
    assert callable(getattr(integrate_with_siril, "filename_fn", None))
