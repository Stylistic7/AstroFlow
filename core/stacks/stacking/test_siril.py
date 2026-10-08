"""The Siril stacker's script and its master's integration time, without
running Siril. Runs as a script too."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits

from file_io.master_integration import read_master_integration_seconds
from stacks.stacking.siril import STACK_OPTIONS, _script
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


def test_each_run_calibrates_with_its_own_masters_then_all_are_merged():
    tmp = Path(tempfile.mkdtemp(prefix="sirtest"))
    (tmp / "proc").mkdir()
    lights = tmp / "lights"; lights.mkdir()
    dark, flat = _fits(tmp / "dark.fits"), _fits(tmp / "flat.fits")
    run0 = _Run({str(k): _Meta(_fits(lights / f"a{k}.fit")) for k in range(2)},
                {"dark": StackIdDependency("d1"), "flat": StackIdDependency("f1")})
    run1 = _Run({str(k): _Meta(_fits(lights / f"b{k}.fit")) for k in range(2)})       # no calibration
    script = _script(_Stack({"r0": run0, "r1": run1}), {"d1": dark, "f1": flat}, tmp)
    assert f"calibrate r0 -dark={dark} -flat={flat} -prefix=pp_" in script
    assert "merge pp_r0 r1 all" in script and "register all" in script
    assert f"stack r_all {STACK_OPTIONS} -out={tmp}/master" in script
    assert len(list((tmp / "run_0").iterdir())) == 2 and len(list((tmp / "run_1").iterdir())) == 2


def test_a_single_run_is_registered_without_a_merge():
    tmp = Path(tempfile.mkdtemp(prefix="sirtest"))
    (tmp / "proc").mkdir()
    run = _Run({"0": _Meta(_fits(tmp / "x.fit"))})
    script = _script(_Stack({"r": run}), {}, tmp)
    assert "merge" not in script and "register r0" in script and "stack r_r0 " in script


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
