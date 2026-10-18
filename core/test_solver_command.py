"""The solve-field command and environment, as `_solve_field` runs it on each
platform: its temp files in our temp dir, its Python steps on this Python,
and on Windows, where it is a Cygwin program, forward-slash paths."""
import subprocess
import sys
import tempfile
from pathlib import Path

import astropy.io.fits as pyfits
import numpy as np

import plate_solving


def _run_with_fake_solver(monkeypatch, platform: str, arguments: list[str]):
    frame = Path(tempfile.mkdtemp()) / "frame.fits"
    pyfits.PrimaryHDU(np.zeros((4, 4), np.float32)).writeto(frame)
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"], seen["env"] = command, kwargs["env"]
        return subprocess.CompletedProcess(command, 1, b"", b"")   # not solved
    monkeypatch.setattr(plate_solving.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "platform", platform)
    assert plate_solving._solve_field(str(frame), arguments) is None
    return seen["command"], seen["env"]


def test_the_solver_gets_our_temp_dir_and_the_python_for_its_steps(monkeypatch):
    command, env = _run_with_fake_solver(monkeypatch, "darwin", ["/tools/bin/solve-field", "--fits-image"])
    assert command[:2] == ["/tools/bin/solve-field", "--fits-image"]
    temp_dir = command[command.index("--temp-dir") + 1]
    assert command[-1].startswith(temp_dir) and command[-1].endswith("frame.fits")
    assert env["ASTROFLOW_PYTHON"] == sys.executable


def test_on_windows_every_path_is_forward_slashed(monkeypatch):
    command, _ = _run_with_fake_solver(
        monkeypatch, "win32", [r"C:\tools\bin\solve-field.exe", "--config", r"C:\Users\w\astrometry.cfg"])
    assert command[0] == "C:/tools/bin/solve-field.exe"
    assert command[2] == "C:/Users/w/astrometry.cfg"
    assert not any("\\" in a for a in command)
