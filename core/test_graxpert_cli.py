"""GraXpert CLI wrapper: builds the right (3.2) command and locates the output.

Ray-free; mocks `subprocess.run` so it needs no GraXpert install. The real CLI
is validated by a live run against an installed GraXpert (the flags here match
GraXpert 3.2's actual `-h`, which has no -correction/-smoothing/-ai_version).
"""
import subprocess
from pathlib import Path

import pytest

import graxpert_cli


def _fake_run_writing_output():
    def fake_run(cmd, **kw):
        fake_run.cmd = cmd
        out = cmd[cmd.index("-output") + 1]
        Path(out).with_suffix(".fits").write_bytes(b"FITS")  # GraXpert appends .fits

        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()
    return fake_run


def test_builds_background_extraction_command(monkeypatch, tmp_path):
    fake = _fake_run_writing_output()
    monkeypatch.setattr(graxpert_cli, "graxpert_base_command", lambda: ["graxpert"])
    monkeypatch.setattr(subprocess, "run", fake)

    out = tmp_path / "corrected.fits"
    graxpert_cli.run_background_extraction(tmp_path / "in.fits", out, gpu=False)

    cmd = fake.cmd
    assert cmd[0] == "graxpert"
    assert cmd[cmd.index("-cmd") + 1] == "background-extraction"
    # -output is the stem; GraXpert appends `.fits`.
    assert cmd[cmd.index("-output") + 1] == str(out.with_suffix(""))
    assert cmd[cmd.index("-gpu") + 1] == "false"
    # GraXpert 3.2 dropped these CLI flags — we must not pass them.
    assert not ({"-correction", "-smoothing", "-ai_version"} & set(cmd))
    assert out.exists()


def test_omits_gpu_when_unset(monkeypatch, tmp_path):
    fake = _fake_run_writing_output()
    monkeypatch.setattr(graxpert_cli, "graxpert_base_command", lambda: ["graxpert"])
    monkeypatch.setattr(subprocess, "run", fake)
    graxpert_cli.run_background_extraction(tmp_path / "in.fits", tmp_path / "out.fits")
    assert "-gpu" not in fake.cmd  # let GraXpert auto-detect the provider


def test_env_override_splits_into_argv(monkeypatch):
    monkeypatch.setenv("ASTROFLOW_GRAXPERT_BIN", "/opt/graxpert/bin/graxpert --extra")
    assert graxpert_cli.graxpert_base_command() == ["/opt/graxpert/bin/graxpert", "--extra"]


def test_raises_when_graxpert_missing(monkeypatch):
    monkeypatch.delenv("ASTROFLOW_GRAXPERT_BIN", raising=False)
    monkeypatch.setattr(graxpert_cli.shutil, "which", lambda _: None)
    monkeypatch.setattr(graxpert_cli.importlib.util, "find_spec", lambda _: None)
    with pytest.raises(graxpert_cli.GraXpertNotFoundError):
        graxpert_cli.graxpert_base_command()


def test_nonzero_exit_raises(monkeypatch, tmp_path):
    def fake_run(cmd, **kw):
        class R:
            returncode = 1
            stdout = "boom"
            stderr = "stderr tail"
        return R()
    monkeypatch.setattr(graxpert_cli, "graxpert_base_command", lambda: ["graxpert"])
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="exited 1"):
        graxpert_cli.run_background_extraction(tmp_path / "in.fits", tmp_path / "out.fits")
