"""GraXpert CLI wrapper: builds the right (3.2) command and locates the output.

Ray-free; mocks `subprocess.run` so it needs no GraXpert install. The real CLI
is validated by a live run against an installed GraXpert.

The flags match GraXpert 3.2.0a2's actual `background-extraction -h`, which DOES
accept -correction/-smoothing/-ai_version. (An earlier version of this file
claimed 3.2 had dropped them; it hadn't. The mistake was invisible because the
one test that checked only ever called the wrapper without them, so their
absence proved nothing — `post_stack_calibrate` passes all three.)
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
    # Omitted when unset so GraXpert applies its own defaults — NOT because the
    # flags don't exist. See test_passes_tuning_flags_when_set.
    assert not ({"-correction", "-smoothing", "-ai_version"} & set(cmd))
    assert out.exists()


def test_passes_tuning_flags_when_set(monkeypatch, tmp_path):
    """The settings `post_stack_calibrate` pins must reach the CLI.

    Guards the mistake this file used to encode: asserting these flags are
    never passed. They are the entire reason GraXpert beats MGC here — model
    1.0.1 left a 0.27% background bowl where 1.0.0 left 0.61%.
    """
    fake = _fake_run_writing_output()
    monkeypatch.setattr(graxpert_cli, "graxpert_base_command", lambda: ["graxpert"])
    monkeypatch.setattr(subprocess, "run", fake)

    graxpert_cli.run_background_extraction(
        tmp_path / "in.fits", tmp_path / "out.fits",
        smoothing=0.25, correction="Subtraction", ai_version="1.0.1",
    )

    cmd = fake.cmd
    assert cmd[cmd.index("-smoothing") + 1] == "0.25"
    assert cmd[cmd.index("-correction") + 1] == "Subtraction"
    assert cmd[cmd.index("-ai_version") + 1] == "1.0.1"


def test_omits_gpu_when_unset(monkeypatch, tmp_path):
    fake = _fake_run_writing_output()
    monkeypatch.setattr(graxpert_cli, "graxpert_base_command", lambda: ["graxpert"])
    monkeypatch.setattr(subprocess, "run", fake)
    graxpert_cli.run_background_extraction(tmp_path / "in.fits", tmp_path / "out.fits")
    assert "-gpu" not in fake.cmd  # let GraXpert auto-detect the provider


def test_env_override_splits_into_argv(monkeypatch):
    monkeypatch.setenv("ASTROFLOW_GRAXPERT_BIN", "/opt/graxpert/bin/graxpert --extra")
    assert graxpert_cli.graxpert_base_command() == ["/opt/graxpert/bin/graxpert", "--extra"]


def test_finds_graxpert_in_uv_tool_bin_dir(monkeypatch, tmp_path):
    """`make install-graxpert` puts GraXpert in the uv tool bin dir, which is
    NOT on a Ray worker's PATH — the worker inherits the raylet's environment,
    not a login shell's. Resolution must not depend on PATH."""
    monkeypatch.delenv("ASTROFLOW_GRAXPERT_BIN", raising=False)
    monkeypatch.setattr(graxpert_cli.shutil, "which", lambda _: None)
    monkeypatch.setenv("UV_TOOL_BIN_DIR", str(tmp_path))
    (tmp_path / "graxpert").write_text("#!/bin/sh\n")

    assert graxpert_cli.graxpert_base_command() == [str(tmp_path / "graxpert")]


def test_raises_when_graxpert_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("ASTROFLOW_GRAXPERT_BIN", raising=False)
    monkeypatch.setattr(graxpert_cli.shutil, "which", lambda _: None)
    monkeypatch.setattr(graxpert_cli.importlib.util, "find_spec", lambda _: None)
    # Point the tool dir at an EMPTY dir: otherwise this passes or fails
    # depending on whether the machine running the tests happens to have
    # GraXpert installed, which is how this test broke.
    monkeypatch.setenv("UV_TOOL_BIN_DIR", str(tmp_path))
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
