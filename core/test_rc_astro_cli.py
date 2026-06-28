"""rc-astro CLI wrapper: builds the right bxt/sxt commands and relocates outputs.

Ray-free; mocks `subprocess.run` (writing the deterministically-named outputs
rc-astro 0.9.3 produces under `-o <dir>`) so it needs no rc-astro install. The
real CLI's flags + output names were validated against a live run.
"""
import subprocess
from pathlib import Path

import pytest

import rc_astro_cli


def _fake_run_writing_outputs():
    """Fake subprocess.run that emulates rc-astro: writes `<stem>-bxt.fits` or
    `<stem>-sxt.fits` (+ `-sxt-stars.fits` when `--stars`) into the `-o` dir."""
    def fake_run(cmd, **kw):
        fake_run.cmd = cmd
        sub = cmd[cmd.index("--no-banner") + 1]
        inp = Path(cmd[cmd.index("--no-banner") + 2])
        odir = Path(cmd[cmd.index("-o") + 1])
        if sub == "bxt":
            (odir / f"{inp.stem}-bxt.fits").write_bytes(b"FITS")
        elif sub == "sxt":
            (odir / f"{inp.stem}-sxt.fits").write_bytes(b"FITS")
            if "--stars" in cmd:
                (odir / f"{inp.stem}-sxt-stars.fits").write_bytes(b"FITS")

        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()
    return fake_run


@pytest.fixture
def fake(monkeypatch):
    f = _fake_run_writing_outputs()
    monkeypatch.setattr(rc_astro_cli, "rc_astro_base_command", lambda: ["rc-astro"])
    monkeypatch.setattr(subprocess, "run", f)
    return f


def test_bxt_command_and_relocates_output(fake, tmp_path):
    out = tmp_path / "out" / "bxt.fits"
    rc_astro_cli.run_bxt(tmp_path / "chan.fits", out, sharpen_stars=0.1)
    cmd = fake.cmd
    assert cmd[cmd.index("--no-banner") + 1] == "bxt"
    assert "--depth" in cmd and cmd[cmd.index("--depth") + 1] == "32F"
    assert "--overwrite" in cmd
    assert cmd[cmd.index("--sharpen-stars") + 1] == "0.1"
    assert out.exists()  # moved from the temp -o dir to the requested path


def test_bxt_correct_only_drops_sharpen(fake, tmp_path):
    rc_astro_cli.run_bxt(tmp_path / "chan.fits", tmp_path / "o.fits", correct_only=True)
    assert "--correct-only" in fake.cmd
    assert "--sharpen-stars" not in fake.cmd


def test_sxt_starless_only_no_stars_flag(fake, tmp_path):
    out = tmp_path / "starless.fits"
    rc_astro_cli.run_sxt(tmp_path / "chan.fits", starless_out=out)
    assert "--stars" not in fake.cmd
    assert out.exists()


def test_sxt_stars_unscreen_writes_both(fake, tmp_path):
    sl, st = tmp_path / "sl.fits", tmp_path / "st.fits"
    rc_astro_cli.run_sxt(tmp_path / "chan.fits", starless_out=sl, stars_out=st, unscreen=True)
    assert "--stars" in fake.cmd and "--unscreen" in fake.cmd
    assert sl.exists() and st.exists()


def test_sxt_stars_only(fake, tmp_path):
    st = tmp_path / "st.fits"
    rc_astro_cli.run_sxt(tmp_path / "chan.fits", stars_out=st, unscreen=True)
    assert "--stars" in fake.cmd
    assert st.exists()


def test_sxt_requires_an_output(fake, tmp_path):
    with pytest.raises(ValueError):
        rc_astro_cli.run_sxt(tmp_path / "chan.fits")


def test_env_override_splits(monkeypatch):
    monkeypatch.setenv("ASTROFLOW_RC_ASTRO_BIN", "/opt/rc-astro/rc-astro --flag")
    assert rc_astro_cli.rc_astro_base_command() == ["/opt/rc-astro/rc-astro", "--flag"]


def test_missing_binary_raises(monkeypatch):
    monkeypatch.delenv("ASTROFLOW_RC_ASTRO_BIN", raising=False)
    monkeypatch.setattr(rc_astro_cli.shutil, "which", lambda _: None)
    with pytest.raises(rc_astro_cli.RcAstroNotFoundError):
        rc_astro_cli.rc_astro_base_command()


def test_nonzero_exit_raises(monkeypatch, tmp_path):
    def boom(cmd, **kw):
        class R:
            returncode = 1
            stdout = "out"
            stderr = "err"
        return R()
    monkeypatch.setattr(rc_astro_cli, "rc_astro_base_command", lambda: ["rc-astro"])
    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(RuntimeError, match="exited 1"):
        rc_astro_cli.run_bxt(tmp_path / "chan.fits", tmp_path / "o.fits")
