"""rc-astro CLI wrapper: builds the right bxt/sxt commands and relocates outputs.

Ray-free; mocks `subprocess.run` (writing the deterministically-named outputs
rc-astro 0.9.3 produces under `-o <dir>`) so it needs no rc-astro install. The
real CLI's flags + output names were validated against a live run.
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import rc_astro_cli
import third_party_tools


_IMG_EXTS = {".xisf", ".fits", ".fit", ".tiff", ".tif"}


def _fake_run_writing_outputs():
    """Fake subprocess.run emulating rc-astro's `-o` semantics: an `-o` path with
    an image extension is the explicit output file (format = that ext, how BXT is
    driven to XISF); a bare `-o <dir>` writes `<stem>-<product>.<input-ext>` into
    the dir (how SXT runs, inheriting its input's format)."""
    def fake_run(cmd, **kw):
        fake_run.cmd = cmd
        sub = cmd[cmd.index("--no-banner") + 1]
        inp = Path(cmd[cmd.index("--no-banner") + 2])
        o = Path(cmd[cmd.index("-o") + 1])
        from file_io.generic import read_image, read_image_geometry
        g = read_image_geometry(inp)
        fake_run.fed = (g.width_pixels, g.height_pixels)
        fake_run.fed_pixels = np.atleast_3d(np.asarray(read_image(inp), np.float32))[..., 0]

        def write(target: Path):     # the "product" is the input, unchanged
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(inp, target)

        if o.suffix.lower() in _IMG_EXTS:        # explicit output file (bxt)
            write(o)
        else:                                    # directory (sxt): inherit input ext
            write(o / f"{inp.stem}-{sub}{inp.suffix}")
            if "--stars" in cmd:
                write(o / f"{inp.stem}-{sub}-stars{inp.suffix}")

        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()
    return fake_run


def _channels(d: Path, w: int = 512, h: int = 512) -> None:
    """`chan.fits` and `chan.xisf`, `w` x `h` (on the 512 ladder by default)."""
    from astropy.io import fits
    from file_io.xisf_file_io import write_xisf_image
    pixels = np.random.default_rng(0).random((h, w)).astype(np.float32)
    fits.PrimaryHDU(pixels).writeto(d / "chan.fits")
    write_xisf_image(d / "chan.xisf", pixels[..., None])


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr(rc_astro_cli, "MODEL_CACHE", tmp_path / "no-model-cache")
    _channels(tmp_path)
    f = _fake_run_writing_outputs()
    monkeypatch.setattr(rc_astro_cli, "rc_astro_base_command", lambda: ["rc-astro"])
    monkeypatch.setattr(subprocess, "run", f)
    return f


def test_bxt_command_and_relocates_output(fake, tmp_path):
    out = tmp_path / "out" / "bxt.xisf"
    rc_astro_cli.run_bxt(tmp_path / "chan.fits", out, sharpen_stars=0.1)
    cmd = fake.cmd
    assert cmd[cmd.index("--no-banner") + 1] == "bxt"
    assert "--depth" in cmd and cmd[cmd.index("--depth") + 1] == "32F"
    assert "--overwrite" in cmd
    assert cmd[cmd.index("--sharpen-stars") + 1] == "0.1"
    # Model + tiling are always pinned, never left to the CLI's own defaults.
    assert cmd[cmd.index("--ml-version") + 1] == "0"
    assert cmd[cmd.index("--overlap") + 1] == "0.2"
    # BXT forces XISF via an explicit `.xisf` output path (else FITS overflows).
    assert cmd[cmd.index("-o") + 1].endswith(".xisf")
    assert out.exists()  # moved from the temp -o file to the requested path


def test_bxt_correct_only_drops_sharpen(fake, tmp_path):
    rc_astro_cli.run_bxt(tmp_path / "chan.fits", tmp_path / "o.xisf", correct_only=True)
    assert "--correct-only" in fake.cmd
    assert "--sharpen-stars" not in fake.cmd


def test_sxt_starless_only_no_stars_flag(fake, tmp_path):
    out = tmp_path / "starless.xisf"
    rc_astro_cli.run_sxt(tmp_path / "chan.xisf", starless_out=out)
    assert "--stars" not in fake.cmd
    # SXT pins model + tiling too (its only output-affecting flags).
    assert fake.cmd[fake.cmd.index("--ml-version") + 1] == "0"
    assert fake.cmd[fake.cmd.index("--overlap") + 1] == "0.2"
    assert out.exists()


def test_sxt_stars_unscreen_writes_both(fake, tmp_path):
    sl, st = tmp_path / "sl.xisf", tmp_path / "st.xisf"
    rc_astro_cli.run_sxt(tmp_path / "chan.xisf", starless_out=sl, stars_out=st, unscreen=True)
    assert "--stars" in fake.cmd and "--unscreen" in fake.cmd
    assert sl.exists() and st.exists()


def test_sxt_stars_only(fake, tmp_path):
    st = tmp_path / "st.xisf"
    rc_astro_cli.run_sxt(tmp_path / "chan.xisf", stars_out=st, unscreen=True)
    assert "--stars" in fake.cmd
    assert st.exists()


def test_sxt_requires_an_output(fake, tmp_path):
    with pytest.raises(ValueError):
        rc_astro_cli.run_sxt(tmp_path / "chan.xisf")


def test_the_tools_page_location_wins(monkeypatch):
    monkeypatch.setattr(third_party_tools.node_settings, "read",
                        lambda: {"tool_paths": {"rc-astro": "/opt/rc-astro/rc-astro"}})
    assert rc_astro_cli.rc_astro_base_command() == ["/opt/rc-astro/rc-astro"]


def test_missing_binary_raises(monkeypatch):
    monkeypatch.setattr(third_party_tools.node_settings, "read", lambda: {})
    monkeypatch.setattr(third_party_tools.shutil, "which", lambda _: None)
    # Nor at its install location, which this Mac may well have.
    monkeypatch.setattr(third_party_tools.Path, "exists", lambda self: False)
    with pytest.raises(third_party_tools.ToolNotFoundError):
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
    monkeypatch.setattr(rc_astro_cli, "MODEL_CACHE", tmp_path / "no-model-cache")
    _channels(tmp_path)
    with pytest.raises(RuntimeError, match="exited 1"):
        rc_astro_cli.run_bxt(tmp_path / "chan.fits", tmp_path / "o.xisf")


def test_an_image_off_the_ladder_runs_reflect_padded_and_comes_back_on_its_own_grid(fake, tmp_path):
    """The Neural Engine compiles a model bundle per input SHAPE, so a master
    off the 512 ladder runs padded to the next rung (right and bottom) and
    every product is cropped back to the input's exact size and pixels. The
    pad is a REFLECTION of the image, never zeros: SXT stretches from
    whole-image statistics, and a zero rim degraded every star in the frame
    and left the tiles touching the rim unprocessed (2026-11-13)."""
    import pyvips
    from file_io.tiff_file_io import write_tiff_image
    src = tmp_path / "master.tiff"
    pixels = np.random.default_rng(1).random((300, 600)).astype(np.float32)
    write_tiff_image(src, pixels, {})
    out = tmp_path / "bxt.tiff"
    rc_astro_cli.run_bxt(src, out)
    assert fake.fed == (1024, 512)
    assert np.array_equal(fake.fed_pixels, np.pad(pixels, ((0, 212), (0, 424)), mode="reflect"))
    got = pyvips.Image.tiffload(str(out), page=0)
    assert (got.width, got.height) == (600, 300)
    arr = np.ndarray(buffer=got.write_to_memory(), dtype=np.float32, shape=(300, 600))
    assert np.array_equal(arr, pixels)


def test_the_model_cache_keeps_only_bundles_young_enough_to_be_a_running_call(tmp_path):
    import os
    now = 1_000_000_000.0
    root = tmp_path / "e5bundlecache"
    ages = [30 * 3600, 3600, rc_astro_cli.TIMEOUT_S + 1, rc_astro_cli.TIMEOUT_S - 60]
    for i, age_s in enumerate(ages):                       # four 1 kB bundles
        b = root / "25G227" / f"bundle{i}"
        b.mkdir(parents=True)
        (b / "model.bin").write_bytes(b"x" * 1000)
        os.utime(b, (now - age_s, now - age_s))
    assert rc_astro_cli.sweep_model_cache(root, now=now) == 3000
    assert [p.name for p in (root / "25G227").iterdir()] == ["bundle3"]
    assert rc_astro_cli.sweep_model_cache(root, now=now) == 0
