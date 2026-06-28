"""Run Russ Croman's rc-astro CLI (BlurXTerminator / StarXTerminator).

`rc-astro` (https://www.rc-astro.com) ships BXT / SXT / NXT as command-line
tools. We drive them as an EXTERNAL CLI — same shape as `graxpert_cli` — so the
heavy CoreML/GPU ML runtime and models live outside this venv and are installed
/ licensed independently. The all-OSS color pipeline uses BXT (deconvolution,
on linear data) and SXT (star separation).

Install / locate (any one is enough):
  * `rc-astro` on PATH (the installer puts it at /usr/local/bin/rc-astro)
  * set `ASTROFLOW_RC_ASTRO_BIN` to its path (or a whole command string)

Output-naming (verified against rc-astro 0.9.3 with `-o <dir>`):
  bxt  <stem>.fits -> <dir>/<stem>-bxt.fits
  sxt  <stem>.fits -> <dir>/<stem>-sxt.fits                 (starless)
  sxt --stars      -> ... plus <dir>/<stem>-sxt-stars.fits  (stars)
So we run with `-o <tempdir>` and move the deterministically-named output(s) to
the requested paths (the input — a full-res mosaic channel — is never copied).
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import structlog

log = structlog.get_logger()


class RcAstroNotFoundError(RuntimeError):
    """rc-astro could not be located. Install it or set ASTROFLOW_RC_ASTRO_BIN."""


def rc_astro_base_command() -> list[str]:
    """How to invoke rc-astro: `$ASTROFLOW_RC_ASTRO_BIN` (a path or whole command
    string), else an `rc-astro` on PATH. Raises if neither is found."""
    override = os.environ.get("ASTROFLOW_RC_ASTRO_BIN")
    if override:
        return override.split()
    on_path = shutil.which("rc-astro")
    if on_path:
        return [on_path]
    raise RcAstroNotFoundError(
        "rc-astro not found. Install it, or set ASTROFLOW_RC_ASTRO_BIN to the "
        "rc-astro executable."
    )


def _run(subcommand_args: list[str], timeout: float | None) -> None:
    """Run `rc-astro --no-banner <subcommand_args>`, raising on a non-zero exit.

    `--no-banner` is a top-level flag (before the subcommand); per-product flags
    (`--depth`, `-o`, `--overwrite`, ...) live in `subcommand_args` after it."""
    cmd = rc_astro_base_command() + ["--no-banner"] + subcommand_args
    log.info("Running rc-astro", cmd=" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"rc-astro exited {result.returncode}.\n"
            f"stdout tail:\n{result.stdout[-2000:]}\n"
            f"stderr tail:\n{result.stderr[-2000:]}"
        )


def _move(produced: Path, dest: Path) -> None:
    if not produced.exists():
        raise FileNotFoundError(f"rc-astro produced no file at {produced}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), str(dest))  # cross-filesystem safe (temp -> output)


def run_bxt(
    input_fits: Path | str,
    output_fits: Path | str,
    *,
    sharpen_stars: float = 0.0,
    sharpen_nonstellar: float = 0.0,
    correct_only: bool = False,
    timeout: float | None = 1800.0,
) -> None:
    """BlurXTerminator `input_fits` to `output_fits` (32-bit float, WCS preserved).

    Operates on LINEAR data. `correct_only` corrects PSF aberrations without
    sharpening (mutually exclusive with the sharpen amounts)."""
    input_fits, output_fits = Path(input_fits), Path(output_fits)
    with tempfile.TemporaryDirectory(prefix="rcastro_bxt_") as td:
        args = ["bxt", str(input_fits), "-o", td, "--depth", "32F", "--overwrite"]
        if correct_only:
            args.append("--correct-only")
        else:
            args += ["--sharpen-stars", str(sharpen_stars),
                     "--sharpen-nonstellar", str(sharpen_nonstellar)]
        _run(args, timeout)
        _move(Path(td) / f"{input_fits.stem}-bxt.fits", output_fits)


def run_sxt(
    input_fits: Path | str,
    starless_out: Path | str | None = None,
    *,
    stars_out: Path | str | None = None,
    unscreen: bool = False,
    timeout: float | None = 1800.0,
) -> None:
    """StarXTerminator `input_fits`, writing the requested output(s).

    `starless_out` and/or `stars_out` may be given (at least one). `--stars` is
    added whenever `stars_out` is requested; `--unscreen` (only with `--stars`)
    writes screen-ready stars to re-apply with a screen blend. Operates on
    LINEAR data."""
    if starless_out is None and stars_out is None:
        raise ValueError("run_sxt needs starless_out and/or stars_out")
    input_fits = Path(input_fits)
    with tempfile.TemporaryDirectory(prefix="rcastro_sxt_") as td:
        args = ["sxt", str(input_fits), "-o", td, "--depth", "32F", "--overwrite"]
        if stars_out is not None:
            args.append("--stars")
            if unscreen:
                args.append("--unscreen")
        _run(args, timeout)
        if starless_out is not None:
            _move(Path(td) / f"{input_fits.stem}-sxt.fits", Path(starless_out))
        if stars_out is not None:
            _move(Path(td) / f"{input_fits.stem}-sxt-stars.fits", Path(stars_out))
