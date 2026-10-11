"""Run Russ Croman's rc-astro CLI (BlurXTerminator / StarXTerminator).

`rc-astro` (https://www.rc-astro.com) ships BXT / SXT / NXT as command-line
tools. We drive them as an EXTERNAL CLI — same shape as `graxpert_cli` — so the
heavy CoreML/GPU ML runtime and models live outside this venv and are installed
/ licensed independently. The all-OSS color pipeline uses BXT (deconvolution,
on linear data) and SXT (star separation).

Install / locate (any one is enough):
  * `rc-astro` on PATH (the installer puts it at /usr/local/bin/rc-astro)
  * set `ASTROFLOW_RC_ASTRO_BIN` to its path (or a whole command string)

Output format is XISF, not FITS: a full-res mosaic channel can exceed 2**31
pixels, which overflows rc-astro's cfitsio FITS *writer* (cfitsio status 306,
"negative number of elements") even though its reader and the ML both handle it.
XISF has no such limit. rc-astro picks the output format from the `-o` filename
extension, so BXT is driven with an explicit `-o <dir>/<stem>-bxt.xisf`; SXT
(two products, one `-o`) writes to `-o <dir>` and inherits XISF from its XISF
input. We then move the produced file(s) to the requested paths (the input — a
full-res mosaic channel — is never copied). `_locate` finds each product by its
`<input-stem>-<product>` base, tolerant of the extension rc-astro chose.

Why XISF and not an OSS format (we prefer TIFF elsewhere): of rc-astro's output
formats, XISF is the ONLY one that survives a >2**31-pixel channel. FITS
overflows cfitsio (status 306); TIFF works small but rc-astro writes CLASSIC
TIFF, not BigTIFF, so a >4 GB output silently corrupts (verified: a 512px BXT
writes a valid TIFF, the 2.25 Gpx BXT writes a 0-byte file). So XISF is confined
here to the transient BXT/SXT output; `autostretch_channel` immediately converts
it to BigTIFF and everything downstream is TIFF + pyvips.

TODO (size-agnostic milestone): rc-astro also holds the WHOLE image in RAM
(~50 GB for 2.25 Gpx), so it can't scale to arbitrarily large single mosaics
regardless of format. The fix is to feed it bounded, OVERLAPPING tiles and
reassemble with pyvips — which also drops XISF (small tiles < 4 GB write fine as
TIFF) and removes the memory ceiling. It's finicky and may cost quality at seams
(BXT/SXT have spatial/PSF context), so the cleaner option is to ask the rc-astro
author to tile internally end-to-end (it already tiles for GPU memory; it just
needs to not hold/write the whole image). Track: see project memory.
"""
import os
import shutil
import subprocess
import tempfile
import uuid
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
    """Publish rc-astro's product to `dest` ATOMICALLY.

    `produced` lives in a local TemporaryDirectory — a DIFFERENT filesystem than
    `dest` — so a plain `shutil.move`/`copy2` writes incrementally straight into
    the final content-addressed path, non-atomically. Under the scheduler's
    re-dispatch (over-submission + broken-file re-queue) two publishers race into
    the same `HASH.xisf`; one truncates the other mid-write, leaving a file cut
    off at the image/metadata boundary that never satisfies the cache → endless
    re-derivation (the BXT loop). Fix: copy to a unique staging file ON THE DEST
    FILESYSTEM, then `os.replace` — a same-filesystem replace is atomic, so a
    reader or a concurrent publisher never observes a partial file, and two
    publishers just both land complete files (last wins), never a torn one.
    """
    if not produced.exists():
        raise FileNotFoundError(f"rc-astro produced no file at {produced}")
    from file_io.publish import publish_atomically
    publish_atomically(produced, dest, consume=True)


def _readable(image: Path, td: Path) -> Path:
    """`image` as rc-astro can read it. Its TIFF reader takes stripped
    none/deflate/LZW only, and AstroFlow writes zstd; a TIFF that needs it gets
    a stripped uncompressed copy under the SAME name (products are named after
    the input's stem) in a subdirectory, so `_locate` never globs it."""
    from file_io.planar import needs_planar, write_planar
    if not needs_planar(image):
        return image
    copy = td / "input" / image.name
    copy.parent.mkdir()
    write_planar(image, copy)
    return copy


def _locate(td: Path, base: str) -> Path:
    """Find rc-astro's product file `<base>.<ext>` in `td`. rc-astro names each
    product `<input-stem>-<product>` and sets the extension from the format it
    wrote (we drive it to XISF); locate by base name, tolerant of that ext."""
    matches = sorted(td.glob(f"{base}.*"))
    if len(matches) == 1:
        return matches[0]
    xisf = [m for m in matches if m.suffix == ".xisf"]
    if len(xisf) == 1:
        return xisf[0]
    raise FileNotFoundError(
        f"rc-astro produced no unique '{base}.*' in {td}: "
        f"{[p.name for p in td.iterdir()]}"
    )


def run_bxt(
    input_image: Path | str,
    output: Path | str,
    *,
    sharpen_stars: float = 0.0,
    sharpen_nonstellar: float = 0.0,
    correct_only: bool = False,
    ml_version: int = 0,
    overlap: float = 0.2,
    timeout: float | None = 1800.0,
) -> None:
    """BlurXTerminator `input_image` to `output` (32-bit float, WCS preserved).

    Output FORMAT follows `output`'s extension. Prefer `.tiff`: rc-astro's XISF
    *writer* is broken — above ~245 MB it silently truncates (declares
    attachment blocks it never writes) while exiting 0, producing unreadable
    files that loop the whole pipeline on re-derivation. Its TIFF writer is fine
    up to classic-TIFF's 4 GB, which covers per-panel/channel data. A bare
    `-o <dir>` would inherit the input's format and can overflow, so we always
    pass an explicit filename carrying the desired extension. Operates on LINEAR
    data. `correct_only` corrects PSF aberrations without sharpening (mutually
    exclusive with the sharpen amounts).

    The sharpen defaults are 0.0 to mirror the rc-astro CLI's own defaults (this
    is a neutral primitive) — callers that want the normal sharpened look must
    ask for it; the OSS pipeline does, via BxtSettings. `ml_version`/`overlap` are
    always passed so the model + tiling are pinned, not left to CLI defaults."""
    input_image, output = Path(input_image), Path(output)
    with tempfile.TemporaryDirectory(prefix="rcastro_bxt_") as td:
        input_image = _readable(input_image, Path(td))
        # Extension drives rc-astro's output format; mirror the caller's request.
        produced = Path(td) / f"{input_image.stem}-bxt{output.suffix}"
        args = ["bxt", str(input_image), "-o", str(produced),
                "--depth", "32F", "--overwrite",
                "--ml-version", str(ml_version), "--overlap", str(overlap)]
        if correct_only:
            args.append("--correct-only")
        else:
            args += ["--sharpen-stars", str(sharpen_stars),
                     "--sharpen-nonstellar", str(sharpen_nonstellar)]
        _run(args, timeout)
        _move(_locate(Path(td), f"{input_image.stem}-bxt"), output)


def run_sxt(
    input_image: Path | str,
    starless_out: Path | str | None = None,
    *,
    stars_out: Path | str | None = None,
    unscreen: bool = False,
    ml_version: int = 0,
    overlap: float = 0.2,
    timeout: float | None = 1800.0,
) -> None:
    """StarXTerminator `input_image`, writing the requested output(s).

    `starless_out` and/or `stars_out` may be given (at least one). `--stars` is
    added whenever `stars_out` is requested; `--unscreen` (only with `--stars`)
    writes screen-ready stars to re-apply with a screen blend. Operates on
    LINEAR data. Products inherit the input's format via `-o <dir>`; fed the
    XISF BXT output, they are XISF (the FITS writer overflows on big channels).

    SXT has no strength dial; `ml_version`/`overlap` are always passed so the
    model + tiling are pinned rather than left to (drifting) CLI defaults."""
    if starless_out is None and stars_out is None:
        raise ValueError("run_sxt needs starless_out and/or stars_out")
    input_image = Path(input_image)
    with tempfile.TemporaryDirectory(prefix="rcastro_sxt_") as td:
        input_image = _readable(input_image, Path(td))
        if stars_out is None:
            # Single product: name it explicitly so the CLI writes the
            # FORMAT the caller's extension requests (as run_bxt does).
            # The directory form inherits the INPUT's format — an XISF
            # input would smuggle XISF bytes into a .tiff-named artifact
            # (sxt_channel with the lab's bxt=False chain, 2026-09-02).
            out_arg = str(Path(td) /
                          f"{input_image.stem}-sxt{Path(starless_out).suffix}")
        else:
            out_arg = td  # two products: directory mode, formats inherited
        args = ["sxt", str(input_image), "-o", out_arg, "--depth", "32F",
                "--overwrite",
                "--ml-version", str(ml_version), "--overlap", str(overlap)]
        if stars_out is not None:
            args.append("--stars")
            if unscreen:
                args.append("--unscreen")
        _run(args, timeout)
        if starless_out is not None:
            _move(_locate(Path(td), f"{input_image.stem}-sxt"), Path(starless_out))
        if stars_out is not None:
            _move(_locate(Path(td), f"{input_image.stem}-sxt-stars"), Path(stars_out))


def run_nxt(
    input_image: Path | str,
    output_xisf: Path | str,
    *,
    denoise_intensity_hf: float = 0.85,   # HF intensity: manual says 80-90%
    denoise_intensity_lf: float = 0.60,   # LF intensity: 50-70% (keeps dust-cloud variation)
    denoise_color_hf: float = 0.95,       # HF colour: 90-100%
    denoise_color_lf: float = 1.00,       # LF colour: 100% (kills large-scale colour blotches)
    frequency_scale: float = 5.0,         # HF/LF transition scale (rc-astro default)
    iterations: int = 2,                  # successive-approximation steps (AI3)
    overlap: float = 0.2,
    timeout: float | None = 1800.0,
) -> None:
    """NoiseXTerminator `input_image` → `output_xisf` (32-bit float).

    Per the RC-Astro manual, NXT is applied AFTER deconvolution (BXT) and AFTER
    channel combination, on the COLOR (multi-band) image, on LINEAR data. The
    intensity/colour + high/low-frequency separation only works on a multi-band
    image and is the whole point for narrowband — it lets us reduce the colour
    noise hard (weak SII/OIII chroma → the objectionable blotches) while keeping
    intensity (Hα-dominated luminance detail) light. Defaults are the manual's
    recommended best-quality combination. XISF output (gigapixel-safe; rc-astro's
    FITS writer overflows on huge images)."""
    input_image, output_xisf = Path(input_image), Path(output_xisf)
    with tempfile.TemporaryDirectory(prefix="rcastro_nxt_") as td:
        input_image = _readable(input_image, Path(td))
        produced = Path(td) / f"{input_image.stem}-nxt.xisf"
        args = ["nxt", str(input_image), "-o", str(produced),
                "--depth", "32F", "--overwrite",
                "--iterations", str(iterations), "--overlap", str(overlap),
                "--frequency-scale", str(frequency_scale),
                "--denoise-intensity-high-freq", str(denoise_intensity_hf),
                "--denoise-intensity-low-freq", str(denoise_intensity_lf),
                "--denoise-color-high-freq", str(denoise_color_hf),
                "--denoise-color-low-freq", str(denoise_color_lf)]
        _run(args, timeout)
        _move(_locate(Path(td), f"{input_image.stem}-nxt"), output_xisf)
