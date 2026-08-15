"""Run GraXpert background-extraction from the command line.

GraXpert (https://github.com/Steffenhir/GraXpert) is an open-source AI
background / gradient corrector — the OSS counterpart to PixInsight's
MultiscaleGradientCorrection, minus the MARS external reference (GraXpert is
data-driven). We drive it as an EXTERNAL CLI tool so its heavy ML runtime and
AI models live outside this venv and are installed / updated independently.

Install (any one is enough):
  * ``pip install graxpert``      → invoked as ``python -m graxpert.main``
  * a standalone GraXpert app     → set ``ASTROFLOW_GRAXPERT_BIN`` to its path
  * a ``graxpert`` on ``PATH``

CLI shape (verified against the installed GraXpert 3.2 `-h`, which differs from
the README — smoothing/correction moved into a preferences file):
  ``<graxpert> <input.fits> -cmd background-extraction -output <name>
     [-gpu true|false] [-preferences_file <prefs.json>]``
"""
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import structlog

log = structlog.get_logger()


class GraXpertNotFoundError(RuntimeError):
    """GraXpert could not be located. Install it or set ASTROFLOW_GRAXPERT_BIN."""


def graxpert_base_command() -> list[str]:
    """How to invoke GraXpert, resolved in priority order:

    1. ``$ASTROFLOW_GRAXPERT_BIN`` — a path, or a whole command string.
    2. a ``graxpert`` executable on ``PATH``.
    3. ``<this-python> -m graxpert.main`` if the ``graxpert`` package is installed.
    """
    override = os.environ.get("ASTROFLOW_GRAXPERT_BIN")
    if override:
        return override.split()
    on_path = shutil.which("graxpert")
    if on_path:
        return [on_path]
    if importlib.util.find_spec("graxpert") is not None:
        return [sys.executable, "-m", "graxpert.main"]
    raise GraXpertNotFoundError(
        "GraXpert not found. `pip install graxpert`, or set "
        "ASTROFLOW_GRAXPERT_BIN to the GraXpert executable."
    )


def run_background_extraction(
    input_fits: Path | str,
    output_fits: Path | str,
    *,
    smoothing: float | None = None,
    correction: str | None = None,
    ai_version: str | None = None,
    gpu: bool | None = None,
    preferences_file: Path | str | None = None,
    timeout: float | None = 1800.0,
) -> None:
    """Background-extract ``input_fits`` with GraXpert, writing ``output_fits``.

    Verified against GraXpert 3.2.0a2's ``background-extraction -h``: the flags
    are ``-correction {Subtraction,Division}``, ``-smoothing`` (0-1),
    ``-ai_version``, ``-gpu``, ``-bg``, and ``-preferences_file``. (An earlier
    docstring here claimed ``-smoothing``/``-correction`` had been removed in
    favour of the preferences file — they are still present, and the
    preferences file is only needed for the non-AI interpolation methods, which
    additionally require background sample points and therefore cannot run
    headless.)

    ``ai_version`` is worth PINNING rather than defaulting to "latest": on the
    2026-08-14 Heart Nebula sweep, model 1.0.1 left a 0.27% background bowl
    where 1.0.0 left 0.61%.

    ``-output`` takes a path that GraXpert resolves to a ``.fits`` file; we pass
    the stem and verify the result lands at ``output_fits`` (renaming if needed).
    NOTE the stem must not contain a dot — GraXpert treats the trailing segment
    as an extension and replaces it (``/tmp/out0.25`` becomes ``/tmp/out0.fits``).
    Raises on a non-zero exit or a missing output, surfacing GraXpert's stdout.
    """
    input_fits = Path(input_fits)
    output_fits = Path(output_fits)
    output_fits.parent.mkdir(parents=True, exist_ok=True)

    out_stem = output_fits.with_suffix("")  # GraXpert appends the extension
    cmd = graxpert_base_command() + [
        str(input_fits),
        "-cli",  # deprecated-but-accepted; keeps older GraXpert out of GUI mode
        "-cmd", "background-extraction",
        "-output", str(out_stem),
    ]
    if smoothing is not None:
        cmd += ["-smoothing", str(smoothing)]
    if correction is not None:
        cmd += ["-correction", correction]
    if ai_version is not None:
        cmd += ["-ai_version", ai_version]
    if gpu is not None:  # otherwise let GraXpert auto-detect the provider
        cmd += ["-gpu", "true" if gpu else "false"]
    if preferences_file is not None:
        cmd += ["-preferences_file", str(preferences_file)]

    log.info("Running GraXpert background-extraction",
             input=str(input_fits), output=str(output_fits), cmd=" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"GraXpert exited {result.returncode}.\n"
            f"stdout tail:\n{result.stdout[-2000:]}\n"
            f"stderr tail:\n{result.stderr[-2000:]}"
        )

    produced = out_stem.with_suffix(".fits")
    if produced.resolve() != output_fits.resolve() and produced.exists():
        produced.replace(output_fits)
    if not output_fits.exists():
        raise FileNotFoundError(
            f"GraXpert produced no file at {output_fits} (expected {produced}).\n"
            f"stdout tail:\n{result.stdout[-1500:]}"
        )
