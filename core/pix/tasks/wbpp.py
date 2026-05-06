"""WeightedBatchPreprocessing — batch stacking of subframes.

WBPP is unique among PixInsight tasks: it operates on directories of files
rather than open views, and produces output files with unpredictable names.
It doesn't fit the lazy Image model, so it's exposed as a direct execution
function that runs a script-based Step immediately.
"""
from pathlib import Path
from typing import Any

import structlog

from pix.recipe import Step

log = structlog.get_logger()

_SCRIPT_FILENAME = (
    "/Applications/PixInsight/src/scripts/BatchPreprocessing/WBPP.js"
)

_DEFAULT_PARAMETERS = {
    "outputDirectory": "",
}


def run_wbpp(
    working_dir: Path,
    output_dir: Path,
    slots: list[int] | None = None,
    timeout: float | None = None,
    **kwargs: Any,
) -> None:
    """Run WeightedBatchPreprocessing on a directory of files.

    This is a direct execution function — it runs PixInsight immediately,
    unlike the lazy Image-based tasks.

    Args:
        working_dir: Directory containing input subframes.
        output_dir: Directory where WBPP writes its outputs.
        slots: PixInsight slot numbers to use.
        timeout: Maximum seconds to wait for completion.
        **kwargs: Additional WBPP parameters.
    """
    from pix.runner import run_scp
    import tempfile

    parameters = _DEFAULT_PARAMETERS | kwargs | {
        "outputDirectory": str(output_dir),
    }
    arguments = {
        "automationMode": "true",
        "dir": str(working_dir),
        "smartNamingOverride": "true",
        "keywords": "RUN",
        "platesolveFallbackManual": "false",
    }

    args_str = " ".join(f'-a="{k}={v}"' for k, v in arguments.items())
    params_str = " ".join(f'-p="{k},{v}"' for k, v in parameters.items())
    scp = f'run -x {args_str} {params_str} "{_SCRIPT_FILENAME}"'

    with tempfile.TemporaryDirectory(prefix="pix-wbpp-") as tmp_dir:
        run_scp(scp, Path(tmp_dir), slots=slots, timeout=timeout)

    # Re-emit WBPP's log files via structlog so the surrounding
    # @capture_logs decorator on stack_files picks them up alongside its
    # own logs. WBPP writes a top-level WBPP-LOG-*.log plus per-step
    # text logs in a `logs/` subdir; both are useful for diagnosing why
    # frames were rejected.
    _capture_wbpp_logs(output_dir)


def _capture_wbpp_logs(output_dir: Path) -> None:
    seen: set[Path] = set()
    for pattern in ("WBPP-LOG-*.log", "WBPP-*.log", "logs/*.txt", "logs/*.log"):
        for path in sorted(output_dir.glob(pattern)):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            try:
                content = path.read_text(errors="replace")
            except Exception as e:
                log.warning("Could not read WBPP log", path=str(path), error=str(e))
                continue
            log.info("WBPP log", source=path.name, content=content)
