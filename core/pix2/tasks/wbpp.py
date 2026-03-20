"""WeightedBatchPreprocessing — batch stacking of subframes.

WBPP is unique among PixInsight tasks: it operates on directories of files
rather than open views, and produces output files with unpredictable names.
It doesn't fit the lazy Image model, so it's exposed as a direct execution
function that runs a script-based Step immediately.
"""
from pathlib import Path
from typing import Any

from pix2.recipe import Step

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
    from pix2.runner import run_scp
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

    with tempfile.TemporaryDirectory(prefix="pix2-wbpp-") as tmp_dir:
        run_scp(scp, Path(tmp_dir), slots=slots, timeout=timeout)
