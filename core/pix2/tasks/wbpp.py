"""WeightedBatchPreprocessing — batch stacking of subframes.

WBPP is unique among PixInsight tasks: it operates on directories of files
rather than open views, and produces output files with unpredictable names.
It doesn't fit the lazy Image model well, so it's exposed as a direct
execution function rather than a recipe step.
"""
from pathlib import Path
from typing import Any

from pix2.recipe import Step
from pix2.runner import run_javascript

_JAVASCRIPT_FILENAME = (
    "/Applications/PixInsight/src/scripts/BatchPreprocessing/WBPP.js"
)

_DEFAULT_PARAMETERS = {
    "outputDirectory": "",
}


def run_wbpp(
    working_dir: Path,
    output_dir: Path,
    mock: bool = False,
    **kwargs: Any,
) -> None:
    """Run WeightedBatchPreprocessing on a directory of files.

    This is a direct execution function — it runs PixInsight immediately,
    unlike the lazy Image-based tasks.

    Args:
        working_dir: Directory containing input subframes.
        output_dir: Directory where WBPP writes its outputs.
        mock: If True, skip PixInsight entirely.
        **kwargs: Additional WBPP parameters.
    """
    if mock:
        return

    parameters = _DEFAULT_PARAMETERS | kwargs | {"outputDirectory": str(output_dir)}

    param_lines = "\n".join(
        f'-p="{k},{v}"' for k, v in parameters.items()
    )
    arg_lines = "\n".join([
        f'-a="automationMode=true"',
        f'-a="dir={working_dir}"',
        f'-a="smartNamingOverride=true"',
        f'-a="keywords=RUN"',
        f'-a="platesolveFallbackManual=false"',
    ])

    # WBPP is invoked as a script with parameters and arguments
    # via the .scp shell script, not via inline JavaScript
    js = f"""
console.beginLog();
console.writeln("WBPP: working_dir={working_dir}");
console.writeln("WBPP: output_dir={output_dir}");
// WBPP will be invoked via .scp script parameters
var log = console.endLog();
File.writeTextFile("{output_dir}/wbpp_log.txt", log.toString());
"""

    import tempfile
    with tempfile.TemporaryDirectory(prefix="pix2-wbpp-") as tmp_dir:
        run_javascript(js, Path(tmp_dir))
