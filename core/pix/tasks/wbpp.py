"""WeightedBatchPreprocessing — batch stacking of subframes.

WBPP is unique among PixInsight tasks: it operates on directories of files
rather than open views, and produces output files with unpredictable names.
It doesn't fit the lazy Image model, so it's exposed as a direct execution
function that runs a script-based Step immediately.
"""
import statistics
import re
from pathlib import Path
from typing import Any

import structlog

from pix.recipe import Step

log = structlog.get_logger()


class WBPPDuplicateMastersError(Exception):
    """WBPP wrote multiple master files of the same logical name —
    typically an internal naming collision that silently produced
    extra autocropped (or other) masters. Caught and re-raised so
    downstream code doesn't pick one at random and proceed.
    """

_SCRIPT_FILENAME = (
    "/Applications/PixInsight/src/scripts/BatchPreprocessing/WBPP.js"
)

_DEFAULT_PARAMETERS = {
    "outputDirectory": "",
    # Force full ImageIntegration on every group. WBPP's default
    # `autoIntegrationMode` silently switches any light group of >=150 frames
    # to FastIntegration (BPP-FrameGroup.js:319), which is faster but stores
    # NONE of the per-frame weights or the `finalNoiseEstimate` (MRS) suite we
    # rely on for effective-integration / depth. Disabling it makes every
    # master carry the full noise + weight metadata, uniformly, at the cost of
    # slower/heavier integration on the largest stacks. Nothing else gates on
    # this flag (audited across the WBPP source).
    "autoIntegrationMode": "false",
}


def solver_geometry(headers) -> dict[str, float]:
    """The camera geometry WBPP's built-in image solver should assume.

    WBPP defaults to `imageSolverPixelSize: 5.4` and `imageSolverFocalLength:
    750.0` (BPP-Global.js). It measures the true plate scale from the star
    field regardless, so a solve never fails and nothing looks wrong, but it
    writes the result back DECOMPOSED against those assumptions: every master
    we had produced carried XPIXSZ 5.4 and FOCALLEN = the real one x 5.4/3.76
    = 1.436 (49 of 53 masters audited, 2026-09-08). The WCS is correct and
    association never reads a master's header, so the damage is confined to
    those two keywords -- but `plate_solving._header_hints` builds the
    solver's scale window from exactly that pair, so a re-solve of such a
    master can hint a scale that cannot match.

    Returns {} when the frames don't carry both keywords, or when they are
    binned: XPIXSZ means the physical pixel on some capture programs and the
    binned pixel on others, and guessing wrong would be worse than WBPP's
    own default. Empty simply leaves today's behaviour in place.
    """
    pixel_sizes: list[float] = []
    focal_lengths: list[float] = []
    for header in headers:
        try:
            if float(header.get("XBINNING", 1) or 1) != 1.0:
                return {}
        except (TypeError, ValueError):
            return {}
        for key, out in (("XPIXSZ", pixel_sizes), ("FOCALLEN", focal_lengths)):
            value = header.get(key)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                out.append(value)
    if not pixel_sizes or not focal_lengths:
        return {}
    return {
        "imageSolverPixelSize": statistics.median(pixel_sizes),
        "imageSolverFocalLength": statistics.median(focal_lengths),
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
        timeout: Optional ABSOLUTE wall-clock ceiling (seconds); None =
            unlimited. Liveness is enforced by the runner's inactivity
            watchdog, so a legitimately-long stack should pass None and let
            it run as long as it keeps making progress.
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
        # Watch `output_dir` (not the sentinel `tmp_dir`): that's where WBPP
        # streams its per-frame calibrated/registered intermediates, so its
        # mtime is the file-activity signal for the inactivity watchdog.
        run_scp(scp, Path(tmp_dir), slots=slots, watch_dir=output_dir, timeout=timeout)

    # Re-emit WBPP's log files via structlog so the surrounding
    # @capture_logs decorator on stack_files picks them up alongside its
    # own logs. WBPP writes a top-level WBPP-LOG-*.log plus per-step
    # text logs in a `logs/` subdir; both are useful for diagnosing why
    # frames were rejected. Capture before validation so the log is
    # available in the per-call entry even when validation raises.
    _capture_wbpp_logs(output_dir)
    _validate_wbpp_outputs(output_dir)


# Trailing `_<digits>.xisf` retry suffix — strip so a retried master
# write groups with its original. Doesn't match `_2x` (drizzle
# variants) because the regex requires `.xisf` immediately after.
_TRAILING_RETRY_SUFFIX = re.compile(r"_\d+(?=\.xisf$)")


def _validate_wbpp_outputs(output_dir: Path) -> None:
    """Raise if WBPP wrote multiple master files of the same flavor.

    A normal WBPP run produces several distinct master files per
    integration: the primary master, an autocropped variant, sometimes
    a drizzled variant. Each should be unique. Two files with the same
    logical name (in different subdirectories, or differing only by a
    `_N` retry suffix) signal an internal WBPP naming collision —
    raising here prevents the surrounding code from picking one at
    random and silently proceeding with possibly the wrong master.

    Also emits a concern with the duplicate paths, so the failure is
    visible on the stack's detail page even after the exception
    propagates.
    """
    masters = list(output_dir.rglob("master*.xisf"))
    by_normalized: dict[str, list[Path]] = {}
    for m in masters:
        by_normalized.setdefault(_TRAILING_RETRY_SUFFIX.sub("", m.name), []).append(m)

    duplicates = {n: paths for n, paths in by_normalized.items() if len(paths) > 1}
    if not duplicates:
        return

    from concerns import error as _concern_error

    detail_lines = []
    for name, paths in sorted(duplicates.items()):
        detail_lines.append(f"{name}:")
        for p in paths:
            try:
                rel = p.relative_to(output_dir)
            except ValueError:
                rel = p
            detail_lines.append(f"  {rel}")
    detail = "\n".join(detail_lines)

    _concern_error(
        title="WBPP produced duplicate master files",
        detail=(
            "WBPP wrote multiple files with the same logical name. This "
            "typically indicates an internal naming collision — the "
            "captured WBPP log for this run should explain what went "
            "wrong.\n\n"
            f"Duplicates:\n{detail}"
        ),
        duplicate_count=sum(len(paths) for paths in duplicates.values()),
        duplicate_names=sorted(duplicates.keys()),
    )
    raise WBPPDuplicateMastersError(
        f"WBPP produced duplicate masters: {sorted(duplicates.keys())}"
    )


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
