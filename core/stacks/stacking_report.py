"""Stacker-agnostic summary of a master image.

A "stacking report" is whatever can be extracted from the on-disk
master file that tells you what the stacker actually did — total
integration time, frames combined, noise estimate, date range, output
dimensions, the pipeline that produced it. The shape is identical
across PixInsight WBPP, AstroFlow-OSS, Siril, etc.; the *source* of
each field varies (XISF Properties for WBPP, FITS keywords for
everything else).

Producers that don't write a value get None for that field — UI
renders accordingly. The goal is to surface the same useful facts
regardless of which stacker ran, so the user can compare like-for-like
across implementations.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from file_io.generic import read_header
from file_io.xisf_file_io import read_xisf_total_exposure_time


@dataclass(frozen=True)
class StackingReport:
    pipeline: str  # human-readable label, e.g. "PixInsight WBPP"

    # The headline number. Sum of EXPTIME across frames that actually
    # contributed to the master (weight > 0). None if the file doesn't
    # carry it.
    total_integration_seconds: Optional[float]

    # How many input frames were integrated. NCOMBINE / NUMINTEG.
    frames_combined: Optional[int]

    # Per-frame exposure (when uniform across inputs). EXPTIME / EXPOSURE
    # of the source frames; the master keeps it as a hint.
    frame_exposure_seconds: Optional[float]

    # Robust noise estimate from PixInsight's NOISExx headers (units
    # of normalized image data). Lower is better; useful for
    # comparing across runs of the same stack.
    noise_estimate: Optional[float]

    # ISO 8601 timestamps of the earliest input frame and the latest
    # frame's end-of-exposure. Together they give "imaging window."
    date_obs: Optional[str]
    date_end: Optional[str]

    # Output pixel dimensions of the master (post-crop, post-register).
    output_width: Optional[int]
    output_height: Optional[int]

    def to_json(self) -> dict:
        return asdict(self)


def _detect_pipeline(path: Path, header: dict) -> str:
    explicit = header.get("PIPELINE")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if path.suffix.lower() == ".xisf":
        return "PixInsight WBPP"
    creator = header.get("CREATOR") or header.get("PROGRAM")
    if creator:
        return str(creator)
    return "unknown"


def _safe_float(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(v) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def read_stacking_report(master_path: Path | str) -> StackingReport:
    """Build a StackingReport from a master image on disk.

    Reads the FITS-style header for the cross-pipeline fields, plus
    XISF Properties for fields PixInsight stores outside the FITS
    keyword block. Tolerates missing fields — anything not present
    on disk comes back as None.
    """
    path = Path(master_path)
    header = read_header(path)
    pipeline = _detect_pipeline(path, header)

    # Total integration: PixInsight stores it as PCL:TotalExposureTime
    # in the XISF property block (not FITS keywords). Other stackers
    # use LIVETIME / TOTALEXP / NCOMBINE × EXPTIME as a fallback.
    total = None
    if path.suffix.lower() == ".xisf":
        total = read_xisf_total_exposure_time(path)
    if total is None:
        total = _safe_float(header.get("LIVETIME") or header.get("TOTALEXP"))
    if total is None:
        ncomb = _safe_int(header.get("NCOMBINE") or header.get("NUMINTEG"))
        per = _safe_float(header.get("EXPTIME") or header.get("EXPOSURE"))
        if ncomb and per:
            total = ncomb * per

    return StackingReport(
        pipeline=pipeline,
        total_integration_seconds=total,
        frames_combined=_safe_int(header.get("NCOMBINE") or header.get("NUMINTEG")),
        frame_exposure_seconds=_safe_float(
            header.get("EXPTIME") or header.get("EXPOSURE")
        ),
        noise_estimate=_safe_float(header.get("NOISE00")),
        date_obs=str(header["DATE-OBS"]) if header.get("DATE-OBS") else None,
        date_end=str(header["DATE-END"]) if header.get("DATE-END") else None,
        output_width=_safe_int(header.get("NAXIS1")),
        output_height=_safe_int(header.get("NAXIS2")),
    )
