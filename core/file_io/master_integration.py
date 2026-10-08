"""Stacker-agnostic reader for a master frame's effective integration time.

A stacked master's "real" integration time is whatever the stacker chose
to keep after its own rejection pass — that number rarely matches
sum(EXPTIME) over the input subs, because the stacker drops blurry,
satellite-trailed, or otherwise unwanted frames. Each stacker stores it
in a different place; this dispatcher centralizes the per-stacker
readers behind one interface so `compute_image_metadata` (and any other
caller) can pull a stacker-agnostic effective integration.

Returns the integration in seconds, or None if no source matches. Adding
a new stacker is one entry in `_READERS`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from file_io.xisf_file_io import read_xisf_number_of_images


_Reader = Callable[[Path, dict[str, Any]], "float | None"]


def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _read_xisf(filename: Path, header: dict[str, Any]) -> float | None:
    """PixInsight master: integrated-frame count × per-sub EXPTIME.

    The count is `numberOfImages` from the XISF `PixInsight:ProcessingHistory`
    (frames that survived registration/rejection); EXPTIME is the uniform
    per-sub exposure from the header. We deliberately do NOT read
    `PCL:TotalExposureTime`: that is a PSF-WEIGHTED effective exposure (a
    fractional multiple of EXPTIME), which collapses toward ~0 for weak-star
    narrowband stacks and made good, fully-integrated masters look "0%
    integrated". Returns None when the count is unreadable or EXPTIME is
    absent/0 (e.g. a bias master), so callers fall back to the prediction.
    """
    if filename.suffix.lower() != ".xisf":
        return None
    n = read_xisf_number_of_images(filename)
    if n is None:
        return None
    exptime = _safe_float(header.get("EXPTIME"))
    if not exptime:
        return None
    return n * exptime


def _read_siril_fits(filename: Path, header: dict[str, Any]) -> float | None:
    """Siril master: `STACKCNT` (images integrated) x per-sub `EXPTIME`, or
    `LIVETIME` when only the total is present."""
    if filename.suffix.lower() not in (".fit", ".fits"):
        return None
    n, exptime = _safe_float(header.get("STACKCNT")), _safe_float(header.get("EXPTIME"))
    if n and exptime:
        return n * exptime
    return _safe_float(header.get("LIVETIME"))


# Registered in priority order — first match wins. Each reader is free
# to inspect the header and skip cheaply when the master isn't theirs.
_READERS: tuple[_Reader, ...] = (
    _read_xisf,
    _read_siril_fits,
)


def read_master_integration_seconds(
    filename: Path, header: dict[str, Any]
) -> float | None:
    """Effective post-rejection integration time of a stacked master.

    Tries each registered stacker-specific reader and returns the first
    non-None result. Returns None if no reader knows how to extract it
    (raw subframe, unrecognized stacker, missing property). Callers
    treat None as "no master integration data available."
    """
    for reader in _READERS:
        result = reader(filename, header)
        if result is not None:
            return result
    return None
