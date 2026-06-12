"""Serialize/deserialize ImageMetadata to/from sidecar JSON files."""
from __future__ import annotations

import datetime
import numpy as np
from beartype import beartype
from pathlib import Path
from typing import Any

from file_io.sidecar import get_section, set_section, get_file_size, sidecar_path_for
from metadata.classes import ImageMetadata, ImageStatistics
import plate_solving
import metadata.image_types as image_types


def _statistics_to_obj(stats: ImageStatistics) -> dict[str, Any]:
    """Serialize ImageStatistics to a JSON-safe dict. The histogram is
    written as a plain list of ints; everything else is already
    primitive. Additive on the sidecar — old readers ignore it."""
    return {
        "histogram_bins": stats.histogram_bins,
        "histogram_range": list(stats.histogram_range),
        "histogram_counts": list(stats.histogram_counts),
        "p0_1": stats.p0_1, "p1": stats.p1, "p5": stats.p5,
        "p25": stats.p25, "p50": stats.p50, "p75": stats.p75,
        "p95": stats.p95, "p99": stats.p99, "p99_9": stats.p99_9,
        "sky_median": stats.sky_median, "sky_rms": stats.sky_rms,
        "clipped_low_fraction": stats.clipped_low_fraction,
        "clipped_high_fraction": stats.clipped_high_fraction,
        "min_value": stats.min_value,
        "max_value": stats.max_value,
        "mean_value": stats.mean_value,
    }


def _statistics_from_obj(obj: dict[str, Any] | None) -> ImageStatistics | None:
    """Reconstruct ImageStatistics from a sidecar dict. Returns None for
    legacy sidecars that predate this field — consumers should treat
    that as 'no per-image stats; use a sensible default'."""
    if obj is None:
        return None
    return ImageStatistics(
        histogram_bins=obj["histogram_bins"],
        histogram_range=tuple(obj["histogram_range"]),
        histogram_counts=tuple(obj["histogram_counts"]),
        p0_1=obj["p0_1"], p1=obj["p1"], p5=obj["p5"],
        p25=obj["p25"], p50=obj["p50"], p75=obj["p75"],
        p95=obj["p95"], p99=obj["p99"], p99_9=obj["p99_9"],
        sky_median=obj["sky_median"], sky_rms=obj["sky_rms"],
        clipped_low_fraction=obj["clipped_low_fraction"],
        clipped_high_fraction=obj["clipped_high_fraction"],
        min_value=obj["min_value"],
        max_value=obj["max_value"],
        mean_value=obj["mean_value"],
    )


@beartype
def _serialize_header(header: dict[str, Any]) -> dict[str, Any]:
    """Convert header values to JSON-safe types."""
    out = {}
    for k, v in header.items():
        if isinstance(v, datetime.datetime):
            out[k] = v.isoformat()
        elif isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        else:
            out[k] = str(v)
    return out


@beartype
def _deserialize_header(header: dict[str, Any]) -> dict[str, Any]:
    """Restore header values from JSON, parsing DATE keys back to datetime."""
    from dateutil.parser import parse

    out = {}
    for k, v in header.items():
        if "DATE" in k and isinstance(v, str):
            try:
                out[k] = parse(v)
                continue
            except (ValueError, OverflowError):
                pass
        out[k] = v
    return out


@beartype
def image_metadata_to_obj(meta: ImageMetadata) -> dict[str, Any]:
    """Serialize ImageMetadata to a JSON-safe dict (excluding filename and plate_solution)."""
    obj: dict[str, Any] = {
        "content_hash": meta.content_hash,
        "stable_hash": meta.stable_hash,
        "header": _serialize_header(meta.header),
        "width_pixels": meta.width_pixels,
        "height_pixels": meta.height_pixels,
        "channels": meta.channels,
        "dtype": str(meta.dtype),
    }
    if meta.master_integration_s is not None:
        obj["master_integration_s"] = meta.master_integration_s
    if meta.statistics is not None:
        obj["statistics"] = _statistics_to_obj(meta.statistics)
    return obj


@beartype
def image_metadata_from_obj(
    obj: dict[str, Any], filename: Path
) -> ImageMetadata:
    """Reconstruct ImageMetadata from a sidecar dict."""
    # Load plate solution from the same sidecar file (not via solve_field,
    # which would redundantly re-read the sidecar or worse, re-run the solver).
    plate_solution = None
    if image_types.is_light_frame(obj["header"]):
        ps_obj = get_section(filename, "plate_solution")
        if isinstance(ps_obj, dict) and ps_obj.get("data") is not None:
            plate_solution = plate_solving._plate_solution_from_obj(ps_obj)

    content_hash = obj["content_hash"]

    return ImageMetadata(
        content_hash=content_hash,
        stable_hash=obj.get("stable_hash", content_hash),
        filename=filename,
        header=_deserialize_header(obj["header"]),
        width_pixels=obj["width_pixels"],
        height_pixels=obj["height_pixels"],
        channels=obj["channels"],
        dtype=np.dtype(obj["dtype"]),
        plate_solution=plate_solution,
        master_integration_s=obj.get("master_integration_s"),
        statistics=_statistics_from_obj(obj.get("statistics")),
    )


@beartype
def write_image_metadata(meta: ImageMetadata) -> None:
    """Write ImageMetadata to the sidecar file for the given image."""
    set_section(meta.filename, "image_metadata", image_metadata_to_obj(meta))


@beartype
def read_image_metadata(filename: Path) -> ImageMetadata | None:
    """Read ImageMetadata from the sidecar file, or return None if missing."""
    obj = get_section(filename, "image_metadata")
    if obj is None:
        return None

    return image_metadata_from_obj(obj, filename)


# ─────────────────────────────────────────────────────────────────────────
# defect_candidates: lifted off ImageMetadata into its own sidecar section
# so that workers that don't need it (every op except
# `identify_camera_fingerprints`) skip the ~1.8 GB deserialize cost.
# Same payload format as before — `{str(pixel_index): float(strength)}` —
# just hoisted from `image_metadata.defect_candidates` to a top-level
# `defect_candidates` section.
# ─────────────────────────────────────────────────────────────────────────

@beartype
def write_defect_candidates(
    filename: Path, defects: dict[int, float] | None,
) -> None:
    """Write per-frame defect candidates to the top-level
    `defect_candidates` sidecar section. None → no-op (caller should
    skip; the absence of the section IS the signal)."""
    if defects is None:
        return
    set_section(
        filename, "defect_candidates",
        {str(k): float(v) for k, v in defects.items()},
    )


@beartype
def read_defect_candidates(filename: Path) -> dict[int, float] | None:
    """Read per-frame defect candidates from the sidecar.

    Tries the new top-level section first, falls back to the legacy nested
    location (`image_metadata.defect_candidates`) for sidecars written by
    pre-migration code. Returns None when neither exists.
    """
    raw = get_section(filename, "defect_candidates")
    if raw is None:
        # Legacy shape: nested inside `image_metadata`. Drop the fallback
        # after the migration script has run and stabilized.
        im = get_section(filename, "image_metadata")
        if isinstance(im, dict):
            raw = im.get("defect_candidates")
    if raw is None:
        return None
    return {int(k): float(v) for k, v in raw.items()}
