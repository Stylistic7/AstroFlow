"""Serialize/deserialize ImageMetadata to/from sidecar JSON files."""
from __future__ import annotations

import datetime
import numpy as np
from beartype import beartype
from pathlib import Path
from typing import Any

from file_io.sidecar import get_section, set_section, get_file_size, sidecar_path_for
from metadata.classes import ImageMetadata
import plate_solving
import metadata.image_types as image_types


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
    return {
        "content_hash": meta.content_hash,
        "header": _serialize_header(meta.header),
        "width_pixels": meta.width_pixels,
        "height_pixels": meta.height_pixels,
        "channels": meta.channels,
        "dtype": str(meta.dtype),
    }


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

    return ImageMetadata(
        content_hash=obj["content_hash"],
        filename=filename,
        header=_deserialize_header(obj["header"]),
        width_pixels=obj["width_pixels"],
        height_pixels=obj["height_pixels"],
        channels=obj["channels"],
        dtype=np.dtype(obj["dtype"]),
        plate_solution=plate_solution,
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
