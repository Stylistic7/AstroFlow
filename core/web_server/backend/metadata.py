import os
from flask import Blueprint, jsonify, abort
from typing import Any

from context import context
from stacks.composite_stack import CalibrationStack
from stacks.composite_stack import Stack
from stacks.master import MastersDict
from targets.types import Target
from mosaics.types import Mosaic, Viewbox
from mosaics.viewbox import compute_viewbox
from products.registry import get_producer, get_filename_fn, MediaType
from previews.types import QualityLevel
from mosaics.trimming import project_to_viewbox
from decisions.decisions import ObjectType, get_decisions_for_object

metadata_bp = Blueprint("metadata", __name__)

_QUALITY_PRIORITY = [
    QualityLevel.POST_PROCESSED,
    QualityLevel.MASTER,
    QualityLevel.DOWNSCALED_MASTER,
    QualityLevel.SUBFRAME,
]


def _available_qualities(object_type: type, *args) -> list[tuple[QualityLevel, str]]:
    """Return all quality levels that have files on disk, in priority order."""
    result = []
    for level in _QUALITY_PRIORITY:
        filename_fn = get_filename_fn(MediaType.IMAGE, object_type, level)
        if filename_fn is None:
            continue
        filename = filename_fn(*args)
        if os.path.exists(filename):
            result.append((level, os.fspath(filename)))
    return result


def _best_quality(object_type: type, *args) -> tuple[QualityLevel, str] | None:
    """Find the best available quality level for an object."""
    available = _available_qualities(object_type, *args)
    return available[0] if available else None


def _get_polygons(object_type: type, level: QualityLevel, *args) -> list[dict[str, Any]]:
    """Compute projected polygons for an object at a given quality level."""
    if object_type is CalibrationStack:
        return []

    obj = args[0]
    masters = context.masters or MastersDict()
    use_master = level in (QualityLevel.MASTER, QualityLevel.POST_PROCESSED)

    viewbox = compute_viewbox(level, obj)

    def project(object: Any, viewbox: Viewbox) -> dict[str, Any]:
        from stacks.master import Master
        polygon = project_to_viewbox(object.footprint, viewbox)
        obj_type = "stack" if isinstance(object, Master) else type(object).__name__.lower()
        return {
            "type": obj_type,
            "color": "0xff0000",
            "stable_hash": object.stable_hash,
            "points": list(polygon.exterior.coords),
        }

    def resolve_stack(sid):
        """Return the Master if available and appropriate, otherwise the Stack."""
        if use_master and sid in masters:
            return masters[sid]
        return stacks[sid].get()

    stacks = context.stacks
    targets = context.targets

    objects = [project(obj, viewbox)]

    if isinstance(obj, Target):
        objects += [project(resolve_stack(s), viewbox) for s in obj.stack_ids]

    if isinstance(obj, Mosaic):
        selected_filter = args[1] if len(args) > 1 else None
        for t in obj.target_ids:
            target = targets[t].get()
            for sid in target.stack_ids:
                stack = stacks[sid].get()
                if selected_filter and stack.filter != selected_filter:
                    continue
                poly = project(resolve_stack(sid), viewbox)
                poly["target_id"] = t
                poly["has_decisions"] = (
                    len(get_decisions_for_object(ObjectType.STACK, sid)) > 0
                )
                objects.append(poly)

    return objects


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _resolve_object(object_type_str, object_id, filter=None):
    """Resolve an object type string + id to (type, *args)."""
    if object_type_str == "stack":
        return Stack, (context.stacks[object_id].get(),)
    if object_type_str == "calibration_stack":
        return CalibrationStack, (context.calibration_stacks[object_id],)
    if object_type_str == "target":
        obj = context.targets[object_id].get()
        return Target, (obj, filter) if filter else (obj,)
    if object_type_str == "mosaic":
        obj = context.mosaics[object_id].get()
        return Mosaic, (obj, filter) if filter else (obj,)
    abort(400, f"Unknown object type: {object_type_str}")


@metadata_bp.route("/api/validity/<object_type>/<object_id>")
def validity_for_object(object_type: str, object_id: str):
    """Return validity per quality level for a single object."""
    from products import is_object_valid
    result = {}
    for level in QualityLevel:
        result[level.value] = is_object_valid(object_type, object_id, level.value)
    return jsonify(result)


@metadata_bp.route("/api/validity")
def validity_all():
    """Return validity for all objects, keyed by type and ID."""
    from products import is_object_valid
    result = {}
    for stack_id in (context.stacks or {}):
        result[f"stack:{stack_id}"] = {
            level.value: is_object_valid("stack", stack_id, level.value)
            for level in QualityLevel
        }
    for target_id in (context.targets or {}):
        result[f"target:{target_id}"] = {
            level.value: is_object_valid("target", target_id, level.value)
            for level in QualityLevel
        }
    for mosaic_id in (context.mosaics or {}):
        result[f"mosaic:{mosaic_id}"] = {
            level.value: is_object_valid("mosaic", mosaic_id, level.value)
            for level in QualityLevel
        }
    return jsonify(result)


@metadata_bp.route("/api/available_qualities/<object_type>/<object_id>")
@metadata_bp.route("/api/available_qualities/<object_type>/<object_id>/<filter>")
def available_qualities(object_type: str, object_id: str, filter: str = None):
    """Return all available quality levels for an object, best first, with validity."""
    from products import is_object_valid
    obj_type, args = _resolve_object(object_type, object_id, filter)
    available = _available_qualities(obj_type, *args)
    if not available:
        return jsonify({"quality_level": None, "image_path": None, "available": []})

    levels = [
        {
            "quality_level": level.value,
            "image_path": path,
            "valid": is_object_valid(object_type, object_id, level.value),
        }
        for level, path in available
    ]

    # Best non-stale level, falling back to best available.
    best = next((l for l in levels if l["valid"]), levels[0])

    return jsonify({
        "quality_level": best["quality_level"],
        "image_path": best["image_path"],
        "available": levels,
    })


@metadata_bp.route("/api/metadata/<object_type>/<object_id>/<level_or_filter>")
@metadata_bp.route("/api/metadata/<object_type>/<object_id>/<filter>/<level>")
def metadata(object_type: str, object_id: str, level_or_filter: str = None, filter: str = None, level: str = None):
    """Return polygons and image path for an object at a specific quality level.

    Supports both:
      /api/metadata/stack/:id/:level
      /api/metadata/mosaic/:id/:filter/:level
    """
    # Parse the URL segments. For two-segment paths, level_or_filter could be
    # either a filter name or a quality level.
    if level is not None:
        # /api/metadata/:type/:id/:filter/:level
        pass
    elif level_or_filter is not None:
        # /api/metadata/:type/:id/:level_or_filter
        # Try to interpret as a quality level first.
        try:
            level = QualityLevel(level_or_filter)
            level_or_filter = None
        except ValueError:
            filter = level_or_filter
            level = None

    obj_type, args = _resolve_object(object_type, object_id, filter)

    if level is not None:
        quality_level = QualityLevel(level) if isinstance(level, str) else level
    else:
        # Fallback: find best quality.
        result = _best_quality(obj_type, *args)
        if result is None:
            return abort(404)
        quality_level, _ = result

    # Get the image path for the requested level.
    filename_fn = get_filename_fn(MediaType.IMAGE, obj_type, quality_level)
    if filename_fn is None:
        return abort(404)
    filename = filename_fn(*args)
    image_path = os.fspath(filename) if os.path.exists(filename) else None

    objects = _get_polygons(obj_type, quality_level, *args)

    return jsonify({
        "quality_level": quality_level.value,
        "image_path": image_path,
        "objects": objects,
    })
