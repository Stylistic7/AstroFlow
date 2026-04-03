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
from products.registry import get_producer, MediaType
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


def _best_quality(object_type: type, *args) -> tuple[QualityLevel, str] | None:
    """Find the best available quality level for an object. Returns (level, image_path) or None."""
    for level in _QUALITY_PRIORITY:
        fn = get_producer(MediaType.IMAGE, object_type, level)
        if fn is None:
            continue
        filename = fn.filename_fn(*args).resolve_at(context.output_dir)
        if os.path.exists(filename):
            return level, os.fspath(filename)
    return None


def _get_polygons(object_type: type, level: QualityLevel, *args) -> list[dict[str, Any]]:
    """Compute projected polygons for an object at a given quality level."""
    if object_type is CalibrationStack:
        return []

    obj = args[0]
    masters = context.masters or MastersDict()
    use_master = level in (QualityLevel.MASTER, QualityLevel.POST_PROCESSED)

    viewbox = compute_viewbox(level, obj)

    def project(object: Any, viewbox: Viewbox) -> dict[str, Any]:
        polygon = project_to_viewbox(object.footprint, viewbox)
        return {
            "type": type(object).__name__.lower(),
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


@metadata_bp.route("/api/best_quality/<object_type>/<object_id>")
@metadata_bp.route("/api/best_quality/<object_type>/<object_id>/<filter>")
def best_quality(object_type: str, object_id: str, filter: str = None):
    """Return the best available quality level for an object."""
    obj_type, args = _resolve_object(object_type, object_id, filter)
    result = _best_quality(obj_type, *args)
    if result is None:
        return abort(404)
    level, image_path = result
    return jsonify({"quality_level": level.value, "image_path": image_path})


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
    fn = get_producer(MediaType.IMAGE, obj_type, quality_level)
    if fn is None:
        return abort(404)
    filename = fn.filename_fn(*args).resolve_at(context.output_dir)
    image_path = os.fspath(filename) if os.path.exists(filename) else None

    objects = _get_polygons(obj_type, quality_level, *args)

    return jsonify({
        "quality_level": quality_level.value,
        "image_path": image_path,
        "objects": objects,
    })
