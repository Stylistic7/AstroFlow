import os
import astropy.io.fits as pyfits
from flask import Blueprint, jsonify, abort
from typing import Any

from context import local_context as context
from previews.types import Implementation
from stacks.composite_stack import CalibrationStack
from stacks.composite_stack import Stack
from targets.types import Target
from mosaics.types import Mosaic, Viewbox
from mosaics.viewbox import compute_viewbox
from products.registry import get_producer, get_filename_fn, MediaType
from previews.types import QualityLevel, QUALITY_PRIORITY
from mosaics.trimming import project_to_viewbox
from metadata.sidecar_io import read_image_metadata
from attributes.astrometric_attributes import _spherical_polygon_for_wcs
from decisions.decisions import ObjectType, get_decisions_for_object

metadata_bp = Blueprint("metadata", __name__)

def _available_qualities(object_type: type, implementation: Implementation, *args) -> list[tuple[QualityLevel, str]]:
    """Return all quality levels that have files on disk, in priority order."""
    result = []
    for level in QUALITY_PRIORITY:
        filename_fn = get_filename_fn(MediaType.IMAGE, object_type, level, implementation)
        if filename_fn is None:
            continue
        filename = filename_fn(*args)
        if filename is None:
            continue
        if os.path.exists(filename):
            result.append((level, os.fspath(filename)))
    return result


def _best_quality(object_type: type, implementation: Implementation, *args) -> tuple[QualityLevel, str] | None:
    """Find the best available quality level for an object."""
    available = _available_qualities(object_type, implementation, *args)
    return available[0] if available else None


def _mosaic_filters(mosaic) -> list[str]:
    """All filters that any target in the mosaic has a stack in."""
    stacks = context.stacks or {}
    targets = context.targets or {}
    filters = set()
    for target_id in mosaic.target_ids:
        target = targets[target_id].get()
        for stack_id in target.stack_ids:
            stack_ref = stacks.get(stack_id)
            if stack_ref is None:
                continue
            filters.add(stack_ref.get().filter)
    order = ["L", "H", "S", "O", "R", "G", "B"]
    return sorted(filters, key=lambda f: (order.index(f) if f in order else len(order), f))


def _get_polygons(object_type: type, level: QualityLevel, *args) -> list[dict[str, Any]]:
    """Compute projected polygons for an object at a given quality level."""
    if object_type is CalibrationStack:
        return []

    obj = args[0]

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
        return stacks[sid].get()

    stacks = context.stacks
    targets = context.targets

    objects = [project(obj, viewbox)]

    if isinstance(obj, Target):
        objects += [project(resolve_stack(s), viewbox) for s in obj.stack_ids]

    if isinstance(obj, Mosaic):
        selected_filter = args[1] if len(args) > 1 else None
        # Output polygon comes from the stacked master's WCS regardless of
        # the displayed level. At Subframe level the per-stack filename is a
        # raw frame, not a Path, and read_image_metadata would reject it.
        master_fn = get_filename_fn(MediaType.IMAGE, Stack, QualityLevel.MASTER)
        overrides = context.settings.decision_overrides
        for t in obj.target_ids:
            target = targets[t].get()
            for sid in target.stack_ids:
                stack = stacks[sid].get()
                if selected_filter and stack.filter != selected_filter:
                    continue
                # Skip quarantined stacks.
                from decisions.compute_graph import _is_quarantined
                if _is_quarantined(overrides, ObjectType.STACK, sid):
                    continue

                # Input polygon: the intersection of Run footprints.
                input_poly = project(resolve_stack(sid), viewbox)
                input_poly["polygon_type"] = "input"
                input_poly["target_id"] = t
                input_poly["has_decisions"] = (
                    len(get_decisions_for_object(ObjectType.STACK, sid)) > 0
                )
                objects.append(input_poly)

                # Output polygon: the WCS rectangle of the stacked master.
                if master_fn:
                    master_path = master_fn(stack)
                    if master_path and os.path.exists(master_path):
                        meta = read_image_metadata(master_path)
                        if meta and meta.plate_solution:
                            from astropy.wcs import WCS
                            wcs = WCS(pyfits.Header(meta.plate_solution.header))
                            output_footprint = _spherical_polygon_for_wcs(wcs)
                            output_poly = project_to_viewbox(output_footprint, viewbox)
                            objects.append({
                                "type": "stack",
                                "polygon_type": "output",
                                "stable_hash": stack.stable_hash,
                                "target_id": t,
                                "points": list(output_poly.exterior.coords),
                            })

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
    """Return validity for all objects, keyed by type and ID.

    Batch-fetches the validity dict and ui_targets/ui_mosaics in minimal
    Ray calls, then computes hierarchy locally.
    """
    validity = context.get("validity", {})
    ui_targets = context.ui_targets or {}
    ui_mosaics = context.ui_mosaics or {}
    stacks = context.stacks or {}

    def stack_valid(stack_id, level):
        return validity.get(("stack", stack_id, level)) is True

    def target_valid(target_id, level):
        target = ui_targets.get(target_id)
        if not target or "stack_ids" not in target:
            return False
        return all(stack_valid(sid, level) for sid in target["stack_ids"])

    def mosaic_valid(mosaic_id, level):
        mosaic = ui_mosaics.get(mosaic_id)
        if not mosaic or "target_ids" not in mosaic:
            return False
        return all(target_valid(tid, level) for tid in mosaic["target_ids"])

    result = {}
    for stack_id in stacks:
        result[f"stack:{stack_id}"] = {
            level.value: stack_valid(stack_id, level.value)
            for level in QualityLevel
        }
    for target_id in ui_targets:
        result[f"target:{target_id}"] = {
            level.value: target_valid(target_id, level.value)
            for level in QualityLevel
        }
    for mosaic_id in ui_mosaics:
        result[f"mosaic:{mosaic_id}"] = {
            level.value: mosaic_valid(mosaic_id, level.value)
            for level in QualityLevel
        }
    return jsonify(result)


@metadata_bp.route("/api/available_qualities/<object_type>/<object_id>")
@metadata_bp.route("/api/available_qualities/<object_type>/<object_id>/<filter>")
def available_qualities(object_type: str, object_id: str, filter: str = None):
    """Return all quality levels with existence and validity status."""
    from products import is_object_valid
    implementation = context.settings.implementation
    obj_type, args = _resolve_object(object_type, object_id, filter)
    on_disk = dict(_available_qualities(obj_type, implementation, *args))  # level -> path

    levels = []
    for level in QUALITY_PRIORITY:
        path = on_disk.get(level)
        levels.append({
            "quality_level": level.value,
            "image_path": os.fspath(path) if path else None,
            "exists": path is not None,
            "valid": is_object_valid(object_type, object_id, level.value) if path else False,
        })

    # Best: first existing+valid, then first existing, then first overall.
    best = (
        next((l for l in levels if l["exists"] and l["valid"]), None)
        or next((l for l in levels if l["exists"]), None)
        or levels[0]
    )

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

    implementation = context.settings.implementation
    obj_type, args = _resolve_object(object_type, object_id, filter)

    if level is not None:
        quality_level = QualityLevel(level) if isinstance(level, str) else level
    else:
        # Fallback: find best quality.
        result = _best_quality(obj_type, implementation, *args)
        if result is None:
            return abort(404)
        quality_level, _ = result

    # Get the image path for the requested level.
    filename_fn = get_filename_fn(MediaType.IMAGE, obj_type, quality_level, implementation)
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


@metadata_bp.route("/api/mosaic_versions/<mosaic_id>")
def mosaic_versions(mosaic_id: str):
    """Return the (implementation × quality × filter) matrix for a mosaic.

    Shape:
      {
        "filters": ["H", "S", "O"],
        "versions": [
          {"implementation": "PixInsight", "quality": "Master",
           "filter": "H", "exists": true, "mtime": 1234567890.0},
          ...
        ],
      }
    """
    mosaics = context.mosaics or {}
    if mosaic_id not in mosaics:
        return abort(404)
    mosaic = mosaics[mosaic_id].get()
    filters = _mosaic_filters(mosaic)

    # Progress tile pyramids (written by merge_images) signal in-progress
    # merges before the final store_mosaic_color_channel runs. Surface them
    # so the dashboard shows thumbnails as merges stream in.
    from pathlib import Path
    output_dir = context.output_dir

    def progress_path(level, f):
        if output_dir is None:
            return None
        return Path(output_dir) / "Mosaic" / str(level) / f"progress_{mosaic.stable_hash}_{f}.tiff"

    versions = []
    for impl in Implementation:
        for level in QUALITY_PRIORITY:
            filename_fn = get_filename_fn(MediaType.IMAGE, Mosaic, level, impl)
            if filename_fn is None:
                continue
            for f in filters:
                path = filename_fn(mosaic, f)
                final_exists = path is not None and os.path.exists(path)
                prog = progress_path(level, f)
                progress_exists = prog is not None and prog.exists()
                exists = final_exists or progress_exists
                if final_exists:
                    mtime = os.path.getmtime(path)
                elif progress_exists:
                    mtime = os.path.getmtime(prog)
                else:
                    mtime = None
                versions.append({
                    "implementation": impl.value,
                    "quality": level.value,
                    "filter": f,
                    "exists": exists,
                    "mtime": mtime,
                })

    return jsonify({
        "filters": filters,
        "versions": versions,
    })
