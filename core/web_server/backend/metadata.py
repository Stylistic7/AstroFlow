import math
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
        from decisions.compute_graph import _is_quarantined

        target_numbers = obj.panel_numbers()

        # Real-time rise time per panel for display in the viewer's
        # panel labels. Computed against the observer's current LST so
        # the wall-clock time the user sees is "next rise from now."
        from astropy import units as u  # type: ignore
        from astropy.coordinates import EarthLocation  # type: ignore
        from astropy.time import Time  # type: ignore
        from mosaics.rise_time import next_rise_utc
        import datetime
        lat, lon, elev = context.settings.observer_location
        observer = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=elev * u.m)
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        reference_astropy = Time(now_utc)
        lst_at_ref = float(
            reference_astropy.sidereal_time("apparent", longitude=observer.lon).hour
        )
        observer_lat_rad = math.radians(lat)

        for t in obj.target_ids:
            target = targets[t].get()
            # The stack at the selected filter (if any) drives the
            # overlays (decision dot, exposure clock) and the green output
            # rectangle. Targets are filter-independent, but each filter
            # view shows the data state for its own stack.
            active_stack = next(
                (stacks[sid].get() for sid in target.stack_ids
                 if (not selected_filter or stacks[sid].get().filter == selected_filter)
                 and not _is_quarantined(overrides, ObjectType.STACK, sid)),
                None,
            )

            # Input polygon: always at the Target level so the panel
            # outline shows up on every filter view, even if this target
            # has no stack at the selected filter.
            input_poly = project(target, viewbox)
            input_poly["polygon_type"] = "input"
            input_poly["target_id"] = t
            input_poly["active_stack_hash"] = active_stack.stable_hash if active_stack else None
            input_poly["has_decisions"] = (
                active_stack is not None
                and len(get_decisions_for_object(ObjectType.STACK, active_stack.stable_hash)) > 0
            )
            input_poly["ra_deg"] = float(target.center[0])
            input_poly["dec_deg"] = float(target.center[1])
            input_poly["rotation_deg"] = math.degrees(float(target.rotation_angle))
            rise = next_rise_utc(
                ra_deg=float(target.center[0]),
                dec_deg=float(target.center[1]),
                observer_lat_rad=observer_lat_rad,
                lst_at_ref_hours=lst_at_ref,
                reference_utc=now_utc,
                altitude_threshold_deg=context.settings.usable_altitude_deg,
            )
            input_poly["rise_time_utc"] = rise.isoformat() if rise else None
            if t in target_numbers:
                input_poly["panel_number"] = target_numbers[t]
            objects.append(input_poly)

            # Output polygon: the master's actual WCS rectangle. Drawn
            # only when both the master image AND its computed metadata
            # exist — green always means "this is exactly where the
            # stacked image sits on disk."
            if active_stack and master_fn:
                master_path = master_fn(active_stack)
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
                            "stable_hash": active_stack.stable_hash,
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


_OBJECT_TYPE_MAP = {
    "mosaic": Mosaic,
    "target": Target,
    "stack": Stack,
    "calibration_stack": CalibrationStack,
}


def _resolve_versioned_object(object_type: str, object_id: str):
    """Return (obj, filters, filename_candidates) for /api/object_versions.

    `filename_candidates(impl, level, f) -> list[Path]` lists every path
    that should count as "this version exists" — the canonical producer
    output plus any object-specific fallbacks (e.g. mosaic in-progress
    merge tiffs). The caller takes the first one that's on disk.
    """
    from pathlib import Path

    obj_cls = _OBJECT_TYPE_MAP.get(object_type)
    if obj_cls is None:
        return None

    if object_type == "mosaic":
        mosaics = context.mosaics or {}
        if object_id not in mosaics:
            return None
        mosaic = mosaics[object_id].get()
        filters = _mosaic_filters(mosaic)
        output_dir = context.output_dir

        def filename_candidates(impl, level, f):
            candidates = []
            fn = get_filename_fn(MediaType.IMAGE, Mosaic, level, impl)
            if fn is not None:
                candidates.append(fn(mosaic, f))
            # merge_images writes a per-(mosaic, filter, level) progress
            # tiff that's overwritten each step; surface it so the
            # dashboard lights up before the final store runs.
            if output_dir is not None:
                candidates.append(
                    Path(output_dir) / "Mosaic" / str(level)
                    / f"progress_{mosaic.stable_hash}_{f}.tiff"
                )
            return candidates

        return mosaic, filters, filename_candidates

    if object_type == "target":
        targets = context.targets or {}
        if object_id not in targets:
            return None
        target = targets[object_id].get()
        stacks_dict = context.stacks or {}
        filters = sorted({
            stacks_dict[sid].get().filter
            for sid in target.stack_ids
            if sid in stacks_dict and stacks_dict[sid].get().filter
        })

        def filename_candidates(impl, level, f):
            fn = get_filename_fn(MediaType.IMAGE, Target, level, impl)
            return [fn(target, f)] if fn else []

        return target, filters, filename_candidates

    if object_type == "stack":
        stacks_dict = context.stacks or {}
        if object_id not in stacks_dict:
            return None
        stack = stacks_dict[object_id].get()
        filters = [stack.filter] if stack.filter else []

        def filename_candidates(impl, level, f):
            fn = get_filename_fn(MediaType.IMAGE, type(stack), level, impl)
            return [fn(stack)] if fn else []

        return stack, filters, filename_candidates

    if object_type == "calibration_stack":
        cal_stacks = context.calibration_stacks or {}
        if object_id not in cal_stacks:
            return None
        stack = cal_stacks[object_id]
        filters = [stack.filter] if stack.filter else []

        def filename_candidates(impl, level, f):
            fn = get_filename_fn(MediaType.IMAGE, type(stack), level, impl)
            return [fn(stack)] if fn else []

        return stack, filters, filename_candidates

    return None


@metadata_bp.route("/api/object_versions/<object_type>/<object_id>")
def object_versions(object_type: str, object_id: str):
    """Return the (implementation × quality × filter) matrix for any
    object type (mosaic / target / stack / calibration_stack).

    Shape matches /api/mosaic_versions so the frontend can consume one
    endpoint for every viewer:

      {
        "filters": ["H", "S"],   # one entry for stacks; possibly empty for cal stacks
        "versions": [
          {"implementation": "PixInsight", "quality": "Master",
           "filter": "H", "exists": true, "mtime": 1234567890.0},
          ...
        ],
      }
    """
    resolved = _resolve_versioned_object(object_type, object_id)
    if resolved is None:
        return abort(404)
    _obj, filters, filename_candidates = resolved

    # Filterless objects (e.g., bias cal stacks) still need one entry per
    # (impl, level) — model that as a single None-filter row.
    iter_filters = filters or [None]

    versions = []
    for impl in Implementation:
        for level in QUALITY_PRIORITY:
            for f in iter_filters:
                candidates = filename_candidates(impl, level, f)
                hit = next(
                    (p for p in candidates if p is not None and os.path.exists(p)),
                    None,
                )
                versions.append({
                    "implementation": impl.value,
                    "quality": level.value,
                    "filter": f,
                    "exists": hit is not None,
                    "mtime": os.path.getmtime(hit) if hit is not None else None,
                })

    return jsonify({"filters": filters, "versions": versions})


