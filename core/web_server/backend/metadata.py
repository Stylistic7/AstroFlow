import math
import os
import astropy.io.fits as pyfits
from flask import Blueprint, jsonify, abort
from typing import Any

from context import local_context as context
from stacks.composite_stack import CalibrationStack
from stacks.composite_stack import Stack
from targets.types import Target
from mosaics.types import Mosaic, Viewbox
from mosaics.viewbox import compute_viewbox
from products.registry import MediaType
from products.identity import (
    StackId, CalibrationStackId, TargetId, MosaicChannelId, identity_for_stack,
)
from previews.types import MASTER
from mosaics.trimming import project_to_viewbox
from metadata.sidecar_io import read_image_metadata
from attributes.astrometric_attributes import _spherical_polygon_for_wcs
from decisions.decisions import ObjectType, get_decisions_for_object

metadata_bp = Blueprint("metadata", __name__)


# URL-side object types map to one or more identity object_types
# that share the same `stable_hash`. The PipelineOutputIndex keys
# taps under each identity's `object_type.__name__.lower()`, but
# URLs only carry the cart-level type — so look up across all
# related identity types when resolving a tap's on-disk path.
# Same mapping the dashboard endpoint uses below.
_RELATED_OBJECT_TYPES = {
    "mosaic": ("mosaic", "mergedimage"),
    "target": ("target", "targetchannel"),
    "stack": ("stack", "calibratedstack", "uncorrectedrun"),
}


def tap_path_for(
    pipeline: str, display: str, object_type: str, object_id: str,
    filter: str | None,
) -> str | None:
    """Find a tap point's on-disk path via the PipelineOutputIndex.

    The index is populated at pipeline-trace time — every Node whose
    task carries `_display` (set by `@makes_image(display=...)`)
    gets recorded under (pipeline, identity_type, object_id, display,
    filter). URLs carry the cart-level type (mosaic / target / stack),
    which may not match the identity's type — e.g. a `Merged` tap on
    a mosaic is keyed under `mergedimage`, not `mosaic`. Try each
    related identity type until one hits.
    """
    from pipelines.output_index import tap_path
    for ident_type in _RELATED_OBJECT_TYPES.get(object_type, (object_type,)):
        p = tap_path(pipeline, ident_type, object_id, display, filter)
        if p and os.path.exists(p):
            return p
    return None

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


def _viewbox_quality_for_pipeline(pipeline: str) -> str:
    """Map a pipeline name to the viewbox-quality the pipeline's
    outputs are computed at. `compute_viewbox` needs one of
    ("Subframe", "Master", "DownscaledMaster") — these are the
    on-disk canvas sizes the pipelines produce.

    Heuristic on the pipeline display name: "Subframe" → Subframe,
    contains "Downscaled" → DownscaledMaster, else Master. User
    pipelines that produce different-sized canvases would want to
    declare their viewbox quality explicitly via @pipeline; that's
    a follow-up."""
    if pipeline == "Subframe":
        return "Subframe"
    if "Downscaled" in pipeline:
        return "DownscaledMaster"
    return "Master"


def _get_polygons(identity_cls: type, pipeline: str, *args) -> list[dict[str, Any]]:
    """Compute projected polygons for an object at the pipeline's
    output canvas resolution."""
    if identity_cls is CalibrationStackId:
        return []

    obj = args[0]

    viewbox = compute_viewbox(_viewbox_quality_for_pipeline(pipeline), obj)

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
        # `stack_files` is the canonical full-res master producer; its
        # filename for a specific stack within this mosaic is resolved
        # through the cached pipeline trace. Compute the trace ONCE
        # per request and index it by stack hash — the per-target loop
        # below then does pure in-memory dict lookups (was: one Ray
        # round-trip per target, which scaled badly for big mosaics).
        from dependency_graph.trace_paths import traced_paths_for
        cart_item = (
            type(obj).__name__.lower(),  # "mosaic" / "target"
            obj.stable_hash,
            pipeline,
        )
        master_path_by_stack: dict[str, str] = {
            e["identity"]["stack"]: e["path"]
            for e in traced_paths_for(cart_item)
            if e["op_name"] == "stack_files" and "stack" in e["identity"]
        }
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
            if active_stack:
                master_path = master_path_by_stack.get(active_stack.stable_hash)
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
    """Resolve an object type string + id to (identity_cls, *args).

    The identity class drives the producer/filename-fn registry lookup;
    the args are the real objects the filename_fn / polygon code needs.
    """
    if object_type_str == "stack":
        stack = context.stacks[object_id].get()
        return identity_for_stack(stack), (stack,)
    if object_type_str == "calibration_stack":
        return CalibrationStackId, (context.calibration_stacks[object_id],)
    if object_type_str == "target":
        obj = context.targets[object_id].get()
        return TargetId, (obj, filter) if filter else (obj,)
    if object_type_str == "mosaic":
        obj = context.mosaics[object_id].get()
        return MosaicChannelId, (obj, filter) if filter else (obj,)
    abort(400, f"Unknown object type: {object_type_str}")


@metadata_bp.route("/api/metadata/<object_type>/<object_id>/<path:pipeline_and_display>/<filter>")
@metadata_bp.route("/api/metadata/<object_type>/<object_id>/<path:pipeline_and_display>")
def metadata(object_type: str, object_id: str, pipeline_and_display: str, filter: str | None = None):
    """Polygons + image path for a specific (pipeline, display) tap.

    `pipeline_and_display` is `<pipeline>/<display>` joined into one
    `<path:>` segment so Flask doesn't have to know that pipeline
    display names contain spaces.
    """
    pipeline, _, display = pipeline_and_display.rpartition("/")
    if not pipeline or not display:
        return abort(404)

    obj_type, args = _resolve_object(object_type, object_id, filter)

    path = tap_path_for(pipeline, display, object_type, object_id, filter)
    image_path = path if path and os.path.exists(path) else None

    objects = _get_polygons(obj_type, pipeline, *args)

    return jsonify({
        "display": display,
        "pipeline": pipeline,
        "image_path": image_path,
        "objects": objects,
    })


_OBJECT_TYPE_MAP = {
    "mosaic": MosaicChannelId,
    "target": TargetId,
    "stack": StackId,
    "calibration_stack": CalibrationStackId,
}


def _resolve_entity(object_type: str, object_id: str):
    """Return the in-context entity (Stack, Target, Mosaic, etc.) for
    an (object_type, object_id) pair, or None if not present. The
    dashboard endpoint uses this to know which pipelines accept the
    entity as input + to discover its filters."""
    if object_type == "stack":
        s = (context.stacks or {}).get(object_id)
        return s.get() if s else None
    if object_type == "calibration_stack":
        return (context.calibration_stacks or {}).get(object_id)
    if object_type == "target":
        t = (context.targets or {}).get(object_id)
        return t.get() if t else None
    if object_type == "mosaic":
        m = (context.mosaics or {}).get(object_id)
        return m.get() if m else None
    return None


def _entity_filters(entity, object_type: str) -> list[str]:
    """Filters the dashboard's per-tap columns iterate over."""
    if object_type == "mosaic":
        return _mosaic_filters(entity)
    if object_type == "target":
        stacks_dict = context.stacks or {}
        return sorted({
            stacks_dict[sid].get().filter
            for sid in entity.stack_ids
            if sid in stacks_dict and stacks_dict[sid].get().filter
        })
    if object_type in ("stack", "calibration_stack"):
        return [entity.filter] if getattr(entity, "filter", None) else []
    return []


# In-process memoisation of which (object_type, object_id) pairs
# we've already traced this session. Pipeline code is immutable
# during a run (per user policy), so an entity's tap-point structure
# can't change after the first trace — re-tracing every request was
# wasted work. The set is process-local; backend restart re-traces
# on demand.
_traced_objects: set[tuple[str, str]] = set()


def _ensure_pipelines_traced(object_type: str, object_id: str, entity) -> None:
    """Trace every registered pipeline for `entity` so the
    PipelineOutputIndex carries its tap-point structure. Idempotent
    per backend session: an (object_type, object_id) pair traced
    once will skip the trace on subsequent calls in the same
    process.

    Tracing itself is pure-Python — walks the @pipeline body,
    records Node identities. No Ray dispatch, no compute. We pass
    a `cfg` whose `calibration_stacks` is a plain dict (not a
    `ContextRef`) so the trace path doesn't call `ray.put`; the
    web server runs in `LocalContext` (direct-to-SQLite, no Ray
    init by design) and pipeline bodies use cfg only structurally
    at trace time.
    """
    key = (object_type, object_id)
    if key in _traced_objects:
        return

    from pipelines.registry import registered_pipelines
    from pipelines.output_index import record_existence
    from compute_graph.compute_graph import _record_pipeline_taps
    from decisions.settings import ProcessingConfig, DevelopmentSettings
    from frozendict import frozendict as fd
    from stacks.downscale import TRACE_ONLY as _DOWNSCALE_TRACE_ONLY

    cfg = ProcessingConfig(
        calibration_stacks=context.calibration_stacks or fd(),
        development_settings=DevelopmentSettings(mock_pixinsight=False),
        min_frames_in_stack=3,
        decisions=fd(),
    )

    # `downscale_stack` is a regular function (runs at trace time);
    # in its normal mode it stat()s every frame on disk and writes
    # any missing downscaled FITS — necessary for the background
    # Ray dispatch path, ruinous on the HTTP path (tens of thousands
    # of stat() calls over network storage). TRACE_ONLY skips that
    # work so the endpoint only constructs the Stack metadata
    # needed to enumerate tap points.
    _tok = _DOWNSCALE_TRACE_ONLY.set(True)
    try:
        for (name, processes_class), pipeline_fn in registered_pipelines().items():
            if processes_class is not type(entity):
                continue
            try:
                graph = pipeline_fn(entity, cfg)
            except Exception as e:
                # A pipeline that can't trace this entity (missing
                # context dependencies etc.) just doesn't contribute
                # taps — caller still sees its name + empty taps. Log
                # so we notice systematic trace failures.
                import traceback
                print(f"### pipeline trace failed: {name} for {object_type}/{object_id}: {e}")
                traceback.print_exc()
                continue
            _record_pipeline_taps(
                record_existence, name, object_type, object_id, graph,
            )
    finally:
        _DOWNSCALE_TRACE_ONLY.reset(_tok)

    _traced_objects.add(key)


@metadata_bp.route("/api/pipeline_outputs/<object_type>/<object_id>")
def pipeline_outputs(object_type: str, object_id: str):
    """Per-pipeline tap-point matrix for the object dashboard.

    Shape:

      {
        "filters": ["H", "S"],
        "pipelines": [
          {
            "name": "PixInsight Master",
            "taps": [
              {"display": "Master", "outputs": [
                {"filter": "H", "exists": true, "mtime": 17...},
                {"filter": "S", "exists": false, "mtime": null},
              ]},
              {"display": "Mosaic", "outputs": [...]},
            ],
          },
          ...
        ],
      }

    Each pipeline's `taps` come straight out of the
    PipelineOutputIndex, populated at pipeline-trace time. On
    endpoint hit we lazily trace any registered pipeline whose
    entries are missing for this object — tracing is in-process and
    cheap, so unprocessed objects get their tap-point structure
    surfaced even before any Ray dispatch happens. File existence
    on disk determines `exists`/`mtime` per filter.
    """
    from pipelines.registry import registered_pipelines
    from pipelines.output_index import for_object

    entity = _resolve_entity(object_type, object_id)
    if entity is None:
        return abort(404)

    filters = _entity_filters(entity, object_type)
    iter_filters = filters or [None]

    _ensure_pipelines_traced(object_type, object_id, entity)
    indexed = for_object(object_type, object_id)

    # Aggregate taps from "child" identity types that share the
    # same `object_id`. Example: `merge_images` registers under
    # `MergedImageId` whose `object_type=MergedImage`, but its
    # `mosaic` field's `stable_hash` matches the mosaic the cart
    # is on — so its "Merged" taps belong in the mosaic dashboard
    # alongside the `MosaicChannelId`-keyed "Mosaic" taps. Skip
    # the URL type itself (already in `indexed`).
    for extra_type in _RELATED_OBJECT_TYPES.get(object_type, ()):
        if extra_type == object_type:
            continue
        extra = for_object(extra_type, object_id)
        # `indexed` shape: {pipeline_name: {display: {filter: path_or_pending}}}.
        # Merge per pipeline; display names are already disambiguated
        # by the trace-time recorder so collisions don't happen.
        for pn, displays in extra.items():
            indexed.setdefault(pn, {}).update(displays)

    pipelines = []
    # Every registered (name, class) entry whose class matches the
    # entity gets a row — even pipelines with no traced taps appear
    # in the dashboard (empty `taps` list), so the user sees the full
    # available menu before queuing anything.
    applicable = sorted({
        name for (name, processes_class) in registered_pipelines().keys()
        if processes_class is type(entity)
    })
    for name in applicable:
        traced_taps = indexed.get(name, {})
        taps = []
        # Preserve insertion order of `traced_taps` — that's
        # graph-reach order (each tap is registered the first
        # time the recorder visits its Node during trace), which
        # also numerically orders the per-display suffix series
        # ("Merged" before "Merged 2" before "Merged 10").
        for display in traced_taps.keys():
            tap_paths = traced_taps[display]
            outputs = []
            for f in iter_filters:
                p = tap_paths.get(str(f))
                exists = p is not None and os.path.exists(p)
                outputs.append({
                    "filter": f,
                    "exists": exists,
                    "mtime": os.path.getmtime(p) if exists else None,
                })
            taps.append({"display": display, "outputs": outputs})
        pipelines.append({"name": name, "taps": taps})

    return jsonify({"filters": filters, "pipelines": pipelines})


