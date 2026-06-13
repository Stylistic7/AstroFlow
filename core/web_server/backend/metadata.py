import math
import os
import threading
from pathlib import Path
import astropy.io.fits as pyfits
from flask import Blueprint, jsonify, abort
from typing import Any

from context import local_context as context
from stacks.composite_stack import CalibrationStack
from stacks.composite_stack import Stack
from targets.types import Target
from mosaics.types import Mosaic, Viewbox
from mosaics.viewbox import compute_viewbox
from previews.types import MASTER
from mosaics.trimming import project_to_viewbox
from metadata.sidecar_io import read_image_metadata, _statistics_to_obj
from attributes.astrometric_attributes import _spherical_polygon_for_wcs
from decisions.decisions import ObjectType, get_decisions_for_object
from web_server.backend._route import tap_route

metadata_bp = Blueprint("metadata", __name__)


# The dashboard / cart route URLs use lowercase / snake_case
# cart-types (`mosaic`, `stack`, `calibration_stack`); the
# PipelineOutputIndex registers taps under `type(first_arg).__name__`
# (PascalCase). This dict bridges the two.
_URL_TO_TYPE_NAME = {
    "mosaic": "Mosaic",
    "target": "Target",
    "stack": "Stack",
    "calibration_stack": "CalibrationStack",
}


def _tap_path_unchecked(
    pipeline: str, display: str, object_type: str, object_id: str,
    filter: str | None,
) -> str | None:
    """Look up a tap's on-disk path via the PipelineOutputIndex WITHOUT
    checking whether the file exists.

    The index is populated at pipeline-trace time — every Node whose
    task carries `_display` (set by `@node(display=...)`) gets recorded
    under (pipeline, type_name, object_id, display, filter), where
    `type_name` is `type(first_arg).__name__`. The URL's lowercase
    cart-type translates to that PascalCase name via `_URL_TO_TYPE_NAME`.

    The PipelineOutputIndex uses `"pending"` as a sentinel for taps
    that were traced but whose path hasn't been resolved yet; that
    isn't a real path and gets filtered out here too.

    Use this when you need the path string regardless of whether the
    master is currently on disk — e.g. to derive a `.tiff` sibling for
    tile-serving when ephemeral cleanup may have reclaimed the master.
    For "is this tap currently available" semantics use `tap_path_for`.
    """
    from pipelines.output_index import tap_path, PENDING
    type_name = _URL_TO_TYPE_NAME.get(object_type, object_type)
    p = tap_path(pipeline, type_name, object_id, display, filter)
    if p and p != PENDING:
        return p
    return None


def tap_path_for(
    pipeline: str, display: str, object_type: str, object_id: str,
    filter: str | None,
) -> str | None:
    """Find a tap point's on-disk path via the PipelineOutputIndex,
    returning None if the master no longer exists on disk.

    Existence-gated wrapper around `_tap_path_unchecked` — preserves
    "tap is available" semantics for download / metadata callers.
    """
    p = _tap_path_unchecked(pipeline, display, object_type, object_id, filter)
    return p if p and os.path.exists(p) else None

_MOSAIC_FILTERS_MEMO: dict[str, list[str]] = {}


def _mosaic_filters(mosaic) -> list[str]:
    """All filters that any target in the mosaic has a stack in.

    Memoized on `mosaic.stable_hash`: target_ids and per-target
    stack_ids are structural (don't change as a cart progresses).
    Walking 23 targets × ~3 stacks × ContextRef.get() each costs
    0.25–2 s; the memo collapses that to a dict lookup."""
    cached = _MOSAIC_FILTERS_MEMO.get(mosaic.stable_hash)
    if cached is not None:
        return cached
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
    out = sorted(filters, key=lambda f: (order.index(f) if f in order else len(order), f))
    _MOSAIC_FILTERS_MEMO[mosaic.stable_hash] = out
    return out


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


def _get_polygons(pipeline: str, *args) -> list[dict[str, Any]]:
    """Compute projected polygons for an object at the pipeline's
    output canvas resolution."""
    obj = args[0]
    if isinstance(obj, CalibrationStack):
        return []

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
                    # traced_paths_for returns paths as `str` (cache-friendly
                    # serialization); read_image_metadata expects a Path.
                    meta = read_image_metadata(Path(master_path))
                    # Stacker-reported post-rejection integration time.
                    # `master_integration_s` is populated by the
                    # `read_master_integration_seconds` dispatcher at
                    # ingestion (PixInsight today via XISF Property
                    # `PCL:TotalExposureTime`; Siril / OSS as readers
                    # are added). None when no stacker has touched the
                    # master yet, or the stacker doesn't expose a
                    # survivor total → frontend renders blue-only.
                    # NOTE: deliberately NOT routed through
                    # `effective_integration` here — that would fall
                    # back to AstroFlow's quality-filter prediction
                    # when the stacker number is absent, and the green
                    # wedge specifically means "what the stacker kept,"
                    # not "AstroFlow's best estimate." The single-value
                    # canonical accessor lives elsewhere (suggestion
                    # icon, /api/stacks integration).
                    wbpp_exposure_s = (
                        meta.master_integration_s if meta is not None else None
                    )
                    if meta and meta.plate_solution:
                        from astropy.wcs import WCS
                        wcs = WCS(pyfits.Header(meta.plate_solution.header))
                        output_footprint = _spherical_polygon_for_wcs(wcs)
                        output_poly = project_to_viewbox(output_footprint, viewbox)
                        objects.append({
                            "type": "stack",
                            "polygon_type": "output",
                            "stable_hash": active_stack.stable_hash,
                            "wbpp_exposure_s": wbpp_exposure_s,
                            "target_id": t,
                            "points": list(output_poly.exterior.coords),
                        })

    return objects


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _resolve_object(object_type_str, object_id, filter):
    """Resolve an object type string + id to the real entity args
    that downstream polygon / filename code needs."""
    if object_type_str == "stack":
        return (context.stacks[object_id].get(),)
    if object_type_str == "calibration_stack":
        return (context.calibration_stacks[object_id],)
    if object_type_str == "target":
        return (context.targets[object_id].get(), filter)
    if object_type_str == "mosaic":
        return (context.mosaics[object_id].get(), filter)
    abort(400, f"Unknown object type: {object_type_str}")


@tap_route(metadata_bp, "/api/metadata", hierarchy=("object_type", "object_id"))
def metadata(object_type: str, object_id: str, pipeline: str, display: str, filter: str):
    args = _resolve_object(object_type, object_id, filter)

    path = tap_path_for(pipeline, display, object_type, object_id, filter)
    image_path = path if path and os.path.exists(path) else None

    objects = _get_polygons(pipeline, *args)

    # ImageStatistics for the tap output — drives the browser-side
    # auto-stretch optimizer. Read straight off the sidecar; the
    # `compute_image_metadata` step populated it when the image was
    # written. None when the sidecar is missing (very small images, or
    # taps that don't carry an ImageMetadata).
    statistics = None
    if image_path:
        meta = read_image_metadata(Path(image_path))
        if meta is not None and meta.statistics is not None:
            statistics = _statistics_to_obj(meta.statistics)

    return jsonify({
        "display": display,
        "pipeline": pipeline,
        "image_path": image_path,
        "objects": objects,
        "statistics": statistics,
    })


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


_TRACE_LOCK = threading.Lock()


def _ensure_pipelines_traced(object_type: str, object_id: str, entity) -> None:
    """Trace every registered pipeline that applies to `entity` (once
    per `(pipeline, source_hash, object)`) and commit the tap names
    to `PipelineOutputIndex`.

    The skip-cache is a per-pipeline persistent receipt in `context`:
    `("_pipeline_traced", pname, source_hash, object_type, object_id)
    → True`. It survives worker restart, is shared across all gunicorn
    workers, and invalidates automatically when a pipeline body
    changes (the source hash is in the key). The trace is pure Python
    on a Mosaic-sized pipeline (~600–900 nodes, 10–30 s); the receipt
    avoids paying that more than once per `(source, object)`.

    `_TRACE_LOCK` serialises concurrent threads inside one worker, so
    `--threads 4` doesn't multiply trace cost. Concurrent first-time
    requests across separate workers will still each pay once — that
    only happens before any receipt is committed and is a one-shot
    cost per fresh cache.
    """
    from pipelines.registry import registered_pipelines
    from pipelines.output_index import commit_taps
    from compute_graph.compute_graph import extract_pipeline_taps
    from decisions.settings import ProcessingConfig, DevelopmentSettings
    from frozendict import frozendict as fd
    from hashing.hashing import hash_source_code

    def receipt_key(pname, source_hash):
        return ("_pipeline_traced", pname, source_hash, object_type, object_id)

    applicable = [
        (pname, pfn, hash_source_code(pfn._function))
        for (pname, pc), pfn in registered_pipelines().items()
        if pc is type(entity)
    ]
    to_trace = [
        (pname, pfn, source_hash)
        for pname, pfn, source_hash in applicable
        if not context.get(receipt_key(pname, source_hash))
    ]
    if not to_trace:
        return

    with _TRACE_LOCK:
        cfg = ProcessingConfig(
            calibration_stacks=context.calibration_stacks or fd(),
            development_settings=DevelopmentSettings(mock_pixinsight=False),
            min_frames_in_stack=3,
            decisions=fd(),
        )
        for pname, pfn, source_hash in to_trace:
            # Another thread may have committed while we waited.
            if context.get(receipt_key(pname, source_hash)):
                continue
            try:
                graph = pfn(entity, cfg)
            except Exception as e:
                import traceback
                print(f"### pipeline trace failed: {pname} for {object_type}/{object_id}: {e}", flush=True)
                traceback.print_exc()
                continue
            taps = tuple(tap for _, tap in extract_pipeline_taps(graph))
            commit_taps(pname, object_type, object_id, taps)
            context[receipt_key(pname, source_hash)] = True


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
    # Taps are keyed under `type(first_arg).__name__` (PascalCase); the
    # URL's lowercase cart-type translates via `_URL_TO_TYPE_NAME`. All
    # of an object's taps (including `merge_images`' "Merged") share the
    # one PascalCase bucket, so no cross-bucket aggregation is needed.
    indexed = for_object(_URL_TO_TYPE_NAME.get(object_type, object_type), object_id)

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


