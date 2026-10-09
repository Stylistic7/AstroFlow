import math
import os
from pathlib import Path
import astropy.io.fits as pyfits
from flask import Blueprint, jsonify, abort, request
from typing import Any

from context import local_context as context
from stacks.composite_stack import CalibrationStack
from stacks.composite_stack import Stack
from targets.types import Target
from mosaics.types import Mosaic, Viewbox
from mosaics.viewbox import compute_viewbox
from previews.types import MASTER
from mosaics.trimming import project_to_viewbox
from metadata.sidecar_io import read_image_metadata
from attributes.astrometric_attributes import _spherical_polygon_for_wcs
from decisions.decisions import ObjectType, get_decisions_for_object
from web_server.backend._route import finish_route

metadata_bp = Blueprint("metadata", __name__)


def _viewbox_quality_for_pipeline(pipeline: str) -> str:
    """Map a pipeline name to the viewbox-quality the pipeline's
    outputs are computed at: "Subframe" for the Subframe pipeline,
    "Master" (native scale) for every other."""
    if pipeline == "Subframe":
        return "Subframe"
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
        # Per-panel master paths, keyed by the stack's stable_hash. The stacking
        # op is the one tapped `display="Master"` routing to a light stack
        # (calibration masters share the op but route to a calibration stack,
        # so object_type excludes them). Pipeline-agnostic — any stacker
        # backend (WBPP / OSS / future) declares that tap — and keyed on the
        # tap's routing entity, not op-node identity.
        master_path_by_stack: dict[str, str] = {
            e["object_id"]: e["path"]
            for e in traced_paths_for(cart_item)
            if e.get("display") == "Master" and e.get("object_type") == "stack"
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
                            # MRS background noise floor of this master (FITS
                            # NOISE00). Drives the mosaic uniformity overlay:
                            # the depth target is the deepest (lowest-σ) panel
                            # of the channel currently in view, and each
                            # panel's gap to it becomes a "+Nh to match" hint.
                            # None → panel drawn without a depth tint.
                            "noise_estimate": (
                                meta.noise_estimate if meta is not None else None
                            ),
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


@finish_route(metadata_bp, "/api/metadata", hierarchy=("object_type", "object_id"))
def metadata(object_type: str, object_id: str, pipeline: str, finish: str):
    """The polygons the viewer overlays for a finish: the objects under its
    PRIMARY input, at the finish's ROLE (the stack filter a channel finish
    shows: H/S/O/R/G/B/L); None for a composite, which spans every filter.
    Not the input's bound filter: a channel finish reads a band-join tap
    bound as "RGB", and no stack carries that filter, so the master
    footprints (and the depth overlay drawn on them) would vanish."""
    from catalog import resolve_finish
    b = resolve_finish(object_type, object_id, pipeline, finish)
    if b is None:
        return abort(404)
    args = _resolve_object(object_type, object_id,
                           b.role if b.role not in (None, "none") else None)
    return jsonify({"objects": _get_polygons(pipeline, *args)})


def _resolve_entity(object_type: str, object_id: str):
    """The in-context entity for an (object_type, object_id), or None. Delegates
    to the catalog (its canonical home); kept as a module-level name so tests
    can still patch `metadata._resolve_entity`. Filter enumeration also lives in
    the catalog now (`catalog.filters_for`)."""
    from catalog.queries import resolve_entity
    return resolve_entity(object_type, object_id)


@metadata_bp.route("/api/finishes/<object_type>/<object_id>")
def finishes(object_type: str, object_id: str):
    """Every finish each pipeline can show for this object — the view axis
    of the pills and the dashboard.

    Shape:

      {
        "pipelines": [
          {"name": "Hybrid: PI Master + OSS Mosaic",
           "finishes": [
             {"name": "Hubble", "role": null, "status": "live", "mtime": 17...,
              "inputs": [{"tap": "Linear SHO", "filter": "RGB", "status": "live"},
                         {"tap": "Linear RGB Stars", "filter": "RGB", "status": "live"}]},
             {"name": "H", "role": "H", "status": "out_of_date", ...},
           ]},
          ...
        ],
        "default": {"pipeline": "…", "finish": "Hubble"} | null,
      }

    A slice of the catalog: every pipeline whose trace routed taps to this
    object that some finish reads, and each finish whose declared inputs
    those taps satisfy (`catalog.finishes_for` — the same resolver the tile
    server uses, so the list can never name a view the viewer cannot serve).
    A pipeline none of whose taps any finish reads is omitted. `status` is
    the worst over the finish's required inputs. Read-only and snapshot-
    only: a web request never triggers a trace.
    """
    from catalog import finishes_matrix

    matrix = finishes_matrix(object_type, object_id)
    if matrix is None:
        return abort(404)
    return jsonify(matrix)


@metadata_bp.route(
    "/api/pipeline_results/<object_type>/<object_id>/<path:pipeline>"
)
def pipeline_results(object_type: str, object_id: str, pipeline: str):
    """Every op in a pipeline's trace for one object, grouped by function —
    backs the "Discard results" dialog's checkbox list. Unlike
    `finishes` (display taps only), this includes non-display
    intermediates (`place_image`, `bxt_correct`, …) so any op's cached results
    can be discarded.

    Shape: `{"pipeline": ..., "ops": [{op_name, display, is_final, count,
    on_disk, bytes}, ...]}` in trace (≈topological) order."""
    from web_server.backend._discard import group_ops

    if _resolve_entity(object_type, object_id) is None:
        return abort(404)
    return jsonify({
        "pipeline": pipeline,
        "ops": group_ops((object_type, object_id, pipeline)),
    })


@metadata_bp.route(
    "/api/discard_results/<object_type>/<object_id>/<path:pipeline>",
    methods=["POST"],
)
def discard_results(object_type: str, object_id: str, pipeline: str):
    """Delete the cached outputs (files + sidecars + tile pyramid + diskcache)
    of the selected ops for this object's pipeline, invalidate the build-time
    tap snapshot (so the rebuild publishes fresh paths), then drain + restart
    so the discarded ops re-run. Body: `{"op_names": [...]}`."""
    from web_server.backend._discard import discard_ops
    from dependency_graph.trace_paths import invalidate_tap_snapshot
    from state.actor import state
    from web_server.backend.announcer import announce

    if _resolve_entity(object_type, object_id) is None:
        return abort(404)

    op_names = set((request.get_json(silent=True) or {}).get("op_names") or [])
    if not op_names:
        return jsonify({"error": "no op_names given"}), 400

    item = (object_type, object_id, pipeline)
    result = discard_ops(item, op_names)
    invalidate_tap_snapshot(item)
    state.drain_and_restart()
    announce("reload_data")
    return jsonify({"discarded_ops": sorted(op_names), **result})
