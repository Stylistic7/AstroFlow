from flask import request, jsonify, Blueprint, Response
import os
from pathlib import Path
import subprocess

from context import local_context as context
from file_io.generic import read_header
from file_io.find_image_files import find_image_files
from file_io.sidecar import get_section
from metadata.image_types import is_light_frame
from web_server.backend.announcer import announce
from web_server.backend._route import tap_route
from state.actor import state

files_bp = Blueprint("files", __name__)


@files_bp.route("/api/list_dir")
def list_dir():
    path = request.args.get("path", default=os.path.expanduser("~"))
    try:
        entries = os.scandir(path)
        files = []
        for entry in entries:
            full_path = os.path.join(path, entry.name)
            file_info = {
                "id": full_path,
                "name": entry.name,
                "isDir": entry.is_dir(),
            }

            if entry.is_file() and entry.name.lower().endswith(
                (".fits", ".fit", ".xisf")
            ):
                try:
                    header = read_header(Path(full_path))
                    file_info.update(
                        {
                            "camera": header.get("INSTRUME"),
                            "exposure": header.get("EXPOSURE"),
                            # Add more fields here if desired
                            "header": header,  # Optional: include full header
                        }
                    )
                except Exception as e:
                    file_info["header_error"] = str(e)

            files.append(file_info)

        return jsonify({"files": files, "path": path})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@files_bp.route("/api/select_file", methods=["POST"])
def select_file():
    # TODO: Reimplement using decision overrides instead of removed StackSettings.dependencies
    return jsonify({"error": "Not yet reimplemented with decision overrides"}), 501


@files_bp.route("/api/stack/<stack_id>/stacking_report", methods=["GET"])
def stacking_report(stack_id):
    """Stacker-agnostic summary read off the master file on disk.

    Resolves the master path from the cached pipeline trace + disk
    (tap-keyed) and parses the file's headers + XISF properties into a
    uniform shape (StackingReport). Returns `null` when the tap point
    hasn't produced an artifact yet.

    Query params (defaults come from the catalog — the preferred pipeline that
    actually produced a tap for this stack, and that type's default display):
      pipeline (default: `catalog.default_pipeline_for`)
      display  (default: `catalog.default_display_for`)
    """
    from stacks.stacking_report import read_stacking_report
    from web_server.backend.metadata import available_tap_path
    from catalog import default_pipeline_for, default_display_for

    if stack_id not in (context.stacks or {}):
        return jsonify({"error": "stack not found"}), 404

    pipeline = request.args.get("pipeline") or default_pipeline_for("stack", stack_id)
    display = request.args.get("display") or default_display_for("stack")

    master_path = available_tap_path(pipeline, display, "stack", stack_id, None)
    if not master_path or not Path(master_path).exists():
        return jsonify(None)

    try:
        return jsonify(read_stacking_report(master_path).to_json())
    except Exception as e:
        return jsonify({"error": f"failed to read report: {e}"}), 500


@files_bp.route("/api/concerns/<object_type>/<object_id>", methods=["GET"])
def concerns_for_object(object_type, object_id):
    """Concerns reported by recent producer runs for this object.

    @capture_logs writes one entry per object at key
    `('concerns', object_type, identity_key)` with value
    `{op_name: [concerns...]}`. Direct O(1) lookup — no cache scan.
    Mosaic concerns also bubble up the concerns from constituent
    stacks at the requested filter.

    Query params:
      filter — required for object_type=mosaic
    """
    from context import get_cache

    cache = get_cache()
    filter_arg = request.args.get("filter", "")

    def concerns_for_key(object_type_name: str, identity_key: str) -> list[dict]:
        """Direct lookup at `('concerns', object_type, identity_key)`.
        Value is `{op_name: [concerns...]}`; we flatten across ops."""
        entry = cache.get(("concerns", object_type_name, identity_key), {})
        if not isinstance(entry, dict):
            return []
        out: list[dict] = []
        for op_concerns in entry.values():
            out.extend(op_concerns)
        return out

    if object_type == "stack":
        if object_id not in (context.stacks or {}):
            return jsonify({"error": "stack not found"}), 404
        stack = context.stacks[object_id].get()
        return jsonify(concerns_for_key(
            type(stack).__name__, stack.stable_hash,
        ))

    if object_type == "mosaic":
        if object_id not in (context.mosaics or {}):
            return jsonify({"error": "mosaic not found"}), 404
        mosaic = context.mosaics[object_id].get()
        direct = concerns_for_key(
            "Mosaic",
            f"{mosaic.stable_hash}_{filter_arg}",
        )
        bubbled = []
        for stack in _constituent_stacks(mosaic, filter_arg):
            for c in concerns_for_key(
                type(stack).__name__, stack.stable_hash,
            ):
                bubbled.append({**c, "from_stack": stack.stable_hash})
        return jsonify(direct + bubbled)

    return jsonify({"error": f"object_type {object_type!r} not supported"}), 400


def _constituent_stacks(mosaic, filter):
    stacks = context.stacks or {}
    targets = context.targets or {}
    out = []
    for target_id in mosaic.target_ids:
        target_ref = targets.get(target_id)
        if target_ref is None:
            continue
        for stack_id in target_ref.get().stack_ids:
            stack_ref = stacks.get(stack_id)
            if stack_ref is None:
                continue
            stack = stack_ref.get()
            if stack.filter == filter:
                out.append(stack)
    return out


@files_bp.route("/api/invalidate_stack/<stack_id>", methods=["POST"])
def invalidate_stack(stack_id):
    """Drop the stack_files cache entry for one stack and rerun the graph.

    Targets the same cache key that the main compute-graph builder uses
    (via stack_files.invalidate which honors the cache_key_narrow hook),
    then drains + restarts so the next graph rebuild picks up the
    cache miss and re-stacks just this one.
    """
    from decisions.settings import ProcessingConfig
    from stacks.stacking import stack_files

    if stack_id not in (context.stacks or {}):
        return jsonify({"error": "stack not found"}), 404

    stack = context.stacks[stack_id].get()
    settings = context.settings
    config = ProcessingConfig(
        # cache_key_narrow strips calibration_stacks; value here is irrelevant.
        calibration_stacks=None,
        development_settings=settings.development_settings,
        min_frames_in_stack=settings.clustering_settings.min_frames_in_stack,
        decisions=settings.decision_overrides,
    )
    deleted = stack_files.invalidate(stack, config)
    if deleted:
        state.drain_and_restart()
    return jsonify({"stack_id": stack_id, "deleted": deleted})


@files_bp.route("/api/open_stack_master/<stack_id>", defaults={"pipeline": None})
@files_bp.route("/api/open_stack_master/<stack_id>/<path:pipeline>")
def open_stack_master(stack_id, pipeline):
    """Open the stack's Master file in PixInsight.

    Resolves the on-disk master through the SAME tap system the stack page
    uses to DISPLAY it (`available_tap_path`, existence-gated) — so open stays in
    lockstep with what's shown, and works for any stacker backend (WBPP
    `stack_files`, OSS `stack_files_oss`, downscaled, ...). The old
    `traced_path_for(..., "stack_files", stack=...)` path was broken: the
    trace's stack_files entries carry an empty `identity`, so it never
    matched. `pipeline` (URL-encoded) selects which pipeline's master; the
    caller passes the pipeline it is currently viewing, or omits it to open
    the catalog's default (`catalog.default_pipeline_for`).
    """
    from web_server.backend.metadata import available_tap_path
    from catalog import default_pipeline_for
    if stack_id not in (context.stacks or {}):
        return Response(status=404)
    if pipeline is None:
        pipeline = default_pipeline_for("stack", stack_id)
    stack = context.stacks[stack_id].get()
    filename = available_tap_path(
        pipeline, "Master", "stack", stack_id, getattr(stack, "filter", None),
    )
    if filename is None:
        return Response(status=404)
    print(f"### opening filename {filename}", flush=True)
    subprocess.Popen(
        [
            "/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight",
            str(filename),
        ]
    )
    return Response(status=200)


def _find_unsolved_light_frames():
    """Scan input dirs for light frames whose plate-solving failed."""
    results = []
    for input_dir in context.input_dirs or []:
        for fits_path in find_image_files(str(input_dir)):
            sidecar = get_section(Path(fits_path), "plate_solution")
            if sidecar is None:
                continue
            if sidecar.get("data") is not None:
                continue
            try:
                header = read_header(Path(fits_path))
            except Exception:
                continue
            if not is_light_frame(header):
                continue
            results.append({
                "filename": str(fits_path),
                "size_bytes": os.path.getsize(fits_path),
                "date_obs": header.get("DATE-OBS"),
                "filter": header.get("FILTER"),
                "exposure": header.get("EXPTIME") or header.get("EXPOSURE"),
                "target": header.get("OBJECT"),
            })
    return results


@files_bp.route("/api/unsolved_frames")
def unsolved_frames():
    frames = _find_unsolved_light_frames()
    return jsonify({
        "count": len(frames),
        "total_bytes": sum(f["size_bytes"] for f in frames),
        "frames": frames,
    })


@files_bp.route("/api/trash_unsolved", methods=["POST"])
def trash_unsolved():
    from send2trash import send2trash

    frames = _find_unsolved_light_frames()
    trashed = 0
    for f in frames:
        path = Path(f["filename"])
        if path.exists():
            send2trash(str(path))
            sidecar = Path(str(path) + ".json")
            if sidecar.exists():
                send2trash(str(sidecar))
            trashed += 1

    announce("reload_data")
    state.drain_and_restart()
    return jsonify({"trashed": trashed})


@files_bp.route("/api/open_image/<path:filename>")
def open_image(filename):
    filename = "/" + filename
    print(f"### opening filename {filename}")
    subprocess.Popen(
        [
            "/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight",
            str(filename),
        ]
    )

    return Response(status=200)


@tap_route(files_bp, "/api/open_mosaic_master", hierarchy=("mosaic_id",))
def open_mosaic_master(mosaic_id, pipeline, display, filter):
    """Open a specific (pipeline, tap) of a mosaic+filter in PixInsight.
    Path resolution goes through the cached pipeline trace + disk
    (`available_tap_path`)."""
    from web_server.backend.metadata import available_tap_path

    mosaics = context.mosaics or {}
    if mosaic_id not in mosaics:
        return jsonify({"error": "mosaic not found"}), 404

    path = available_tap_path(pipeline, display, "mosaic", mosaic_id, filter)
    if path is None:
        return jsonify({"error": "no image available to open"}), 404

    print(f"### opening {path} in PixInsight")
    subprocess.Popen([
        "/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight",
        str(path),
    ])
    return jsonify({"opened": str(path)})
