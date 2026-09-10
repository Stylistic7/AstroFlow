from flask import request, jsonify, Blueprint, Response
import os
from pathlib import Path
import subprocess

from context import local_context as context
from file_io.generic import read_header
from file_io.find_image_files import is_external_master_name
from file_io.sidecar import get_section
from metadata.image_types import is_light_frame
from web_server.backend.announcer import announce
from web_server.backend._route import tap_route
from state.actor import state

files_bp = Blueprint("files", __name__)


@files_bp.route("/api/list_dir")
def list_dir():
    """Browse for input frames. Restricted to the configured data roots
    (plus the user's home as the browse ENTRY point for picking new input
    dirs) — previously this walked any path on the machine, unauthenticated
    on a 0.0.0.0 bind."""
    path = request.args.get("path", default=os.path.expanduser("~"))
    resolved = os.path.realpath(path)
    allowed_roots = [
        os.path.realpath(str(d))
        for d in [
            *(context.input_dirs or []), context.output_dir,
            os.path.expanduser("~"), "/Volumes",
        ]
        if d
    ]
    if not any(resolved == root or resolved.startswith(root + os.sep)
               for root in allowed_roots):
        return jsonify({"error": "path outside allowed roots"}), 403
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
    """Light frames whose plate solve was ATTEMPTED and FAILED.

    Narrowed against the in-memory library FIRST. The old scan walked every
    input dir and called `get_section` per file, and `get_section` json-parses
    the WHOLE sidecar — `defect_candidates` and all — to reach one small
    section: ~10 GB of parsing across 32k frames, plus a FITS open apiece for
    the header, on every dialog open. The library already holds each frame's
    header and plate solution, so the solved majority is discarded for free.

    A sidecar read survives, for the handful that reach it, because the
    library cannot tell the two failure modes apart: `plate_solution is None`
    means EITHER the solve failed (a tombstone section, `data: null`) or it
    was never attempted (no section at all). Only the first may be offered
    for deletion.
    """
    ref = context["metadata_ref"]
    if ref is None:
        return []

    results = []
    for meta in ref.get().values():
        if meta.plate_solution is not None:
            continue
        if is_external_master_name(Path(meta.filename).name):
            continue      # someone's finished master, never ours to trash
        if not is_light_frame(meta.header):
            continue
        section = get_section(Path(meta.filename), "plate_solution")
        if section is None:
            continue      # never attempted
        if section.get("data") is not None:
            continue      # solved since the library was built
        try:
            size_bytes = os.path.getsize(meta.filename)
        except OSError:
            continue      # gone from disk, or its volume is unplugged
        results.append({
            "filename": str(meta.filename),
            "size_bytes": size_bytes,
            "date_obs": meta.header.get("DATE-OBS"),
            "filter": meta.header.get("FILTER"),
            "exposure": (meta.header.get("EXPTIME")
                         or meta.header.get("EXPOSURE")),
            "target": meta.header.get("OBJECT"),
        })
    return sorted(results, key=lambda f: f["filename"])


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


@files_bp.route("/api/open_image/<path:filename>", methods=["POST"])
def open_image(filename):
    """Launch PixInsight on a file. POST-only: as a GET, any web page the
    user visited could trigger a cross-origin process launch with an
    arbitrary path (CORS blocks reading responses, not sending requests).
    The path must also live under a configured data root — this endpoint
    is an opener for AstroFlow's own files, not a general launcher."""
    filename = "/" + filename
    resolved = os.path.realpath(filename)
    roots = [
        os.path.realpath(str(d))
        for d in [*(context.input_dirs or []), context.output_dir]
        if d
    ]
    if not any(resolved.startswith(root + os.sep) or resolved == root
               for root in roots):
        return jsonify({"error": "path outside AstroFlow data roots"}), 403
    # PixInsight cannot read a TILED tiff and fails SILENTLY on one
    # (ImageWindow.open returns an empty array, no exception) — so a pyramid
    # handed over directly looks like a no-op. Reconstruct page 0 as a stripped
    # TIFF first; a non-tiled source passes through at zero cost. See
    # file_io/planar.py for the measurements.
    from file_io.planar import reconstruct_planar_tiff
    from pix.runner import open_in_gui
    to_open = Path(reconstruct_planar_tiff(resolved))
    if os.fspath(to_open) != resolved:
        print(f"### open_image: reconstructed planar TIFF {to_open}")
    open_in_gui({to_open.stem: to_open})

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

    # Same tiled-TIFF reconstruction as `open_image` — the scene-format taps
    # this most often resolves to are exactly the ones stored ONLY as pyramids.
    from file_io.planar import reconstruct_planar_tiff
    from pix.runner import open_in_gui
    to_open = Path(reconstruct_planar_tiff(path))
    open_in_gui({filter or to_open.stem: to_open})
    return jsonify({"opened": str(to_open)})


@files_bp.route("/api/open_target_masters/<target_id>/<path:pipeline>",
                methods=["POST"])
def open_target_masters(target_id, pipeline):
    """Open every channel master of a target in PixInsight, each view named
    for its filter.

    Paths come from the pipeline trace (`available_tap_path` at the
    "Channel" tap that `merge_target_channel` declares), never from a
    filesystem scan. Scanning for a master by stable_hash matches
    superseded generations as readily as current ones, and a stale master
    opened by hand is indistinguishable from live output once it is on
    screen — which is the whole reason this button exists.
    """
    from web_server.backend.metadata import available_tap_path
    from file_io.planar import reconstruct_planar_tiff
    from pix.runner import open_in_gui

    targets = context.targets or {}
    if target_id not in targets:
        return jsonify({"error": "target not found"}), 404
    target = targets[target_id].get()
    stacks = context.stacks or {}

    views: dict[str, Path] = {}
    missing: list[str] = []
    for stack_id in target.stack_ids:
        filter = stacks[stack_id].get().filter
        path = available_tap_path(
            pipeline, "Channel", "target", target_id, filter)
        if path is None:
            missing.append(filter)
            continue
        views[filter] = Path(reconstruct_planar_tiff(path))

    if not views:
        return jsonify(
            {"error": "no channel masters on disk", "missing": missing}), 404

    open_in_gui(views)
    return jsonify({"opened": {k: str(v) for k, v in views.items()},
                    "missing": missing})
