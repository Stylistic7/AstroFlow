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

    Resolves the master path via the PipelineOutputIndex (tap-keyed)
    and parses the file's headers + XISF properties into a uniform
    shape (StackingReport). Returns `null` when the tap point hasn't
    produced an artifact yet.

    Query params (defaults reflect the PixInsight Master pipeline's
    Master tap):
      pipeline (default "PixInsight Master")
      display  (default "Master")
    """
    from stacks.stacking_report import read_stacking_report
    from web_server.backend.metadata import tap_path_for

    if stack_id not in (context.stacks or {}):
        return jsonify({"error": "stack not found"}), 404

    pipeline = request.args.get("pipeline", "PixInsight Master")
    display = request.args.get("display", "Master")

    master_path = tap_path_for(pipeline, display, "stack", stack_id, None)
    if not master_path or not Path(master_path).exists():
        return jsonify(None)

    try:
        return jsonify(read_stacking_report(master_path).to_json())
    except Exception as e:
        return jsonify({"error": f"failed to read report: {e}"}), 500


@files_bp.route("/api/concerns/<object_type>/<object_id>", methods=["GET"])
def concerns_for_object(object_type, object_id):
    """Concerns reported by recent producer runs for this object.

    @capture_logs writes one entry per `(op_name, object_type, id)`
    tuple. PI Master and PI Downscaled now SHARE the same op
    (`post_process_stack`), so their concerns merge under the same
    op_name — distinct on-disk outputs come from the call_hash, not
    a separate op name. This endpoint scans every entry for the
    requested object and returns the merged list; mosaic concerns
    also bubble up the concerns from constituent stacks at the
    requested filter.

    Query params:
      filter — required for object_type=mosaic
    """
    from products.identity import StackId, MosaicChannelId
    from context import get_cache

    cache = get_cache()
    filter_arg = request.args.get("filter", "")

    def concerns_for_key(object_type_name: str, identity_key: tuple) -> list[dict]:
        """Sweep every cache entry of the shape
        `('concerns', op_name, object_type_name, identity_key)` and
        return the merged list. We don't know which ops have written —
        any producer that ran and called `capture_logs.concern` could
        have, so we pattern-match the keys."""
        out: list[dict] = []
        for key in list(cache.iterkeys() if hasattr(cache, "iterkeys") else []):
            if (
                isinstance(key, tuple) and len(key) == 4
                and key[0] == "concerns"
                and key[2] == object_type_name
                and key[3] == identity_key
            ):
                out.extend(cache.get(key, []))
        return out

    if object_type == "stack":
        if object_id not in (context.stacks or {}):
            return jsonify({"error": "stack not found"}), 404
        stack = context.stacks[object_id].get()
        return jsonify(concerns_for_key(
            type(stack).__name__, StackId(stack=stack).cache_key(),
        ))

    if object_type == "mosaic":
        if object_id not in (context.mosaics or {}):
            return jsonify({"error": "mosaic not found"}), 404
        mosaic = context.mosaics[object_id].get()
        direct = concerns_for_key(
            "Mosaic",
            MosaicChannelId(mosaic=mosaic, filter=filter_arg).cache_key(),
        )
        bubbled = []
        for stack in _constituent_stacks(mosaic, filter_arg):
            for c in concerns_for_key(
                type(stack).__name__, StackId(stack=stack).cache_key(),
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


@files_bp.route("/api/open_stack_master/<stack_id>", defaults={"pipeline": "PixInsight"})
@files_bp.route("/api/open_stack_master/<stack_id>/<path:pipeline>")
def open_stack_master(stack_id, pipeline):
    """Open the stack's master file in PixInsight.

    `pipeline` (URL-encoded) names the pipeline whose master we want —
    different pipelines produce masters at different paths (downscaled
    vs full-res, OSS vs PixInsight backend, etc.). Default "PixInsight"
    matches the legacy single-segment URL the frontend uses today.
    Resolved through the cached pipeline trace, same path the op
    actually wrote to.
    """
    from dependency_graph.trace_paths import traced_path_for
    stack = context.stacks[stack_id].get()
    # The cart item walks the trace from the most-specific entity that
    # could own the stack — typically the target the stack belongs to.
    # Falling back to the bare stack works for stack-targeted pipelines.
    cart_item = _cart_item_for_stack(stack, pipeline)
    if cart_item is None:
        return Response(status=404)
    filename = traced_path_for(
        cart_item, "stack_files", stack=stack.stable_hash,
    )
    if filename is None:
        return Response(status=404)
    print(f"### opening filename {filename}")
    subprocess.Popen(
        [
            "/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight",
            str(filename),
        ]
    )

    return Response(status=200)


def _cart_item_for_stack(stack, pipeline: str) -> tuple[str, str, str] | None:
    """Pick a cart item whose trace will include `stack`'s stack_files
    call. Prefers the owning target/mosaic so the trace reflects the
    same config the pipeline would use end-to-end."""
    for t_id, t_ref in (context.targets or {}).items():
        target = t_ref.get() if hasattr(t_ref, "get") else t_ref
        if stack.stable_hash in target.stack_ids:
            return ("target", t_id, pipeline)
    for m_id, m_ref in (context.mosaics or {}).items():
        mosaic = m_ref.get() if hasattr(m_ref, "get") else m_ref
        for t_id in mosaic.target_ids:
            t_ref = (context.targets or {}).get(t_id)
            if t_ref is None:
                continue
            target = t_ref.get() if hasattr(t_ref, "get") else t_ref
            if stack.stable_hash in target.stack_ids:
                return ("mosaic", m_id, pipeline)
    return ("stack", stack.stable_hash, pipeline)


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


@files_bp.route("/api/open_mosaic_master/<mosaic_id>/<path:pipeline_and_display>/<filter>")
def open_mosaic_master(mosaic_id, pipeline_and_display, filter):
    """Open a specific (pipeline, tap) of a mosaic+filter in PixInsight.

    Path resolution goes through the PipelineOutputIndex
    (`tap_path_for`); `<path:pipeline_and_display>` is the
    `<pipeline>/<display>` pair joined into one URL segment so
    pipeline names with spaces don't need extra escaping.
    """
    from web_server.backend.metadata import tap_path_for

    pipeline, _, display = pipeline_and_display.rpartition("/")
    if not pipeline or not display:
        return jsonify({"error": "bad pipeline/display"}), 400

    mosaics = context.mosaics or {}
    if mosaic_id not in mosaics:
        return jsonify({"error": "mosaic not found"}), 404

    path = tap_path_for(pipeline, display, "mosaic", mosaic_id, filter)
    if path is None:
        return jsonify({"error": "no image available to open"}), 404

    print(f"### opening {path} in PixInsight")
    subprocess.Popen([
        "/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight",
        str(path),
    ])
    return jsonify({"opened": str(path)})
