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


@files_bp.route("/api/concerns/<object_type>/<object_id>", methods=["GET"])
def concerns_for_object(object_type, object_id):
    """Concerns reported by the most recent producer run for this object.

    Looks up the per-object key written by `@capture_logs` when a
    `@makes_image`-decorated producer ran. Returns a list of dicts;
    each carries `event` (title), `detail`, and `_concern`
    ("warning"/"error") plus any structured fields the emitter attached.

    Query params (all required for full disambiguation):
      quality (default "Master")
      implementation (default "PixInsight")
      filter — required for object_type=mosaic
    """
    from products.identity import StackId, MosaicChannelId
    from previews.types import QualityLevel, Implementation
    from context import get_cache

    quality_str = request.args.get("quality", "Master")
    impl_str = request.args.get("implementation", "PixInsight")
    try:
        quality = QualityLevel(quality_str)
        implementation = Implementation(impl_str)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if object_type == "stack":
        if object_id not in (context.stacks or {}):
            return jsonify({"error": "stack not found"}), 404
        stack = context.stacks[object_id].get()
        obj_id = StackId(stack=stack)
        type_name = type(stack).__name__  # "Stack" or "CalibrationStack"
    elif object_type == "mosaic":
        if object_id not in (context.mosaics or {}):
            return jsonify({"error": "mosaic not found"}), 404
        mosaic = context.mosaics[object_id].get()
        filter_arg = request.args.get("filter", "")
        cache = get_cache()
        # Mosaic-level direct concerns (from store_mosaic_color_channel
        # and friends) plus bubbled-up stack-level concerns from each
        # constituent stack at this filter — so a WBPP frame-drop on
        # any input shows up on the mosaic detail page.
        direct = cache.get(
            (
                "concerns", "Mosaic", str(quality), str(implementation),
                MosaicChannelId(mosaic=mosaic, filter=filter_arg).cache_key(),
            ),
            [],
        )
        bubbled = []
        for stack in _constituent_stacks(mosaic, filter_arg):
            stack_concerns = cache.get(
                (
                    "concerns", type(stack).__name__, str(quality),
                    str(implementation), StackId(stack=stack).cache_key(),
                ),
                [],
            )
            for c in stack_concerns:
                bubbled.append({**c, "from_stack": stack.stable_hash})
        return jsonify(direct + bubbled)
    else:
        return jsonify({"error": f"object_type {object_type!r} not supported"}), 400

    key = (
        "concerns",
        type_name,
        str(quality),
        str(implementation),
        obj_id.cache_key(),
    )
    cache = get_cache()
    return jsonify(cache.get(key, []))


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
    from previews.types import QualityLevel, Implementation
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
        quality=QualityLevel.MASTER,
        implementation=Implementation.PIXINSIGHT,
    )
    deleted = stack_files.invalidate(stack, config)
    if deleted:
        state.drain_and_restart()
    return jsonify({"stack_id": stack_id, "deleted": deleted})


@files_bp.route("/api/open_stack_master/<stack_id>")
def open_stack_master(stack_id):
    from products.registry import get_filename_fn, MediaType
    from previews.types import QualityLevel
    stack = context.stacks[stack_id].get()
    filename_fn = get_filename_fn(MediaType.IMAGE, type(stack), QualityLevel.MASTER)
    filename = filename_fn(stack)
    print(f"### opening filename {filename}")
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


@files_bp.route("/api/open_mosaic_master/<mosaic_id>/<implementation>/<filter>")
def open_mosaic_master(mosaic_id, implementation, filter):
    """Open the most-processed image for a mosaic+filter in PixInsight.

    Prefers the canonical Master output of store_mosaic_color_channel_*.
    Falls back to the per-filter merge progress TIFF — overwritten by
    each merge_images step, so it always reflects the latest state of
    the in-flight mosaic for that filter.
    """
    from pathlib import Path
    from products.registry import get_filename_fn, MediaType
    from previews.types import QualityLevel, Implementation
    from mosaics.types import Mosaic

    mosaics = context.mosaics or {}
    if mosaic_id not in mosaics:
        return jsonify({"error": "mosaic not found"}), 404
    mosaic = mosaics[mosaic_id].get()
    impl = Implementation(implementation)

    candidates = []

    filename_fn = get_filename_fn(MediaType.IMAGE, Mosaic, QualityLevel.MASTER, impl)
    if filename_fn is not None:
        candidates.append(filename_fn(mosaic, filter))

    if context.output_dir is not None:
        candidates.append(
            Path(context.output_dir) / "Mosaic" / str(QualityLevel.MASTER)
            / f"progress_{mosaic.stable_hash}_{filter}.tiff"
        )

    for path in candidates:
        if path is not None and Path(path).exists():
            print(f"### opening {path} in PixInsight")
            subprocess.Popen([
                "/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight",
                str(path),
            ])
            return jsonify({"opened": str(path)})

    return jsonify({"error": "no image available to open"}), 404
