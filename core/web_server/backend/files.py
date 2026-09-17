from flask import request, jsonify, Blueprint, Response
import os
from pathlib import Path

from context import local_context as context
from file_io.generic import read_header
from file_io.find_image_files import is_external_master_name
from file_io.sidecar import get_section
from metadata.image_types import is_light_frame
from web_server.backend.announcer import announce
from web_server.backend._route import finish_route
from products.product import Product
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

    Query params:
      pipeline (default: the pipeline of the stack's default finish)
    The report IS the stacker's master, so it reads the "Master" tap — the
    one tap name the report sanctions, whichever finish the page shows.
    """
    from stacks.stacking_report import read_stacking_report
    from catalog import Product, default_finish_for, master_path

    if stack_id not in (context.stacks or {}):
        return jsonify({"error": "stack not found"}), 404

    pipeline = request.args.get("pipeline")
    if not pipeline:
        b = default_finish_for("stack", stack_id)
        if b is None:
            return jsonify(None)
        pipeline = b.pipeline
    stack = context.stacks[stack_id].get()

    path = master_path(Product("stack", stack_id, pipeline, "Master", stack.filter))
    if path is None:
        return jsonify(None)

    try:
        return jsonify(read_stacking_report(str(path)).to_json())
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
      filter — a filter ROLE (H, S, O, R, G, B, L); required for
               object_type=mosaic
    """
    from context import get_cache
    from catalog import role_of

    cache = get_cache()
    role = role_of(request.args.get("filter", ""))

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
        stacks = _constituent_stacks(mosaic, role)
        direct = []
        for raw in sorted({s.filter for s in stacks}):
            direct.extend(concerns_for_key("Mosaic", f"{mosaic.stable_hash}_{raw}"))
        bubbled = []
        for stack in stacks:
            for c in concerns_for_key(
                type(stack).__name__, stack.stable_hash,
            ):
                bubbled.append({**c, "from_stack": stack.stable_hash})
        return jsonify(direct + bubbled)

    return jsonify({"error": f"object_type {object_type!r} not supported"}), 400


def _constituent_stacks(mosaic, role):
    """The mosaic's stacks whose filter normalises to `role`."""
    from catalog import role_of
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
            if role_of(stack.filter) == role:
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


def _open_layers(b) -> "tuple[dict[str, Path], list[str]]":
    """What PixInsight gets for a finish: for each bound input, the file
    itself when it is one layer, else the mono layers the join was built from
    — the same LAYER (payload tag) in the pipeline's per-filter taps, one per
    band role of the join (only the finish's own band for a channel finish).
    A join with no traced constituents falls back to itself. Data-driven:
    payload tags and filter roles, never a tap name."""
    from catalog import band_roles, master_path, product_taps, resolve_product
    from catalog.finishes import role_of
    from previews.create_tile_pyramid import payload_tags
    from previews.pyramid_pages import _get_vips_page

    def _describe(path):
        img = _get_vips_page(os.fspath(path), 0)
        tags = payload_tags(img)
        # The stars joins and the mono layers stamp LAYER; the starless joins
        # (`stack_sho_channels`, `blend_linear_sho`, `linear_rgb_starless`)
        # predate the tag and carry none — the same reading the finishes use
        # (`stars_inputs`: stars iff LAYER == "stars"). Stamping them is a
        # rebuild of the NXT tail, so it waits for a wipe window.
        return img.bands, tags.get("LAYER") or "starless", tags

    taps = product_taps(b.object_type, b.object_id, b.pipeline)
    views: dict[str, Path] = {}
    missing: list[str] = []
    for i in b.inputs:
        product = i.product(b)
        path = (master_path(product) if product is not None else None) or i.path
        if path is None:
            missing.append(i.tap)
            continue
        try:
            bands, layer, tags = _describe(path)
        except Exception:
            bands, layer, tags = 1, "starless", {}
        if bands == 1:
            views[f"{i.tap} {i.filter}"] = Path(path)
            continue
        # a channel finish opens only its own band of the PRIMARY; the stars
        # it screens in are every band of the stars join
        wanted = [b.role] if (b.role and i.index == 0) else band_roles(bands, tags, i.filter)
        found: dict[str, Path] = {}
        for e in taps:
            role = role_of(e["filter"])
            if role not in wanted or role == "none":
                continue
            cpath, _ = resolve_product(Product(b.object_type, b.object_id, b.pipeline,
                                               e["display"], e["filter"]))
            if cpath is None:
                continue
            try:
                cbands, clayer, _ = _describe(cpath)
            except Exception:
                continue
            if cbands == 1 and clayer == layer:
                found[f"{e['display']} {e['filter']}"] = Path(cpath)
        if found and len(found) == len(wanted):
            views.update(found)
        else:
            views[f"{i.tap} {i.filter}"] = Path(path)
    return views, missing


def _export_and_open(views: "dict[str, Path]") -> None:
    """Off the request thread: rewrite what PixInsight cannot read (tiled
    TIFF → stripped, uncompressed, cached), then open every view by name."""
    from file_io.planar import reconstruct_planar_tiff
    from pix.runner import open_in_gui
    try:
        ready = {name: Path(reconstruct_planar_tiff(path)) for name, path in views.items()}
        open_in_gui(ready)
    except Exception as e:  # a background thread: log, never die silently
        print(f"### open_finish: export failed: {type(e).__name__}: {e}")


@finish_route(files_bp, "/api/open", hierarchy=("object_type", "object_id"), methods=["POST"])
def open_finish(object_type, object_id, pipeline, finish):
    """Open a finish's linear inputs in PixInsight: the Hubble page opens the
    S, H, O starless layers and the R, G, B stars layers its joins were built
    from; the H page opens the H layer. The finish itself is a look, applied
    live; what PixInsight wants is the linear data under it (`_open_layers`).

    Paths come from the catalog, never from a filesystem scan: scanning by
    stable_hash matches superseded generations as readily as current ones,
    and a stale master opened by hand is indistinguishable from live output
    once on screen. The deliverable itself (`master_path`) when the current
    generation is built; else what the viewer serves — the last-good
    generation (`resolve_product`'s N-1), so the button opens the picture
    on screen, badged out of date there.

    Returns 202 at once: a tiled gigapixel layer is rewritten for PixInsight
    first (uncompressed, ~25 s per 8 GB channel), which no request should
    wait on — the windows appear in PixInsight as they land."""
    import threading
    from catalog import resolve_finish
    from file_io.planar import is_tiled_tiff

    b = resolve_finish(object_type, object_id, pipeline, finish)
    if b is None:
        return jsonify({"error": "no such finish for this object"}), 404
    views, missing = _open_layers(b)
    if not views:
        return jsonify({"error": "no image available to open", "missing": missing}), 404
    threading.Thread(target=_export_and_open, args=(views,), daemon=True).start()
    return jsonify({
        "opening": sorted(views),
        "exporting": sorted(n for n, p in views.items() if is_tiled_tiff(p)),
        "missing": missing,
    }), 202
