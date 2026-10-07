"""Open a finish's output in Photoshop (2026-10-04): the finish's math and
the viewer's current parameters baked into an uncompressed 16-bit sRGB TIFF
under /tmp/AstroFlow/exports, then opened in the newest installed Photoshop,
where the user decides where to keep it. The bake runs in its own process
(`python -m finishes.export`); the button polls a status file for it.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

from flask import Blueprint, abort, jsonify, request

from web_server.backend._route import finish_route

finish_export_bp = Blueprint("finish_export", __name__)

EXPORT_DIR = Path("/tmp/AstroFlow/exports")
_JOBS = EXPORT_DIR / ".jobs"
_CORE = Path(__file__).resolve().parents[2]


def _params() -> dict:
    params = (request.get_json(silent=True) or {}).get("params")
    if not isinstance(params, dict) or not params or not all(
            isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool)
            for k, v in params.items()):
        abort(400, "params must be a non-empty {name: number} object")
    return {k: float(v) for k, v in params.items()}


@finish_route(finish_export_bp, "/api/finish_export",
              hierarchy=("object_type", "object_id"), methods=["POST"])
def start_export(object_type, object_id, pipeline, finish):
    from catalog import resolve_finish
    from finishes.export import newest_photoshop
    params = _params()
    b = resolve_finish(object_type, object_id, pipeline, finish)
    if b is None or b.inputs[0].path is None:
        abort(404, "this view isn't built yet")
    app = newest_photoshop()
    if app is None:
        abort(409, "Adobe Photoshop isn't installed on this machine")
    job = uuid.uuid4().hex[:12]
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = re.sub(r"[^A-Za-z0-9]+", "-", finish).strip("-")
    dest = EXPORT_DIR / f"{name}-{object_type}-{object_id}-{stamp}.tif"
    _JOBS.mkdir(parents=True, exist_ok=True)
    status = _JOBS / f"{job}.json"
    status.write_text(json.dumps({"state": "running", "path": str(dest)}))
    job_file = _JOBS / f"{job}.job"
    job_file.write_text(json.dumps({
        "finish": finish, "primary": b.inputs[0].tap,
        "paths": [None if i.path is None else os.fspath(i.path) for i in b.inputs],
        "params": params, "dest": str(dest), "status": str(status), "app": app,
    }))
    subprocess.Popen([sys.executable, "-m", "finishes.export", str(job_file)],
                     cwd=_CORE, start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return jsonify({"job": job, "path": str(dest)})


@finish_export_bp.route("/api/finish_export/job/<job>")
def export_status(job):
    if not re.fullmatch(r"[0-9a-f]{12}", job):
        abort(404)
    try:
        return jsonify(json.loads((_JOBS / f"{job}.json").read_text()))
    except FileNotFoundError:
        abort(404)
