"""The stack page's subframe video (`previews/stack_video.py`), rendered on
demand in its own process and served as a file.

    GET /video/stack/<id>            the video (200, ranges honoured) when it
                                     is rendered; 202 {state} while it renders;
                                     404 when it is neither and was not asked for
    GET /video/stack/<id>?start=1    ...and ask for it when it is neither

One job per video key, so two tabs asking at once start one render. A
status that has said "running" for longer than `_STALE` is a process that
died; a new ask starts over.
"""
from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path

from flask import Blueprint, abort, jsonify, request, send_file

from context import local_context as context
from previews.stack_video import JOBS, VIDEOS, WIDTH, video_key

videos_bp = Blueprint("videos", __name__)
_CORE = Path(__file__).resolve().parents[2]
_STALE = datetime.timedelta(minutes=30)


def _status(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None


def _running(status: dict | None) -> bool:
    if not status or status.get("state") != "running":
        return False
    started = datetime.datetime.fromisoformat(status["started"])
    return datetime.datetime.now(datetime.timezone.utc) - started < _STALE


@videos_bp.route("/video/stack/<stack_id>")
def stack_video(stack_id: str):
    stack = context.stacks[stack_id].get()
    width = request.args.get("width", default=WIDTH, type=int)
    key = video_key(stack, context.settings.decision_overrides, width)
    dest, status_file = VIDEOS / f"{key}.mp4", JOBS / f"{key}.json"
    if dest.exists():
        return send_file(dest, mimetype="video/mp4", conditional=True)
    status = _status(status_file)
    if _running(status):
        return jsonify(status), 202
    if not request.args.get("start"):
        if status and status.get("state") == "error":
            return jsonify(status), 404
        abort(404)
    JOBS.mkdir(parents=True, exist_ok=True)
    status = {"state": "running", "path": str(dest),
              "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    status_file.write_text(json.dumps(status))
    job_file = JOBS / f"{key}.job"
    job_file.write_text(json.dumps({"stack_id": stack_id, "width": width,
                                    "dest": str(dest), "status": str(status_file)}))
    subprocess.Popen([sys.executable, "-m", "previews.stack_video", str(job_file)],
                     cwd=_CORE, start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return jsonify(status), 202
