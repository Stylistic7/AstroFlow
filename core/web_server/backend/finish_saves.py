"""Saved finish looks (2026-10-04): one click saves the current parameters
of a finish for one object and pipeline, one click restores them, one click
deletes a save. Stored durably (user curation) under
("finish_saves", object_type, object_id, pipeline, finish) as a list of
{id, created, params, thumbnail}; the thumbnail is rendered server-side from
the saved parameters, so it shows exactly that look.
"""
from __future__ import annotations

import base64
import datetime
import uuid

import cv2
import numpy as np
from flask import Blueprint, abort, jsonify, request

from context import local_context as context
from web_server.backend._route import finish_route

finish_saves_bp = Blueprint("finish_saves", __name__)

_HIER = ("object_type", "object_id")
_THUMB_W = 160


def _key(object_type, object_id, pipeline, finish) -> tuple:
    return ("finish_saves", object_type, object_id, pipeline, finish)


def _thumbnail(object_type, object_id, pipeline, finish, params) -> "str | None":
    """A small JPEG data URL of the finish rendered with `params`."""
    from finishes.preview import render_finish_preview
    r = render_finish_preview(object_type, object_id, pipeline, finish,
                              min_px=2 * _THUMB_W, values=params)
    if r is None:
        return None
    img = cv2.imdecode(np.frombuffer(r[0], np.uint8), cv2.IMREAD_COLOR)
    h = max(1, round(img.shape[0] * _THUMB_W / img.shape[1]))
    img = cv2.resize(img, (_THUMB_W, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


@finish_route(finish_saves_bp, "/api/finish_saves", hierarchy=_HIER, methods=["GET"])
def list_saves(object_type, object_id, pipeline, finish):
    return jsonify({"saves": context.get(_key(object_type, object_id, pipeline, finish)) or []})


@finish_route(finish_saves_bp, "/api/finish_saves", hierarchy=_HIER, methods=["POST"])
def add_save(object_type, object_id, pipeline, finish):
    params = (request.get_json(silent=True) or {}).get("params")
    if not isinstance(params, dict) or not params or not all(
            isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool)
            for k, v in params.items()):
        return abort(400, "params must be a non-empty {name: number} object")
    params = {k: float(v) for k, v in params.items()}
    key = _key(object_type, object_id, pipeline, finish)
    saves = list(context.get(key) or [])
    saves.append({
        "id": uuid.uuid4().hex[:12],
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "params": params,
        "thumbnail": _thumbnail(object_type, object_id, pipeline, finish, params),
    })
    context.set(key, saves)
    return jsonify({"saves": saves})


@finish_route(finish_saves_bp, "/api/finish_saves", hierarchy=_HIER,
              suffix="/<save_id>", methods=["DELETE"])
def delete_save(object_type, object_id, pipeline, finish, save_id):
    key = _key(object_type, object_id, pipeline, finish)
    saves = [s for s in (context.get(key) or []) if s.get("id") != save_id]
    context.set(key, saves)
    return jsonify({"saves": saves})
