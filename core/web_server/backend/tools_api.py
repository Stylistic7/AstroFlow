"""The Third-Party Tools page's API: where each program AstroFlow runs is,
and whether it works on every Mac in the cluster (`third_party_tools`).

A location set here is the same on every Mac (`node_settings`), like the
folders; leaving it empty uses where the program normally installs (for
GraXpert, solve-field and libvips, inside AstroFlow.app). The one other
setting is solve-field's: a folder of extra index files, for fields smaller
than the app's own set covers.
"""
from pathlib import Path

from flask import Blueprint, abort, jsonify, request

import node_settings
from context import local_context as context
from third_party_tools import TOOLS, check_cluster

tools_bp = Blueprint("tools", __name__)


def _current() -> dict:
    configured = context.tool_paths or {}
    index = context.astrometry_index_dir
    return {"tools": [{"key": t.key, "name": t.name, "used_for": t.used_for, "install": t.install,
                       "locatable": t.locatable, "path": configured.get(t.key)} for t in TOOLS],
            "extra_index_dir": str(index) if index else None}


@tools_bp.route("/api/tools", methods=["GET"])
def get_tools():
    return jsonify(_current())


@tools_bp.route("/api/tools", methods=["PUT"])
def put_tools():
    body = request.get_json(force=True)
    paths = {}
    for t in TOOLS:
        path = str((body.get("paths") or {}).get(t.key) or "").strip()
        if not path:
            continue
        if not t.locatable:
            abort(400, description=f"{t.name} has no location to set")
        if not Path(path).is_absolute():
            abort(400, description=f"{t.name}: {path} must be a full path")
        paths[t.key] = path
    index = str(body.get("extra_index_dir") or "").strip()
    if index and not Path(index).is_absolute():
        abort(400, description=f"Extra index files: {index} must be a full path")
    if index and not Path(index).is_dir():
        abort(400, description=f"Extra index files: {index} is not a folder this Mac can see")
    # Every Mac first: if one can't take the settings, nothing is saved.
    node_settings.push_to_cluster({**node_settings.current(), "tool_paths": paths,
                                   "astrometry_index_dir": index or None})
    context.tool_paths = paths
    context.astrometry_index_dir = Path(index) if index else None
    return jsonify(_current())


@tools_bp.route("/api/tools/check", methods=["POST"])
def check_tools():
    """Every tool's trial on every Mac, the head first. Takes a few seconds:
    GraXpert's trial runs a real background extraction."""
    return jsonify({"nodes": check_cluster()})


@tools_bp.route("/api/node_settings", methods=["GET"])
def get_node_settings():
    """What every Mac keeps a copy of (`node_settings`); a joining worker
    reads it here."""
    return jsonify(node_settings.current())
