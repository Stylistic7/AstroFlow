"""The Third-Party Tools page's API: where each program AstroFlow runs is,
and whether it works on every Mac in the cluster (`third_party_tools`).

A location set here is the same on every Mac (`node_settings`), like the
folders; leaving it empty uses where the program normally installs.
"""
from pathlib import Path

from flask import Blueprint, abort, jsonify, request

import node_settings
from context import local_context as context
from third_party_tools import TOOLS, check_cluster

tools_bp = Blueprint("tools", __name__)


def _current() -> dict:
    configured = context.tool_paths or {}
    return {"tools": [{"key": t.key, "name": t.name, "used_for": t.used_for, "install": t.install,
                       "locatable": t.locatable, "path": configured.get(t.key)} for t in TOOLS]}


@tools_bp.route("/api/tools", methods=["GET"])
def get_tools():
    return jsonify(_current())


@tools_bp.route("/api/tools", methods=["PUT"])
def put_tools():
    paths = {}
    for t in TOOLS:
        path = str((request.get_json(force=True).get("paths") or {}).get(t.key) or "").strip()
        if not path:
            continue
        if not t.locatable:
            abort(400, description=f"{t.name} has no location to set")
        if not Path(path).is_absolute():
            abort(400, description=f"{t.name}: {path} must be a full path")
        paths[t.key] = path
    # Every Mac first: if one can't take the settings, nothing is saved.
    node_settings.push_to_cluster({**node_settings.current(), "tool_paths": paths})
    context.tool_paths = paths
    return jsonify(_current())


@tools_bp.route("/api/tools/check", methods=["POST"])
def check_tools():
    """Every tool's trial on every Mac, the head first. Takes a few seconds:
    GraXpert's trial runs a real background extraction."""
    return jsonify({"nodes": check_cluster()})
