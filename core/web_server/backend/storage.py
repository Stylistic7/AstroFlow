"""Where AstroFlow reads raw frames from and writes everything it makes.

The folders are the user's choice, kept in the durable store
(`context.input_dirs`, `context.output_dir`). Until both are set nothing is
scanned or built (`state/actor.py`). Changing them re-runs the graph: a new
input folder is scanned, and a new output folder is rebuilt from scratch,
because every product is filed under the output folder.
"""
import subprocess
from pathlib import Path

from flask import Blueprint, abort, jsonify, request

from context import local_context as context

storage_bp = Blueprint("storage", __name__)


def _current() -> dict:
    output = context.output_dir
    return {
        "input_dirs": [str(d) for d in (context.input_dirs or [])],
        "output_dir": str(output) if output else None,
    }


def _folder(raw, what: str) -> Path:
    path = Path(str(raw or "").strip()).expanduser()
    if not path.is_absolute():
        abort(400, description=f"{what} must be a full path, like /Volumes/Astro/Raw")
    if not path.is_dir():
        abort(400, description=f"{what} {path} is not a folder this Mac can see")
    return path


@storage_bp.route("/api/storage", methods=["GET"])
def get_storage():
    return jsonify(_current())


@storage_bp.route("/api/storage", methods=["PUT"])
def put_storage():
    from state.actor import state

    body = request.get_json(force=True)
    inputs = [_folder(d, "Input folder") for d in body.get("input_dirs") or []]
    if not inputs:
        abort(400, description="Add at least one input folder")
    if len(set(inputs)) != len(inputs):
        abort(400, description="The same input folder is listed twice")
    output = _folder(body.get("output_dir"), "Output folder")
    for d in inputs:
        if output == d or output.is_relative_to(d) or d.is_relative_to(output):
            abort(400, description=f"The output folder can't be inside an input folder, or contain one ({d})")

    changed = inputs != list(context.input_dirs or []) or output != context.output_dir
    context.input_dirs = inputs
    context.output_dir = output
    if changed:
        state.drain_and_restart()
    return jsonify(_current())


@storage_bp.route("/api/storage/choose_folder", methods=["POST"])
def choose_folder():
    """The Mac's own folder picker, opened on the machine running AstroFlow.
    A browser can't hand a page a folder's path, so the server asks.
    `{"file": true}` picks a file instead. Returns {"path": null} when the
    user cancels."""
    body = request.get_json(silent=True) or {}
    prompt = str(body.get("prompt") or "Choose a folder")
    kind = "file" if body.get("file") else "folder"
    script = f'POSIX path of (choose {kind} with prompt {_applescript_str(prompt)})'
    result = subprocess.run(["/usr/bin/osascript", "-e", script], capture_output=True, text=True)
    if result.returncode != 0:  # -128: the user cancelled
        return jsonify({"path": None})
    return jsonify({"path": result.stdout.strip().rstrip("/") or "/"})


def _applescript_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
