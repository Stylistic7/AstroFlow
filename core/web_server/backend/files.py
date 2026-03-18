from flask import request, jsonify, Blueprint, Response
import os
from pathlib import Path
import subprocess

from context import context
from file_io.generic import read_header
from customization.settings import Settings, StackSettings
from state.actor import state
from stacks.stacking import stack_files
from stacks.types import FilenameDependency

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
    data = request.get_json()
    type = data["type"]
    stack_id = data["stack_id"]
    run_id = data["run_id"]
    filename = data["filename"]

    # TODO: Watch out for race conditions here.
    settings = context.settings or Settings()
    stack_settings = settings.stack_settings.get(run_id, StackSettings())
    stack_settings.dependencies[type] = FilenameDependency(filename=filename)
    settings.stack_settings[run_id] = stack_settings
    context.settings = settings

    # TODO: Maybe don't pass the entire settings struct?
    state.dispatch(dict(type="update_settings", stack_id=stack_id, settings=settings))

    return jsonify({"status": "ok"})


@files_bp.route("/api/open_stack_master/<stack_id>")
def open_stack_master(stack_id):
    stack = context.stacks[stack_id].get()
    filename = stack_files.filename_fn(stack)
    print(f"### opening filename {filename}")
    subprocess.Popen(
        [
            "/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight",
            str(filename),
        ]
    )

    return Response(status=200)


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
