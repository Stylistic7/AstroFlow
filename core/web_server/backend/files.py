from flask import Flask, request, jsonify, send_file, Blueprint
import os

files_bp = Blueprint("files", __name__)


@files_bp.route("/api/list_dir")
def list_dir():
  path = request.args.get("path", default=os.path.expanduser("~"))

  try:
    entries = os.scandir(path)
    files = []
    for entry in entries:
      files.append({
          "id": os.path.join(path, entry.name),
          "name": entry.name,
          "isDir": entry.is_dir(),
      })
    return jsonify({"files": files, "path": path})
  except Exception as e:
    return jsonify({"error": str(e)}), 400
