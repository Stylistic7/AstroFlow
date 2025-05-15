from flask import Flask, request, jsonify, send_file, Blueprint
import os
from pathlib import Path

from context import context
from file_io.generic import read_header
from customization.settings import Settings, StackSettings
from state.state_machine import enqueue_action
from customization.compute_graph import scatter_customize_stacks

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
          ('.fits', '.fit', '.xisf')):
        try:
          header = read_header(Path(full_path))
          file_info.update({
              "camera": header.get("INSTRUME"),
              "exposure": header.get("EXPOSURE"),
              # Add more fields here if desired
              "header": header,  # Optional: include full header
          })
        except Exception as e:
          file_info["header_error"] = str(e)

      files.append(file_info)

    return jsonify({"files": files, "path": path})
  except Exception as e:
    return jsonify({"error": str(e)}), 400


@files_bp.route('/api/select_file', methods=['POST'])
def select_file():
  data = request.get_json()
  print(f'### select_file data {data}')

  type = data['type']
  stack_id = data['stack_id']
  run_id = data['run_id']
  filename = data['filename']

  settings = context.settings or Settings()
  stack_settings = settings.stack_settings.get(run_id, StackSettings())
  stack_settings.dependencies[type] = {'filename': filename}
  settings.stack_settings[run_id] = stack_settings
  context.settings = settings
  print(f'### context.settings is now {context.settings}')

  # TODO: Only customize the single stack affected here.
  enqueue_action(
      dict(type='run_task',
           task=(scatter_customize_stacks,),
           announce_message='reload_data'))

  return jsonify({"status": "ok"})
