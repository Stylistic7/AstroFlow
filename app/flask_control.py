import webbrowser
import os
import subprocess
import signal
import flask
import threading
import sys
from pathlib import Path

from base_path import get_base_path
# from web_server.backend.app import *

_ASTROFLOW_URL = 'http://localhost:80'

process = None

base_path = get_base_path()
# print(f'### base_path {base_path}')

# app = flask.Flask(
#     __name__, template_folder=f'{base_path}/core/web_server/backend/templates')

# def flask_thread_body():
#   app.run(host='0.0.0.0', port=80)

# def start_flask():
#   print('### start_flask()')
#   app.run(host='0.0.0.0', port=80)


def start_flask():
  global process

  # # TODO: Isn't this a race condition?
  print(f'Starting Flask server at base path {base_path}...')
  # # python_path = os.path.join(base_path, 'bin', 'python3.11')
  # # python_path = Path(sys.executable).with_stem("python")
  command = f"""cd {base_path}/core && {sys.executable} -m web_server.backend.app --subprocess"""
  print(f'Command is {command}')
  process = subprocess.Popen(command, shell=True, start_new_session=True)

  # flask_thread = threading.Thread(target=flask_thread_body)
  # flask_thread.start()

  # TODO: The subprocess above blocks forever, so we need another way to
  # open the web browser upon success. Maybe polling :80?
  # webbrowser.open(_ASTROFLOW_URL)

  return process


def stop_flask():
  global process

  os.killpg(os.getpgid(process.pid), signal.SIGKILL)
