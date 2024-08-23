import webbrowser
import os
import subprocess
import signal

from core.base_path import get_base_path

_ASTROFLOW_URL = 'http://localhost:80'

process = None


def start_flask():
  global process

  # TODO: Isn't this a race condition?
  base_path = get_base_path()
  print(f'Starting Flask server at base path {base_path}...')
  command = f"""source ~/.zshrc && conda activate AstroFlow && cd {base_path} && python3 -m web_server.backend.app"""
  process = subprocess.Popen(command, shell=True, start_new_session=True)

  # TODO: The subprocess above blocks forever, so we need another way to
  # open the web browser upon success. Maybe polling :80?
  webbrowser.open(_ASTROFLOW_URL)


def stop_flask():
  global process

  os.killpg(os.getpgid(process.pid), signal.SIGKILL)
