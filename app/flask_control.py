from flask import Flask, render_template, request, session
import threading
import webbrowser
import os
import subprocess

_ASTROFLOW_URL = 'http://localhost:80'

app = Flask(__name__,
            template_folder="templates",
            static_folder="templates/static")


@app.route("/")
def index():
  return render_template('index.html')


def start_flask():
  # change current directory to package root directory
  os.chdir(app.root_path)

  # TODO: Isn't this a race condition?
  print('Starting Flask server...')
  command = """source ~/.zshrc && conda activate AstroFlow && cd core && python3 -m web_server.backend.app"""
  subprocess.run(command, shell=True, start_new_session=True)

  # TODO: The subprocess above blocks forever, so we need another way to
  # open the web browser upon success. Maybe polling :80?
  webbrowser.open(_ASTROFLOW_URL)
