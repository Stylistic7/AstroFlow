from flask import Flask, render_template, request, session
import threading
import webbrowser
import os

from web_server import stacks

_ASTROFLOW_URL = 'http://localhost:8080'

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
  threading.Thread(target=lambda: app.run(
      host='0.0.0.0', port=8080, debug=True, use_reloader=False)).start()
  webbrowser.open(_ASTROFLOW_URL)
