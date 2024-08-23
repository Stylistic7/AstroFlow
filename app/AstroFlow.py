import tkinter as tk
from tkinter import scrolledtext
import subprocess
import threading
import sys
import os
from flask import Flask, render_template, request, session
import webbrowser
import time

# TODO: The goals here are to:
#
# 1. Check for the existence of miniconda. If it is not installed, install it.
# 2. Check for the existence and correctness of the AstroFlow environment.
#    If it doesn't exist, or isn't correct, delete it and create it anew.
# 3. Start the web server and launch the browser.
# 4. Start the Ray head node.
#    TODO: Offer options to connect to an existing head node.

# TODO: Prevent launching multiple copies on the same machine.

_ASTROFLOW_URL = 'http://localhost:8080'

app = Flask(__name__,
            template_folder="templates",
            static_folder="templates/static")


@app.route("/")
def index():
  return render_template('index.html')


if getattr(sys, 'frozen', False):
  # If the application is frozen (running as a standalone executable)
  base_path = sys._MEIPASS
else:
  # If the application is running in a development environment
  base_path = os.path.dirname(__file__)


def _log(text_widget, message, color=None):
  text_widget.insert(tk.END, message, color)
  text_widget.yview(tk.END)  # Auto-scroll to the end of the text widget
  text_widget.update_idletasks()


def install_conda(text_widget, root):
  expected_env_file = os.path.join(base_path, 'expected_env.yml')

  commands = [
      "ls -l",
      "conda --version",
      "conda init zsh",
      "source ~/.zshrc && conda env export --name AstroFlow > /tmp/current_env.yml",
      "ls -l /tmp/current_env.yml",
      f"ls -l {expected_env_file}",
      f"diff /tmp/current_env.yml {expected_env_file}",
  ]

  any_failure = False
  for command in commands:
    returncode = run_command(command, text_widget)

    if returncode != 0:
      any_failure = True
      break

  # TODO: Remove this hack.
  any_failure = False

  if not any_failure:
    text_widget.after(0, _log, text_widget,
                      "AstroFlow conda environment is good!\n", "green")
    root.withdraw()
    start_flask()
    start_ray_head_node()
    return

  # Install the Conda environment from scratch.
  # TODO: Make sure this works.
  # TODO: Pull the list of pip packages out somewhere.
  commands = [
      "source ~/.zshrc && conda env update --file environment.yml --prune",
      "source ~/.zshrc && conda activate AstroFlow && pip3 install spherical-geometry xisf largestinteriorrectangle xxhash python-statemachine",
      "source ~/.zshrc && conda activate AstroFlow && python3 -m unittest dependency_graph/test_dependency_graph.py"
  ]

  any_failure = False
  for command in commands:
    returncode = run_command(command, text_widget)

    if returncode != 0:
      any_failure = True
      break

  if not any_failure:
    text_widget.after(0, _log, text_widget,
                      "Successfully installed AstroFlow conda environment",
                      "green")
    root.withdraw()
    start_flask()
    start_ray_head_node()


def run_command(command, text_widget):
  # _log(text_widget, f"$ {command}\n", "green")
  text_widget.after(0, _log, text_widget, f"$ {command}\n", "green")

  # Run the command and get the output
  process = subprocess.Popen(command,
                             shell=True,
                             executable="/bin/zsh",
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             universal_newlines=True)

  for line in iter(process.stdout.readline, ""):
    # Insert the output into the Tkinter text widget
    text_widget.after(0, _log, text_widget, line, None)

  process.stdout.close()
  process.wait()

  text_widget.after(0, _log, text_widget, "\n", None)

  if process.returncode != 0:
    text_widget.after(0, _log, text_widget,
                      f"command failed with status {process.returncode}\n",
                      "red")

  return process.returncode


def start_flask():
  # change current directory to package root directory
  os.chdir(app.root_path)

  # TODO: Isn't this a race condition?
  print('Starting Flask server...')
  threading.Thread(target=lambda: app.run(
      host='0.0.0.0', port=8080, debug=True, use_reloader=False)).start()
  webbrowser.open(_ASTROFLOW_URL)


def start_ray_head_node_thread_body():
  # TODO: This may not work on Windows.
  command = """source ~/.zshrc && conda activate AstroFlow && ulimit -n 10000 && ray start --head --resources '{"PixInsight": 1, "diskcache": 1, "fastest_storage": 1}'"""
  subprocess.run(command, shell=True, start_new_session=True)

  # Spin until `ray status` tells us the node is up.
  while True:
    print('Calling `ray status`')
    process = subprocess.Popen(
        "source ~/.zshrc && conda activate AstroFlow && ray status",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        text=True)
    output = process.communicate()[0]
    print('ray status output:', output)
    if "Autoscaler status" in output:
      print('Ray head node is up')
      break

    time.sleep(1)


def start_ray_head_node():
  threading.Thread(target=start_ray_head_node_thread_body).start()


def reopen_handler():
  webbrowser.open(_ASTROFLOW_URL)


def main():
  # Create the Tkinter window
  root = tk.Tk()
  root.title("AstroFlow Configuration")

  # TODO: Intercept the tkinter CloseApplication event and run `ray stop`.
  root.createcommand("::tk::mac::ReopenApplication", reopen_handler)

  # Create a scrolled text widget
  text_widget = scrolledtext.ScrolledText(root,
                                          wrap=tk.WORD,
                                          height=40,
                                          width=160)

  text_widget.tag_configure("red", foreground="red")
  text_widget.tag_configure("green", foreground="green")
  text_widget.tag_configure("blue", foreground="blue")

  text_widget.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

  # Start the command automatically
  threading.Thread(target=install_conda, args=(text_widget, root),
                   daemon=True).start()

  # Configure the grid to expand and contract with the window size
  root.grid_columnconfigure(0, weight=1)
  root.grid_rowconfigure(0, weight=1)

  # Start the Tkinter main loop; this blocks until the application is quit.
  root.mainloop()


if __name__ == '__main__':
  main()
