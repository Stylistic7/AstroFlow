# if __name__ == '__main__':
#   import logging
#   logging.basicConfig(filename='/tmp/myapp.log', level=logging.DEBUG)
#   logging.debug('This is a debug message')

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
# 1. Check for the existence of miniconda. If it is not installed, install it.
# 2. Check for the existence and correctness of the AstroFlow environment.
#    If it doesn't exist, or isn't correct, delete it and create it anew.
# 3. Start the web server and launch the browser.

app = Flask(__name__)


@app.route('/')
def index():
  return 'AstroFlow online!'


if getattr(sys, 'frozen', False):
  # If the application is frozen (running as a standalone executable)
  base_path = sys._MEIPASS
else:
  # If the application is running in a development environment
  base_path = os.path.dirname(__file__)


def _log(text_widget, message, color=None):
  text_widget.insert(tk.END, message, color)
  text_widget.yview(tk.END)  # Auto-scroll to the end of the text widget


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
    _log(text_widget, "AstroFlow conda environment is good!\n", "green")
    root.destroy()

  # TODO: Make sure this works.
  commands = [
      "source ~/.zshrc && conda env update --file environment.yml --prune",
      "source ~/.zshrc && conda activate AstroFlow && pip3 install spherical-geometry xisf largestinteriorrectangle xxhash",
      "source ~/.zshrc && conda activate AstroFlow && python3 -m unittest dependency_graph/test_dependency_graph.py"
  ]

  any_failure = False
  for command in commands:
    returncode = run_command(command, text_widget)

    if returncode != 0:
      any_failure = True
      break

  if not any_failure:
    _log(text_widget, "Successfully installed AstroFlow conda environment",
         "green")
    root.destroy()


def run_command(command, text_widget):
  _log(text_widget, f"$ {command}\n", "green")

  # Run the command and get the output
  process = subprocess.Popen(command,
                             shell=True,
                             executable="/bin/zsh",
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             universal_newlines=True)

  for line in iter(process.stdout.readline, ""):
    # Insert the output into the Tkinter text widget
    _log(text_widget, line)
    text_widget.update_idletasks()  # Ensure the GUI is updated

  process.stdout.close()
  process.wait()

  _log(text_widget, "\n")

  if process.returncode != 0:
    _log(text_widget, f"command failed with status {process.returncode}\n",
         "red")

  return process.returncode


def start_flask():
  # change current directory to package root directory
  os.chdir(app.root_path)

  # TODO: Isn't this a race condition?
  print('Starting Flask server...')
  webbrowser.open('http://localhost:8080')
  app.run(host='0.0.0.0', port=8080)


def main():
  # Create the Tkinter window
  root = tk.Tk()
  root.title("AstroFlow Configuration")

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

  # Start the Tkinter main loop; this blocks until the `install_conda()` function
  # is done.
  root.mainloop()
  root.quit()

  # All the tkinter stuff is done; start the Flask server.
  start_flask()


if __name__ == '__main__':
  main()
