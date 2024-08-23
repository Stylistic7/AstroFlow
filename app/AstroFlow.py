import tkinter as tk
from tkinter import scrolledtext
import threading
import sys
import os
import webbrowser
from conda import ensure_conda
from ray_control import start_ray_head_node
from flask_control import start_flask

# TODO: The goals here are to:
#
# 1. Check for the existence of miniconda. If it is not installed, install it.
# 2. Check for the existence and correctness of the AstroFlow environment.
#    If it doesn't exist, or isn't correct, delete it and create it anew.
# 3. Start the web server and launch the browser.
# 4. Start the Ray head node.
#    TODO: Offer options to connect to an existing head node.

# TODO: Prevent launching multiple copies on the same machine.

_ASTROFLOW_URL = 'http://localhost:80'


def reopen_handler():
  webbrowser.open(_ASTROFLOW_URL)


def start_services():
  # TODO: Let the user choose which Ray cluster to connect to through the UI.
  start_ray_head_node(start_flask)


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
  threading.Thread(target=ensure_conda,
                   args=(text_widget, root, start_services),
                   daemon=True).start()

  # Configure the grid to expand and contract with the window size
  root.grid_columnconfigure(0, weight=1)
  root.grid_rowconfigure(0, weight=1)

  # Start the Tkinter main loop; this blocks until the application is quit.
  root.mainloop()


if __name__ == '__main__':
  main()
