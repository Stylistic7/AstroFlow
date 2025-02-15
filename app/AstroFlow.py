import webbrowser
import sys
import runpy
import tkinter as tk
from tkinter import scrolledtext
import threading

# from homebrew import ensure_homebrew
from console import log
from ray_control import start_ray_head_node
from flask_control import start_flask, stop_flask

# TODO: The goals here are to:
# 1. Start the web server and launch the browser.
# 2. Start the Ray head node.
#    TODO: Offer options to connect to an existing head node.

# TODO: Prevent launching multiple copies on the same machine.

_ASTROFLOW_URL = 'http://localhost:80'


def reopen_handler():
  webbrowser.open(_ASTROFLOW_URL)


def quit_handler():
  stop_flask()

  # TODO: Maybe stop the Ray head node. The action taken depends on the user's
  # preferences.


def start_services(text_widget, root):
  # TODO: Let the user choose which Ray cluster to connect to through the UI.
  text_widget.after(0, log, text_widget, "Starting Ray head node...", None)
  ray_process = start_ray_head_node()

  text_widget.after(0, log, text_widget, "Starting Flask server...", None)
  flask_process = start_flask()

  return [ray_process, flask_process]


def main():
  # Create the Tkinter window
  root = tk.Tk()
  root.title("AstroFlow Configuration")

  # TODO: Intercept the tkinter CloseApplication event and run `ray stop`.
  root.createcommand("::tk::mac::ReopenApplication", reopen_handler)
  root.createcommand("::tk::mac::Quit", quit_handler)

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
  threading.Thread(target=start_services, args=(text_widget, root),
                   daemon=True).start()

  # processes = start_services(text_widget)
  # for process in processes:
  #   process.wait()

  # Configure the grid to expand and contract with the window size
  root.grid_columnconfigure(0, weight=1)
  root.grid_rowconfigure(0, weight=1)

  # Start the Tkinter main loop; this blocks until the application is quit.
  root.mainloop()


if __name__ == '__main__':
  if "--subprocess" in sys.argv:
    print("Running as a subprocess.")
    runpy.run_module("web_server.backend.app",
                     run_name="__main__",
                     alter_sys=True)

  else:
    print('### Starting AstroFlow...')

    # processes = start_services()

    # webbrowser.open(_ASTROFLOW_URL)

    # for process in processes:
    #   process.wait()

    main()

    print('### AstroFlow done')
