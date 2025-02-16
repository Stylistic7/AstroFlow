import webbrowser
import sys
import tkinter as tk
from tkinter import scrolledtext
import Quartz

# from homebrew import ensure_homebrew
from subprocesses import SubprocessManager
from ray_control import start_ray_head_node

# TODO: The goals here are to:
# 1. Start the web server and launch the browser.
# 2. Start the Ray head node.
#    TODO: Offer options to connect to an existing head node.

# TODO: Prevent launching multiple copies on the same machine.

_ASTROFLOW_URL = 'http://localhost:80'


def is_option_key_pressed():
  """Check if the Option (⌥) key is currently held down."""
  flags = Quartz.CGEventSourceFlagsState(
      Quartz.kCGEventSourceStateHIDSystemState)
  return bool(flags & Quartz.kCGEventFlagMaskAlternate)


def open_web_ui(unused):
  webbrowser.open(_ASTROFLOW_URL)


class RedirectText:

  def __init__(self, text_widget):
    self.text_widget = text_widget

  def write(self, string):
    self.text_widget.insert(tk.END, string)
    self.text_widget.see(tk.END)

  def flush(self):
    pass


def main():
  # Create the Tkinter window
  root = tk.Tk()
  root.title("AstroFlow Configuration")
  root.deiconify()
  root.lift()

  def reopen_handler():
    if is_option_key_pressed():
      root.lift()
    else:
      webbrowser.open(_ASTROFLOW_URL)

  # TODO: Intercept the tkinter CloseApplication event and run `ray stop`.
  root.createcommand("::tk::mac::ReopenApplication", reopen_handler)
  root.createcommand("::tk::mac::Quit", root.quit)

  # Create a scrolled text widget
  text_widget = scrolledtext.ScrolledText(root,
                                          wrap=tk.WORD,
                                          height=40,
                                          width=160)

  text_widget.tag_configure("red", foreground="red")
  text_widget.tag_configure("green", foreground="green")
  text_widget.tag_configure("blue", foreground="blue")

  text_widget.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

  text_widget.insert(tk.END, f"sys.argv is {sys.argv}\n", "red")
  text_widget.see(tk.END)  # Auto-scroll

  # Configure the grid to expand and contract with the window size
  root.grid_columnconfigure(0, weight=1)
  root.grid_rowconfigure(0, weight=1)

  sys.stdout = RedirectText(text_widget)  # Redirect stdout

  manager = SubprocessManager(root, text_widget)
  start_ray_head_node(manager)

  root.protocol("WM_DELETE_WINDOW", manager.stop_all)  # Stop processes on close
  root.mainloop()


if __name__ == '__main__':
  main()
