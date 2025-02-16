import tkinter as tk
import subprocess
import select
import time


class SubprocessManager:

  def __init__(self, root, text_widget):
    self.root = root
    self.text_widget = text_widget
    self.processes = []
    self.process_output = {}
    self.done_callbacks = {}

    # Start polling for output
    self.root.after(100, self.check_output)

  def start_process(self, command, done_callback=None):
    """Start a subprocess and store it in the list."""
    self.log(f"$ {command}\n", "green")

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # Line-buffered output
    )
    self.processes.append(process)
    self.process_output[process] = ""

    if done_callback is not None:
      self.done_callbacks[process] = done_callback

  def check_output(self):
    for process in self.processes[:]:  # Iterate over a copy to avoid modification issues
      ready, _, _ = select.select([process.stdout], [], [], 0)
      if ready:
        output = process.stdout.read()
        if output:
          self.log(output, None)
          self.process_output[process] += output

      if process.poll() is not None:  # Process has exited
        if process in self.processes:  # Ensure it's still in the list before removing
          self.processes.remove(process)
          if process in self.done_callbacks:
            self.done_callbacks[process](self.process_output[process])
            del self.done_callbacks[process]

    # Schedule next poll
    self.root.after(100, self.check_output)

  def log(self, message, tag=None):
    """Append text to the text widget."""
    self.text_widget.insert(tk.END, message, tag)
    self.text_widget.see(tk.END)  # Auto-scroll

  def stop_all(self):
    """Stop all running subprocesses."""
    for process in self.processes:
      process.terminate()
    self.processes.clear()
