import subprocess

from console import log


def run_command(command, text_widget):
  text_widget.after(0, log, text_widget, f"$ {command}\n", "green")

  # Run the command and get the output
  process = subprocess.Popen(command,
                             shell=True,
                             executable="/bin/zsh",
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             universal_newlines=True)

  for line in iter(process.stdout.readline, ""):
    # Insert the output into the Tkinter text widget
    text_widget.after(0, log, text_widget, line, None)

  process.stdout.close()
  process.wait()

  text_widget.after(0, log, text_widget, "\n", None)

  if process.returncode != 0:
    text_widget.after(0, log, text_widget,
                      f"command failed with status {process.returncode}\n",
                      "red")

  return process.returncode
