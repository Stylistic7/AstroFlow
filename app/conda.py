import os
import sys

from subprocesses import run_command
from console import log
from base_path import get_base_path


def _check_conda(text_widget):
  expected_env_file = os.path.join(get_base_path(), 'expected_env.yml')

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
    text_widget.after(0, log, text_widget,
                      "AstroFlow conda environment is good!\n", "green")

  return not any_failure


def _install_conda(text_widget):
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
    text_widget.after(0, log, text_widget,
                      "Successfully installed AstroFlow conda environment",
                      "green")

  return not any_failure


def ensure_conda(text_widget, root, callback_fn):
  if _check_conda(text_widget):
    root.withdraw()
    callback_fn()
  else:
    # TODO: Don't leave the user hanging here if the install failed.
    if _install_conda(text_widget):
      root.withdraw()
      callback_fn()
