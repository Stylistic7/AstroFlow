import tempfile
import os
import subprocess
import time
from contextlib import nullcontext
from typing import ContextManager, cast
import os
import sys
import uuid

from pix import ShellScriptTask

_SLOT_NUMBER: int = 1


def _start_pixinsight_instance(slot: int) -> int:
  # This should be start and detach. If it's already started, this
  # attempt should just exit immediately with no other effect.
  subprocess.Popen([
      '/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight',
      f'-n={slot}', '--no-startup-check-updates', '--no-startup-gui-messages',
      '--no-splash', '--without-printer'
  ],
                   start_new_session=True)

  print(f"Started new PixInsight instance at slot {slot}")

  # TODO: Poll until the PixInsight instance is reliably responding to pokes.
  time.sleep(30)

  return slot


def _save_javascript_file(tmp_dir: str,
                          task: ShellScriptTask) -> ShellScriptTask:
  if task.javascript and not task.javascript_filename:
    script_filename = os.path.join(tmp_dir, f'{uuid.uuid4()}.js')
    with open(script_filename, "w") as f:
      javascript = task.javascript
      if type(javascript) is list:
        javascript = '\n'.join(javascript)

      print(f'### Writing JavaScript to file {script_filename}')
      print(javascript)
      f.write(javascript)

    task.javascript_filename = script_filename

  return task


def _create_run_command(task: ShellScriptTask) -> str:
  arguments = ' '.join(
      f'-a="{name}={value}"' for name, value in task.arguments.items())
  parameters = ' '.join(
      f'-p="{name},{value}"' for name, value in task.parameters.items())
  return f'run -x {arguments} {parameters} "{task.javascript_filename}"'


# TODO: Maybe allow multiple PixInsight instances.
# TODO: Add a start sentinel that gets written to disk when the script starts.
# If, after some timeout, the script hasn't started, kill it and either retry
# or more on to the next one. Obey Ray's retry mechanism.
def run_script(tasks: list[ShellScriptTask],
               tmp_dir_in: str | None = None,
               close_after: bool = True,
               sleep_after: bool = False,
               dry_run: bool = False) -> None:
  assert tasks

  if not dry_run:
    start_pixinsight_instances()

  with (tempfile.TemporaryDirectory() if tmp_dir_in is None else cast(
      ContextManager, nullcontext(tmp_dir_in))) as tmp_dir:
    start_sentinel_filename = os.path.join(tmp_dir, 'start_sentinel.txt')
    end_sentinel_filename = os.path.join(tmp_dir, 'end_sentinel.txt')
    script_filename = os.path.join(tmp_dir, 'script.scp')

    tasks = [_save_javascript_file(tmp_dir, t) for t in tasks]
    script = f'![echo "start" > {start_sentinel_filename}\n'
    script += '\n\n'.join(_create_run_command(t) for t in tasks) + '\n\n'
    if close_after:
      script += f'close --force *\n'
    if sleep_after:
      script += f'!sleep 30\n'
    script += f'![echo "end" > {end_sentinel_filename}\n'

    with open(script_filename, "w") as f:
      print('### toplevel script is', script)
      f.write(script)

    if not dry_run:
      print(f'Running PixInsight shell script...')
      ret = subprocess.run([
          '/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight',
          f'--execute={_SLOT_NUMBER}:{script_filename}'
      ]).returncode
      print(f'--execute returned')

      while not os.path.exists(start_sentinel_filename):
        # TODO: Bring the PixInsight window to the front on other platforms,
        # too.
        print(f'Forcing PixInsight to the foreground...')
        if sys.platform == 'darwin':
          subprocess.run(
              """osascript -e 'tell application "PixInsight" to activate'""",
              shell=True)

        time.sleep(5)

      while not os.path.exists(end_sentinel_filename):
        time.sleep(1)


def start_pixinsight_instances() -> None:
  # At startup, open N instances of PixInsight, and let them run throughout the
  # entire runtime of AstroFlow.
  _start_pixinsight_instance(_SLOT_NUMBER)
