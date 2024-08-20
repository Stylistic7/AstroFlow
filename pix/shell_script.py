import tempfile
import os
import subprocess
import time
from contextlib import nullcontext
import os
import uuid

_SLOT_NUMBER = 1


def _start_pixinsight_instance(slot):
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


def _save_javascript_file(tmp_dir, task):
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


def _create_run_command(task):
  arguments = ' '.join(
      f'-a="{name}={value}"' for name, value in task.arguments.items())
  parameters = ' '.join(
      f'-p="{name},{value}"' for name, value in task.parameters.items())
  return f'run -x {arguments} {parameters} "{task.javascript_filename}"'


# TODO: Maybe allow multiple PixInsight instances.
# TODO: Add a start sentinel that gets written to disk when the script starts.
# If, after some timeout, the script hasn't started, kill it and either retry
# or more on to the next one. Obey Ray's retry mechanism.
def run_script(tasks,
               blocking=True,
               tmp_dir_in=None,
               close_after=True,
               sleep_after=False,
               dry_run=False):
  assert tasks

  if not dry_run:
    start_pixinsight_instances()

  with tempfile.TemporaryDirectory() if not tmp_dir_in else nullcontext(
      tmp_dir_in) as tmp_dir:
    sentinel_filename = os.path.join(tmp_dir, 'sentinel.txt')
    script_filename = os.path.join(tmp_dir, 'script.scp')

    tasks = [_save_javascript_file(tmp_dir, t) for t in tasks]
    script = '\n\n'.join(_create_run_command(t) for t in tasks) + '\n\n'
    if close_after:
      script += f'close --force *\n'
    if sleep_after:
      script += f'!sleep 30\n'
    script += f'![echo "done" > {sentinel_filename}\n'

    with open(script_filename, "w") as f:
      print('script is', script)
      f.write(script)

    if not dry_run:
      print(f'Forcing PixInsight to the foreground...')
      # TODO: Implement for MacOS, then check on other platforms.

      print(f'Running PixInsight shell script...')
      ret = subprocess.run([
          '/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight',
          f'--execute={_SLOT_NUMBER}:{script_filename}'
      ]).returncode
      print(f'--execute returned')

      while not os.path.exists(sentinel_filename):
        time.sleep(1)


def start_pixinsight_instances():
  # At startup, open N instances of PixInsight, and let them run throughout the
  # entire runtime of AstroFlow.
  _start_pixinsight_instance(_SLOT_NUMBER)
