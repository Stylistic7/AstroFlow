from dataclasses import dataclass, field
import tempfile
import os
import subprocess
import time
import concurrent.futures
from contextlib import nullcontext
import os
import diskcache
import inspect
import functools
import uuid

from . import open_images

cache = diskcache.Cache(os.path.join(os.getcwd(), '.cache'))

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _clear_cache():
  with cache.transact():
    cache['pixinsight_blocking_instance'] = False
    cache['pixinsight_slots'] = []


def _wait_until_new_instance_is_allowed():
  while True:
    with cache.transact():
      pixinsight_blocking_instance = cache.get('pixinsight_blocking_instance',
                                               default=False)
      too_many_instances_running = len(
          cache.get('pixinsight_slots', default=[])) >= os.cpu_count()

      if not pixinsight_blocking_instance and not too_many_instances_running:
        return

    time.sleep(5)


def _start_pixinsight_instance(blocking=False):
  _wait_until_new_instance_is_allowed()

  with cache.transact():
    slots = cache.get('pixinsight_slots', default=[])
    print(f'Running PixInsight slots:', slots)
    slot = max(slots) + 1 if slots else 100
    cache['pixinsight_slots'] = slots + [slot]
    if blocking:
      cache['pixinsight_blocking_instance'] = True

  def done_callback(future):
    with cache.transact():
      print(f'PixInsight slot {slot} exited.')
      slots = cache['pixinsight_slots']
      slots.remove(slot)
      cache['pixinsight_slots'] = slots
      if blocking:
        cache['pixinsight_blocking_instance'] = False

  f = executor.submit(subprocess.call, [
      '/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight',
      f'-n={slot}', '--no-startup-check-updates', '--no-startup-gui-messages',
      '--no-splash', '--without-printer'
  ])
  f.add_done_callback(done_callback)

  print(f"Started new PixInsight instance at slot {slot}, "
        f"cache['pixinsight_slots'] is {cache['pixinsight_slots']}")

  # FIXME poll -e until the slot is actually running
  time.sleep(15)

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


# TODO: Re-use already launched PixInsight instance(s) that are idle.
def run_script(tasks, blocking=True, tmp_dir_in=None, sleep_after=False):
  slot = _start_pixinsight_instance(blocking)
  print(f'PixInsight is now running in a background process in slot {slot}')

  with tempfile.TemporaryDirectory() if not tmp_dir_in else nullcontext(
      tmp_dir_in) as tmp_dir:
    print(f'### run_script() tmp_dir {tmp_dir}')
    sentinel_filename = os.path.join(tmp_dir, 'sentinel.txt')
    script_filename = os.path.join(tmp_dir, 'script.scp')

    tasks = [_save_javascript_file(tmp_dir, t) for t in tasks]
    script = '\n\n'.join(_create_run_command(t) for t in tasks) + '\n\n'
    if sleep_after:
      script += f'!sleep {30 * (slot - 99)}\n'
    script += f'![echo "done" > {sentinel_filename}\n'
    script += f'exit\n'

    with open(script_filename, "w") as f:
      print('script is', script)
      f.write(script)

    print(f'Running PixInsight shell script...')
    ret = subprocess.run([
        '/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight',
        f'--execute={slot}:{script_filename}'
    ]).returncode
    print(f'--execute returned')

    while not os.path.exists(sentinel_filename):
      time.sleep(1)


if __name__ == '__main__':
  run_script([], blocking=True)
