import tempfile
import os
import subprocess
import time
import functools
from contextlib import nullcontext
import os
import uuid
import diskcache
import atexit

# TODO: This prevents a circular import, but could be cleaned up.
from . import open_images

# This cache is in the /tmp directory, which is shared among all processes on
# a node.
cache = diskcache.Cache('/tmp/AstroFlow/pix_cache')

MAX_INSTANCES = 1


def _clear_cache():
  with cache.transact():
    cache['pixinsight_blocking_instance'] = False
    cache['pixinsight_slots'] = []
    cache['pixinsight_idle_slots'] = []


def _start_pixinsight_instance(slot):
  subprocess.Popen([
      '/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight',
      f'-n={slot}', '--no-startup-check-updates', '--no-startup-gui-messages',
      '--no-splash', '--without-printer'
  ])

  print(f"Started new PixInsight instance at slot {slot}")

  with cache.transact():
    slots = cache.get('pixinsight_slots', default=[])
    assert slot not in slots
    slots.append(slot)
    cache['pixinsight_slots'] = slots

    idle_slots = cache.get('pixinsight_idle_slots', [])
    assert slot not in idle_slots
    idle_slots.append(slot)
    cache['pixinsight_idle_slots'] = idle_slots

  return slot


def _wait_until_slot_is_idle():
  while True:
    with cache.transact():
      idle_slots = cache.get('pixinsight_idle_slots', [])
      if idle_slots:
        slot = idle_slots.pop()
        print(f'Reusing idle slot {slot}')
        cache['pixinsight_idle_slots'] = idle_slots
        return slot

    print('Waiting until a PixInsight slot is open...')
    time.sleep(5)


def _slot_is_idle(slot):
  with cache.transact():
    if slot not in cache.get('pixinsight_slots', default=[]):
      print(f'### Error: PixInsight slot {slot} exited, but is not in '
            f'pixinsight_slots')

    idle_slots = cache.get('pixinsight_idle_slots', [])
    if slot in idle_slots:
      print(f'### Error: PixInsight slot {slot} exited, but is in '
            f'pixinsight_idle_slots')

    idle_slots.append(slot)
    print(f'Slot {slot} is now idle')
    cache['pixinsight_idle_slots'] = idle_slots


def _save_javascript_file(tmp_dir, task):
  if task.javascript and not task.javascript_filename:
    script_filename = os.path.join(tmp_dir, f'{uuid.uuid4()}.js')
    with open(script_filename, "w") as f:
      javascript = task.javascript
      if type(javascript) is list:
        javascript = '\n'.join(javascript)

      print(f'Writing JavaScript to file {script_filename}')
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
               sleep_after=False):
  slot = _wait_until_slot_is_idle()

  with tempfile.TemporaryDirectory() if not tmp_dir_in else nullcontext(
      tmp_dir_in) as tmp_dir:
    sentinel_filename = os.path.join(tmp_dir, 'sentinel.txt')
    script_filename = os.path.join(tmp_dir, 'script.scp')

    tasks = [_save_javascript_file(tmp_dir, t) for t in tasks]
    script = '\n\n'.join(_create_run_command(t) for t in tasks) + '\n\n'
    if close_after:
      script += f'close --force *\n'
    if sleep_after:
      script += f'!sleep {30 * (slot - 99)}\n'
    script += f'![echo "done" > {sentinel_filename}\n'

    # TODO: Gracefully quit PixInsight instances using the `atexit` module.
    # script += f'exit\n'

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

    _slot_is_idle(slot)


def _stop_pixinsight_instances():
  for slot in cache.get('pixinsight_slots', []):
    subprocess.run([
        '/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight',
        f'--terminate={slot}'
    ])
    _clear_cache()


def start_pixinsight_instances():
  # At startup, open N instances of PixInsight, and let them run throughout the
  # entire runtime of AstroFlow.
  _clear_cache()
  for slot in range(100, 100 + MAX_INSTANCES):
    print(f'### Starting PixInsight instance in slot {slot}')
    _start_pixinsight_instance(slot)


# When AstroFlow exits, stop all the instances.
atexit.register(_stop_pixinsight_instances)
