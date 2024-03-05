import tempfile
import os
import subprocess
import time
import concurrent.futures
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

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

MAX_INSTANCES = os.cpu_count()


def _clear_cache():
  with cache.transact():
    cache['pixinsight_blocking_instance'] = False
    cache['pixinsight_slots'] = []
    cache['pixinsight_idle_slots'] = []


def _wait_until_new_instance_is_allowed():
  while True:
    print('Waiting until another PixInsight process is allowed...')
    with cache.transact():
      pixinsight_blocking_instance = cache.get('pixinsight_blocking_instance',
                                               default=False)
      too_many_instances_running = len(cache.get('pixinsight_slots',
                                                 default=[])) >= MAX_INSTANCES

      if not pixinsight_blocking_instance and not too_many_instances_running:
        return

    time.sleep(5)


# TODO: Make the slot number configurable, including in unit tests (so the
# unit test won't disturb a simultaneous normal run).
def _start_pixinsight_instance(blocking=False):
  with cache.transact():
    idle_slots = cache.get('pixinsight_idle_slots', [])
    if idle_slots:
      slot = idle_slots.pop()
      print(f'Reusing idle slot {slot}')
      cache['pixinsight_idle_slots'] = idle_slots

      return slot

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

  print(f"Started new PixInsight instance at slot {slot}")

  # TODO: poll -e until the slot is actually running
  time.sleep(15)

  return slot


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
def run_script(tasks,
               blocking=True,
               tmp_dir_in=None,
               close_after=True,
               sleep_after=False):
  slot = _start_pixinsight_instance(blocking)
  print(f'PixInsight is now running in a background process in slot {slot}')

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


atexit.register(_stop_pixinsight_instances)

if __name__ == '__main__':
  run_script([], blocking=True)
