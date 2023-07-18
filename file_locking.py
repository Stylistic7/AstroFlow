import time
import os


def remove_lockfile(lockfile_directory, resource_name):
  lock_filename = os.path.join(lockfile_directory, f"{resource_name}.lock")
  if os.path.exists(lock_filename):
    os.remove(lock_filename)


def serialize(lockfile_directory, resource_name, fn, *args, **kwargs):
  lock_filename = os.path.join(lockfile_directory, f"{resource_name}.lock")
  while True:
    try:
      os.open(lock_filename, os.O_CREAT | os.O_EXCL)
      print(f'Acquired lock for resource {resource_name}')
      result = fn(*args, **kwargs)
      os.remove(lock_filename)
      return result
    except FileExistsError:
      # print(f'Waiting for lock on resource {resource_name}')
      time.sleep(5)
    except Exception as e:
      print(f'serialize: Caught exception {e}')