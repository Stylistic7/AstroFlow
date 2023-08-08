import functools
import os
import time
import traceback


def serialize(resource_name, lockfile_directory='.'):
  def decorator_serialize(fn):
    @functools.wraps(fn)
    def wrapper_serialize(*args, **kwargs):
      lock_filename = os.path.join(lockfile_directory,
                                   f"AstroFlow.{resource_name}.lock")
      while True:
        try:
          os.open(lock_filename, os.O_CREAT | os.O_EXCL)
          print(
              f'Acquired lock for resource {resource_name} with lock filename '
              f'{lock_filename}')
          result = fn(*args, **kwargs)
          os.remove(lock_filename)
          print(
              f'Released lock for resource {resource_name} with lock filename '
              f'{lock_filename}')
          return result
        except FileExistsError:
          time.sleep(5)
        except Exception as e:
          print('serialization: received unexpected exception',
                traceback.format_exc())

    return wrapper_serialize

  return decorator_serialize


def acquire_lock(resource_name, lockfile_directory='.'):
  lock_filename = os.path.join(lockfile_directory,
                               f"AstroFlow.{resource_name}.lock")
  while True:
    try:
      os.open(lock_filename, os.O_CREAT | os.O_EXCL)
      print(f'Acquired lock for resource {resource_name} with lock filename '
            f'{lock_filename}')
      break
    except FileExistsError:
      time.sleep(5)
    except Exception as e:
      print('serialization: received unexpected exception',
            traceback.format_exc())

  print('acquire_lock returning')


def release_lock(resource_name, lockfile_directory='.'):
  lock_filename = os.path.join(lockfile_directory,
                               f"AstroFlow.{resource_name}.lock")
  if os.path.exists(lock_filename):
    os.remove(lock_filename)

  print(f'Released lock for resource {resource_name} with lock filename '
        f'{lock_filename}')
