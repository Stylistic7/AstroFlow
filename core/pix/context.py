import inspect
import os
import uuid
from pathlib import Path

from pix.classes import ShellScriptTask, Image


def append_task(task: ShellScriptTask) -> None:
  frame = inspect.currentframe()

  assert frame is not None
  assert frame.f_globals is not None
  assert frame.f_back is not None

  while 'tasks' not in frame.f_globals:
    frame = frame.f_back

  frame.f_globals['tasks'].append(task)


def get_next_process_number() -> int:
  frame = inspect.currentframe()

  assert frame is not None
  assert frame.f_globals is not None
  assert frame.f_back is not None

  while 'process_number' not in frame.f_globals:
    frame = frame.f_back

  frame.f_globals['process_number'] += 1
  return frame.f_globals['process_number']


# def get_tmp_dir_obj():
#   frame = inspect.currentframe()
#   while 'tmp_dir_obj' not in frame.f_globals:
#     frame = frame.f_back

#   return frame.f_globals['tmp_dir_obj']

# def get_uuid_filename():
#   return os.path.join(get_tmp_dir_obj().name, f'{uuid.uuid4()}.xisf')


def extend_filename(image: Image, funcname: str) -> str:
  assert image.filename is not None
  path = Path(image.filename)

  if image.tag:
    return str(path.parent /
               (path.stem + '_' + funcname + '_' + image.tag + path.suffix))
  else:
    return str(path.parent / (path.stem + '_' + funcname + path.suffix))


# TODO: Using `tmp_dir` this way is akward. I should find a more specific way
# to detect the decorator's frame.
def add_images_to_save(images: list[Image]) -> None:
  frame = inspect.currentframe()

  assert frame is not None
  assert frame.f_globals is not None
  assert frame.f_back is not None

  while 'tmp_dir_obj' not in frame.f_globals:
    frame = frame.f_back

  frame.f_globals['images_to_save'].extend(images)


# def add_debug_images(images):
#   frame = inspect.currentframe()
#   while 'tmp_dir_obj' not in frame.f_globals:
#     frame = frame.f_back

#   frame.f_globals['debug_images'].extend(images)
