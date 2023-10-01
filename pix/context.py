import inspect


def append_task(task):
  frame = inspect.currentframe()
  while 'tasks' not in frame.f_globals:
    frame = frame.f_back

  frame.f_globals['tasks'].append(task)


def get_next_process_number():
  frame = inspect.currentframe()
  while 'process_number' not in frame.f_globals:
    frame = frame.f_back

  frame.f_globals['process_number'] += 1
  return frame.f_globals['process_number']


def get_tmp_dir():
  frame = inspect.currentframe()
  while 'tmp_dir' not in frame.f_globals:
    frame = frame.f_back

  return frame.f_globals['tmp_dir']
