from .shell_script import get_next_process_number


def _format(value):
  if value is True:
    return 'true'
  elif value is False:
    return 'false'

  return str(value)


def code_for_process(process_name, parameters, view_ids):
  process_number = get_next_process_number()
  code = [f'var P{process_number} = new {process_name};']
  for k, v in parameters.items():
    code.append(f'P{process_number}.{k} = {_format(v)};')

  for view_id in view_ids:
    code.append(f'P{process_number}.executeOn(View.viewById("{view_id}"));')

  return code