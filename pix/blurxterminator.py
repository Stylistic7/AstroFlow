from shell_script import append_task, ShellScriptTask, get_next_process_number, accepts_view_ids

_default_parameters = {
    'ai_file': 'BlurXTerminator.2.mlpackage',
    'correct_only': False,
    'correct_first': False,
    'nonstellar_then_stellar': False,
    'lum_only': False,
    'sharpen_stars': 0.25,
    'adjust_halos': 0.00,
    'nonstellar_psf_diameter': 0.00,
    'auto_nonstellar_psf': True,
    'sharpen_nonstellar': 0.90,
}


# TODO: This stuff should be hidden or automated.
def _format(value):
  if value is True:
    return 'true'
  elif value is False:
    return 'false'
  elif type(value) is float:
    return str(value)
  elif type(value) is str:
    return f'"{value}"'


@accepts_view_ids
def BlurXTerminator(view_ids, **kwargs):
  print(f'### Entering BlurXTerminator, view_ids {view_ids}')
  parameters = _default_parameters | kwargs

  process_number = get_next_process_number()
  print(f'### BlurXTerminator is using process number {process_number}')

  # TODO: The process_number stuff should be automatically hidden.
  code = [f'var P{process_number} = new BlurXTerminator;']
  for k, v in parameters.items():
    code.append(f'P{process_number}.{k} = {_format(v)}')

  for view_id in view_ids:
    code.append(f'P{process_number}.executeOn(View.viewById("{view_id}"))')

  append_task(ShellScriptTask(javascript=code))