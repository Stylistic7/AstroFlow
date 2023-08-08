import traceback

from metaflow import Flow, FlowSpec, Parameter, catch, step, JSONType, schedule

from compact_repr import compact_repr


def find(step, output_names, input_names=None, inputs=None, debug=False):
  # FIXME remove this hack
  def equivalent(a, b):
    if type(a) is list and type(b) is list:
      return list(sorted(a)) == list(sorted(b))
    else:
      return a == b

  def value_or_none(task, name):
    try:
      return task[name].data
    except:
      return None

  sorted_runs = list(
      reversed(sorted(Flow('AstroFlow').runs(), key=lambda r: r.id)))[1:]
  for i, run in enumerate(sorted_runs):
    if debug: print(f'find: looking at run {i}')

    if step not in run:
      print(f'step {step} not in run {i}')
      continue

    try:
      for task in run[step]:
        if not task.successful:
          print(f'step {step} was not successful')
          continue

        if debug:
          for name, value in zip(input_names or [], inputs or []):
            if name not in task:
              print(f'find: input {name} is missing from task')

            if task[name].data == value:
              equality_string = ' == '
            elif type(
                task[name].data) is list and type(value) is list and sorted(
                    task[name].data) == sorted(value):
              equality_string = ' == (with sort) '
            else:
              equality_string = ' != '
            print(f'find: input {name}: {compact_repr(task[name].data)} '
                  f'{equality_string} {compact_repr(value)}')

        # If any of the inputs don't match, ignore this task.
        if not all(
            equivalent(task[name].data, value)
            for name, value in zip(input_names or [], inputs or [])):
          continue

        if debug:
          print('find: found matching run')
          print('version in task', 'version' in task.data)
          print('master in task', 'master' in task.data)

        if type(output_names) is list:
          output = [value_or_none(task, name) for name in output_names]
        else:
          output = value_or_none(task, output_names)

        if debug: print(f'find: returning output {output}')
        return output
    except Exception as e:
      print(traceback.format_exc())

  if type(output_names) is list:
    return [None for o in output_names]
  else:
    return None