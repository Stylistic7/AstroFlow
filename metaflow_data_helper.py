from metaflow import Flow, FlowSpec, Parameter, catch, step, JSONType, schedule


def find_most_recent_successful_task_output(step,
                                            keys=[],
                                            values=[],
                                            output_keys=[],
                                            debug=False):
  task = find_most_recent_successful_task(step, keys, values, debug)
  if not task:
    if debug: print('Task was false')
    return None

  output_values = []
  for k in output_keys:
    if k not in task.data:
      if debug: print(f'Output key {k} was not in task.data')
      return None
    output_values.append(task[k].data)

  if debug: print(f'Returning output_values {output_values}')
  return output_values


def find_most_recent_successful_task(step, keys=[], values=[], debug=False):
  sorted_runs = list(
      reversed(sorted(Flow('AstroFlow').runs(), key=lambda r: r.id)))[1:]
  for run in sorted_runs:
    try:
      if debug: print(f'Looking at run {run}')
      for task in run[step]:
        if debug: print(f'  Looking at task {task}')
        if not task.successful:
          if debug: print(f'Task was not successful')
          continue
        else:
          if debug: print(f'Task was successful!')
          pass

        failed = False
        for k, v in zip(keys, values):
          if k not in task.data:
            if debug: print(f'Data key {k} not in task.data')
            break
          else:
            if debug: print(f'Data key {k} is in task.data!')
            pass

          if task[k].data == v:
            if debug: print(f'Found value for data key {k}')
            pass
          else:
            if debug:
              print(f'Value for data key {k} ({task[k].data}) '
                    f'did not match value ({v})')
            failed = True
            break

        if not failed:
          if debug: print('Found a matching task.')
          return task
    except Exception as e:
      if debug:
        print(f'Exception encountered while querying run {run.id}: {str(e)}')
      pass

  if debug: print('Did not find a matching task.')
  return None


if __name__ == '__main__':
  find_most_recent_successful_task('measure_for_rig', ['light_frames'],
                                   [['foo']])