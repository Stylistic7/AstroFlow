from metaflow import Flow, FlowSpec, Parameter, catch, step, JSONType, schedule


def find_most_recent_successful_task(step, keys, values):
  sorted_runs = reversed(
      list(sorted(Flow('AstroFlow').runs(), key=lambda r: r.id)))
  for run in sorted_runs:
    try:
      # print(f'Looking at run {run}')
      for task in run[step]:
        # print(f'Looking at task {task}')
        if not task.successful:
          # print(f'Task was not successful')
          continue
        else:
          # print(f'Task was successful!')
          pass

        failed = False
        for k, v in zip(keys, values):
          if k not in task.data:
            # print(f'Data key {k} not in task.data')
            break
          else:
            # print(f'Data key {k} is in task.data!')
            pass

          if sorted(task[k].data) == sorted(v):
            # print(f'Found value for data key {k}')
            pass
          else:
            # print(f'Value for data key {k} (len {len(task[k].data)}) '
            #       f'did not match value (len {len(v)})')
            failed = True
            break

        if not failed:
          return task
    except:
      # print(f'Exception encountered while querying run {run.id}')
      pass

  return None


if __name__ == '__main__':
  find_most_recent_successful_task('measure_for_rig', ['light_frames'],
                                   [['foo']])