import datetime
import pytz
import itertools


def get_runs(frames, fits_headers):
  if not frames: return []
  frames = sorted(frames, key=lambda f: fits_headers[f]['DATE-OBS'])

  max_gap = datetime.timedelta(hours=1)
  runs = []
  prev_date_obs = pytz.utc.localize(datetime.datetime.min)
  run_start = None
  frame_count = 0
  for frame in frames:
    current_date_obs = fits_headers[frame]['DATE-OBS']
    if current_date_obs > prev_date_obs + max_gap:
      # Save the current run, if there is one.
      if run_start:
        assert frame_count
        runs.append((run_start, prev_date_obs))

      # Start a new run.
      run_start = current_date_obs
      frame_count = 1
    else:
      frame_count += 1

    prev_date_obs = current_date_obs

  runs.append((run_start, prev_date_obs))

  return runs


def get_frames_for_run(run, frames, fits_headers):
  # print(f'### get_frames_for_run run {run} frames {frames}')
  start, end = run
  result = [
      f for f in frames if fits_headers[f]['DATE-OBS'] >= start and
      fits_headers[f]['DATE-OBS'] <= end
  ]
  assert None not in result
  # print(f'### result {result}')
  return result


def get_frames_for_runs(runs, frames, fits_headers):
  L = [get_frames_for_run(run, frames, fits_headers) for run in runs]
  print('### L', L)
  return sorted(itertools.chain.from_iterable(L))