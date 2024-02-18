import datetime
import pytz
import itertools


def run_time_delta(a, b):
  start_a, end_a = a
  midpoint_a = start_a + (end_a - start_a) / 2

  start_b, end_b = b
  midpoint_b = start_b + (end_b - start_a) / 2

  return abs(midpoint_a - midpoint_b)


# TODO: Put the minimum stack size in the UI.
def get_runs(frames,
             fits_headers,
             min_frames=3,
             max_gap=datetime.timedelta(minutes=15)):
  if not frames:
    return []
  frames = sorted(frames, key=lambda f: fits_headers[f]['DATE-OBS'])

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
        if frame_count >= min_frames:
          runs.append((run_start, prev_date_obs))

      # Start a new run.
      run_start = current_date_obs
      frame_count = 1
    else:
      frame_count += 1

    prev_date_obs = current_date_obs

  if frame_count >= min_frames:
    runs.append((run_start, prev_date_obs))

  return runs


def get_frames_for_run(run, frames, fits_headers):
  assert len(frames) == len(set(frames)), "Input contained duplicate frames"

  start, end = run
  result = [
      f for f in frames if fits_headers[f]['DATE-OBS'] >= start and
      fits_headers[f]['DATE-OBS'] <= end
  ]

  assert None not in result
  assert len(result) == len(set(result)), "Output contained duplicate frames"
  return result


def get_frames_for_runs(runs, frames, fits_headers):
  return sorted(
      itertools.chain.from_iterable(
          get_frames_for_run(run, frames, fits_headers) for run in runs))
