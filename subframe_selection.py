from typing import NewType
import numpy as np
import copy
import itertools
import file_io

from hashing import hash_list
from runs import get_runs, get_frames_for_run
from stack import Stack, Run, FrameHash


def _recompute_runs_by_star_count(context, runs, threshold, fits_headers,
                                  plate_solutions):
  # TODO: Why is this here?
  # assert not any(run.stack_dependencies for run in runs)

  flattened = list(
      itertools.chain.from_iterable([r.frames for r in runs.values()]))
  selected = [f for f in flattened if plate_solutions[f].num_stars >= threshold]

  output = {}
  for run in get_runs(selected, fits_headers, context.max_run_gap,
                      context.min_frames_in_run):
    frames = get_frames_for_run(run, selected, fits_headers)
    assert all(file_io.is_light_frame(fits_headers[f]) for f in frames)

    # TODO: Move this into a common place. Use it to fill in the metadata of
    # both Runs and Stacks.
    def get_quality(quality):
      qualities = set(fits_headers[f][quality] for f in frames)
      assert len(qualities) < 2, f'multiple qualities {qualities}'
      return next(iter(qualities)) if len(qualities) else None

    camera = get_quality('INSTRUME')
    exposure_s = get_quality('EXPOSURE')
    filter = get_quality('FILTER')

    all_dates = sorted(fits_headers[f]['DATE-OBS'] for f in frames)
    mean_temperature = np.mean([fits_headers[f]['CCD-TEMP'] for f in frames])

    content_hash = hash_list(frames)
    output[content_hash] = Run(frames=frames,
                               content_hash=content_hash,
                               type='light',
                               num_frames=len(frames),
                               camera=camera,
                               mean_temperature=mean_temperature,
                               exposure_s=exposure_s,
                               filter=filter,
                               oldest_date=all_dates[0],
                               newest_date=all_dates[-1])

  return output


class StarCountFraction:
  threshold: float = 1.0

  def __init__(self, threshold):
    self.threshold = threshold

  def select_subframes(self, context, runs, fits_headers, plate_solutions):
    light_frames_and_num_stars = get_light_frames_sorted_by_stars(
        runs, plate_solutions)
    _, highest_star_count = light_frames_and_num_stars[0]

    threshold = self.threshold * highest_star_count

    return _recompute_runs_by_star_count(context, runs, threshold, fits_headers,
                                         plate_solutions)


class StarCountStdDev:
  threshold: float = 1.0

  def __init__(self, threshold):
    self.threshold = threshold

  def select_subframes(self, context, runs, fits_headers, plate_solutions):
    light_frames = [f for run in runs.values() for f in run.frames]
    star_counts = [plate_solutions[f].num_stars for f in light_frames]
    mean = np.mean(star_counts)
    std_dev = np.std(star_counts)

    threshold = mean - self.threshold * std_dev

    return _recompute_runs_by_star_count(context, runs, threshold, fits_headers,
                                         plate_solutions)


def decode_selection_criterion(criterion: tuple[str, float]):
  name, threshold = criterion
  if name == 'StarCountFraction':
    return StarCountFraction(threshold)
  elif name == 'StarCountStdDev':
    return StarCountStdDev(threshold)


SelectionCriterion = StarCountFraction | StarCountStdDev


def get_light_frames_sorted_by_stars(runs, plate_solutions):
  light_frames = [f for run in runs.values() for f in run.frames]
  light_frames = sorted(light_frames,
                        key=lambda f: plate_solutions[f].num_stars,
                        reverse=True)

  return [(f, plate_solutions[f].num_stars) for f in light_frames]
