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
    exposure_s = sum(fits_headers[h]['EXPOSURE'] for h in frames)
    content_hash = hash_list(frames)
    output[content_hash] = Run(frames=frames,
                               content_hash=content_hash,
                               type='light',
                               num_frames=len(frames),
                               exposure_s=exposure_s)

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
