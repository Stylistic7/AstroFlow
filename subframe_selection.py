from typing import NewType
import numpy as np


class StarCountFraction:
  threshold: float = 1.0

  def __init__(self, threshold):
    self.threshold = threshold

  def select_subframes(self, stack, plate_solutions):
    light_frames_and_num_stars = get_light_frames_sorted_by_stars(
        stack, plate_solutions)
    _, highest_star_count = light_frames_and_num_stars[0]

    return [
        f for f, n in light_frames_and_num_stars
        if n >= self.threshold * highest_star_count
    ]


class StarCountStdDev:
  threshold: float = 1.0

  def __init__(self, threshold):
    self.threshold = threshold

  def select_subframes(self, stack, plate_solutions):
    light_frames = [f for run in stack.runs for f in run.frames]
    star_counts = [plate_solutions[f].num_stars for f in light_frames]
    mean = np.mean(star_counts)
    std_dev = np.std(star_counts)

    return [
        f for f in light_frames
        if plate_solutions[f].num_stars >= (mean - self.threshold * std_dev)
    ]


def decode_selection_criterion(criterion: tuple[str, float]):
  name, threshold = criterion
  if name == 'StarCountFraction':
    return StarCountFraction(threshold)
  elif name == 'StarCountStdDev':
    return StarCountStdDev(threshold)


SelectionCriterion = StarCountFraction | StarCountStdDev


def get_light_frames_sorted_by_stars(stack, plate_solutions):
  light_frames = [f for run in stack.runs for f in run.frames]
  light_frames = sorted(light_frames,
                        key=lambda f: plate_solutions[f].num_stars,
                        reverse=True)

  return [(f, plate_solutions[f].num_stars) for f in light_frames]
