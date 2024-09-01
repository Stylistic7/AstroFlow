import numpy as np
import operator

from metadata.actor import get_metadata


class StarCountFraction:
  threshold: float = 1.0

  def __init__(self, threshold):
    self.threshold = threshold

  def select_subframes(self, light_frames, plate_solutions):
    light_frames_and_num_stars = get_light_frames_sorted_by_stars(
        light_frames, plate_solutions)
    _, highest_star_count = light_frames_and_num_stars[0]

    threshold = self.threshold * highest_star_count

    return [
        f for f in light_frames if plate_solutions[f].num_stars >= threshold
    ]


class StarCountStdDev:
  threshold: float = 1.0

  def __init__(self, threshold):
    self.threshold = threshold

  def select_subframes(self, light_frames, plate_solutions):
    star_counts = [plate_solutions[f].num_stars for f in light_frames]
    mean = np.mean(star_counts)
    std_dev = np.std(star_counts)

    threshold = mean - self.threshold * std_dev

    return [
        f for f in light_frames if plate_solutions[f].num_stars >= threshold
    ]


def decode_selection_criterion(criterion: tuple[str, float]):
  name, threshold = criterion
  if name == 'StarCountFraction':
    return StarCountFraction(threshold)
  elif name == 'StarCountStdDev':
    return StarCountStdDev(threshold)


SelectionCriterion = StarCountFraction | StarCountStdDev


def get_light_frames_sorted_by_stars(light_frames, metadata=None):
  if metadata is None:
    metadata = get_metadata(light_frames)
  else:
    metadata = [i for h, i in metadata.items() if h in light_frames]

  return sorted(metadata,
                key=lambda i: i.plate_solution.num_stars,
                reverse=True)
