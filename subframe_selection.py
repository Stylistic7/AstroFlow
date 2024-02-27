class Fraction(float):
  pass


class StdDev(float):
  pass


def get_light_frames_sorted_by_stars(stack, plate_solutions):
  light_frames = [f for run in stack.runs for f in run.frames]
  light_frames = sorted(light_frames,
                        key=lambda f: plate_solutions[f].num_stars,
                        reverse=True)

  return [(f, plate_solutions[f].num_stars) for f in light_frames]
