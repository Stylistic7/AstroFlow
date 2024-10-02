import numpy as np


def compute_viewbox(
    points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
  xs = [p[0] for p in points]
  ys = [p[1] for p in points]
  x_min, x_max = np.min(xs), np.max(xs)
  y_min, y_max = np.min(ys), np.max(ys)

  return x_min, y_min, x_max - x_min, y_max - y_min
