import numpy as np


def compute_viewbox(points):
  xs = [p[0] for p in points]
  ys = [p[1] for p in points]
  x_min, x_max = np.min(xs), np.max(xs)
  y_min, y_max = np.min(ys), np.max(ys)

  return x_min, y_min, x_max - x_min, y_max - y_min


class Projection:
  def sample_viewbox_lon_lat(self, viewbox, num_samples_x):
    x_min, y_min, x_span, y_span = viewbox
    resolution = x_span / num_samples_x
    num_samples_y = int(y_span / resolution)

    xx, yy = np.meshgrid(np.linspace(x_min, x_min + x_span, num_samples_x),
                         np.linspace(y_min, y_min + y_span, num_samples_y))

    return self.unproject(xx, yy)