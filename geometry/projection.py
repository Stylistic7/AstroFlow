import numpy as np


class Projection:

  def sample_viewbox_lon_lat(self, viewbox, num_samples_x):
    x_min, y_min, x_span, y_span = viewbox
    resolution = x_span / num_samples_x
    num_samples_y = int(y_span / resolution)

    xx, yy = np.meshgrid(np.linspace(x_min, x_min + x_span, num_samples_x),
                         np.linspace(y_min, y_min + y_span, num_samples_y))

    return self.unproject(xx, yy)
