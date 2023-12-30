import csv
import numpy as np
import logging

import mollweide

constellation_line_path = "catalogs/sfa_constellation_lines.csv"


def _read_constellation_csv():
  rows = []
  with open(constellation_line_path) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    num_rows = 0
    for i, row in enumerate(reader):
      assert len(row) == 14
      if i > 7:
        rows.append(row)

  return rows


# TODO: Memoize this.
def polylines():
  rows = _read_constellation_csv()
  proj = mollweide.MollweideProjection()

  polylines = []
  polyline = []
  min_theta, max_theta = np.inf, -np.inf
  last_theta = None

  def end_polyline(polyline):
    if not polyline:
      return

    logging.debug(f'Polyline had {len(polyline)} points')
    polylines.append(polyline)
    last_theta = None

  for row in rows:
    if row[1] == '':
      end_polyline(polyline)
      polyline = []
      continue

    # TODO: Process all of the points at once in a batch.
    ra, dec = 15 * float(row[2]), float(row[3])
    theta, phi = mollweide.equatorial_to_galactic(ra, dec)

    # TODO: This is a hacky way to remove "around the edge" lines.
    if last_theta and np.abs(theta - last_theta) > np.radians(270):
      logging.debug(f'Ending polyline because theta {theta} and '
                    f'last_theta {last_theta}')
      end_polyline(polyline)
      polyline = []
      continue

    last_theta = theta

    min_theta = min(min_theta, theta)
    max_theta = max(max_theta, theta)

    x, y = proj.project(theta, phi)
    polyline.append((x, y))

  assert len(polyline) == 0

  logging.debug('min_theta', min_theta, 'max_theta', max_theta)

  min_x = np.min([p[0] for polyline in polylines for p in polyline])
  max_x = np.max([p[0] for polyline in polylines for p in polyline])
  min_y = np.min([p[1] for polyline in polylines for p in polyline])
  max_y = np.max([p[1] for polyline in polylines for p in polyline])
  logging.debug('min_x', min_x, 'max_x', max_x)
  logging.debug('min_y', min_y, 'max_y', max_y)

  return polylines


if __name__ == '__main__':
  print(polylines())