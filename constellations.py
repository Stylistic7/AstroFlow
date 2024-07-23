import csv
import numpy as np
import logging

import geometry.projections.mollweide as mollweide

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

  polylines = []
  polyline = []

  def end_polyline(polyline):
    if not polyline:
      return

    logging.debug(f'Polyline had {len(polyline)} points')
    polylines.append(polyline)

  for row in rows:
    if row[1] == '':
      end_polyline(polyline)
      polyline = []
      continue

    # TODO: Process all of the points at once in a batch.
    ra, dec = np.radians(15 * float(row[2])), np.radians(float(row[3]))
    theta, phi = mollweide.equatorial_to_galactic(ra, dec)
    polyline.append((theta, phi))

  assert len(polyline) == 0

  return polylines


if __name__ == '__main__':
  print(polylines())
