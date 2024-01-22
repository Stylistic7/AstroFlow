import numpy as np
from numpy import radians, degrees
import os

from distance import distance


def _read_names_catalog():
  entries = []
  with open('catalogs/names.dat') as f:
    for line in f.readlines():
      entry = {}
      entry['name'] = line[0:37].strip()
      entry['id'] = line[37:42].strip().replace(' ', '')

      entries.append(entry)

  return entries


def _read_ngc2000_catalog():
  names = _read_names_catalog()

  entries = []
  with open('catalogs/ngc2000.dat') as f:
    for line in f.readlines():
      entry = {}
      entry['id'] = line[0:5].strip()
      entry['type'] = line[6:9].strip()
      entry['ra_string'] = line[10:17].strip()
      hours, minutes = entry['ra_string'].split(' ')
      entry['ra_degrees'] = 15.0 * (float(hours) + float(minutes) / 60.0)
      entry['dec_string'] = line[18:25].strip()
      degrees, minutes = entry['dec_string'].split(' ')
      entry['dec_degrees'] = float(degrees) + float(minutes) / 60.0
      entry['reference'] = line[26:27].strip()
      entry['constellation'] = line[28:32].strip()

      try:
        entry['radius'] = float(line[32:38].strip()) / 60.0 / 2.0
      except:
        entry['radius'] = 0.0

      try:
        entry['magnitude'] = float(line[39:44].strip())
      except:
        entry['magnitude'] = 0.0

      entry['p'] = line[44:45].strip()
      entry['comment'] = line[46:].strip()

      # TODO: This all probably has a bunch of weird edge cases.
      entry['name'] = next((n['name'] for n in names if n['id'] == entry['id']),
                           None)
      if entry['name']:
        # Convert 'M  51' into 'M51'.
        if entry['name'].startswith('M '):
          entry['name'] = entry['name'].replace(' ', '')
      else:
        # Convert 'I 1396' into 'I1396'.
        # Convert '7000' into 'NGC7000'.
        id = entry['id'].replace(' ', '')
        if entry['id'].startswith('I'):
          entry['name'] = 'IC' + id[1:].strip()
        else:
          entry['name'] = 'NGC' + id

      entries.append(entry)

  return entries


def _find_best_object_by_size(ra, dec):
  # Sort the objects by descending radius.
  entries = sorted(_read_ngc2000_catalog(),
                   key=lambda x: x['radius'],
                   reverse=True)

  # Compute the normalized radial distance to the object. If the object is 120
  # minutes in size (e.g. 60 minutes in radius), and the given {ra, dec} is
  # 70 minutes away, the normalized radius is 70 / 60 = 1.16.
  #
  # We sort the entries by their normalized radius, and simply exclude objects
  # that do not have sizes. (We'll look through those later.) The object with
  # the smallest normalized radius is probably the right one.
  smallest_normalized_radius = np.inf
  best_distance = np.inf
  best_entry = None
  for entry in entries:
    if not entry['radius']:
      continue
    d = distance(radians(ra), radians(dec), radians(entry['ra_degrees']),
                 radians(entry['dec_degrees']))
    normalized_radius = d / entry['radius']

    if (normalized_radius < 2.0 and
        normalized_radius < smallest_normalized_radius):
      smallest_normalized_radius = normalized_radius
      best_distance = d
      best_entry = entry

  return best_distance, best_entry


def _find_best_object_by_distance(ra, dec):
  best_distance = np.inf
  best_entry = None
  for entry in entries:
    d = distance(radians(ra), radians(dec), radians(entry['ra_degrees']),
                 radians(entry['dec_degrees']))
    if d < best_distance:
      best_distance = d
      best_entry = entry

  return best_distance, best_entry


# @cache.memoize(tag='ngc2000_catalog')
def find_best_object(ra, dec):
  distance, entry = _find_best_object_by_size(ra, dec)
  if entry:
    return distance, entry

  return _find_best_object_by_distance(ra, dec)


if __name__ == '__main__':
  d = distance(ra_1=0, dec_1=0, ra_2=360, dec_2=0)
  print(f'distance is {d}')

  best_distance, best_entry = find_best_object(ra=325.4, dec=58.2)
  print('best_distance', best_distance)
  print('best_entry', best_entry)
