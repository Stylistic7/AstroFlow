import ngc2000_catalog


def ra_dec_string(ra, dec):
  def round_to_nearest_tenth(number):
    return int(round(number * 10.0)) / 10.0

  ra = round_to_nearest_tenth(ra)
  dec = round_to_nearest_tenth(dec)

  return f'RA_{ra:.2f}_DEC_{dec:.2f}'


def canonical_target_string(position):
  ra, dec = position

  distance, entry = ngc2000_catalog.find_best_object(ra, dec)
  if entry and entry['name']:
    return entry['name']
  else:
    return ra_dec_string(ra, dec)
