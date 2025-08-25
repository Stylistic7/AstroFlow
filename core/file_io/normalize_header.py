from dateutil.parser import parse


def normalize_header(header: dict) -> dict:
  output = dict()
  for k, v in header.items():
    if k in ['COMMENT', 'HISTORY']:
      continue

    if k == 'IMAGETYP':
      output[k] = v.lower()
      continue

    # Try to coerce the type to a date.
    if 'DATE' in k:
      try:
        output[k] = parse(v + ' UTC')
        continue
      except Exception as e:
        output[k] = v

    # Leave it as a string.
    output[k] = v

  if 'EXPTIME' in header and 'EXPOSURE' not in header:
    header['EXPOSURE'] = float(header['EXPTIME'])
    print(f"### Setting EXPOSURE to EXPTIME, value {header['EXPOSURE']}")

  return output
