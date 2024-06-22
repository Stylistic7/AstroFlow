from dateutil.parser import parse


def normalize_header(header: dict) -> dict:
  output = dict()
  for k, v in header.items():
    if k in ['COMMENT', 'HISTORY']:
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

  return output
