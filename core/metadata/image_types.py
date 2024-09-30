def is_light_frame(header: dict) -> bool:
  if not header:
    return False
  if 'IMAGETYP' not in header:
    return False
  image_type = header['IMAGETYP'].lower()
  return 'light' in image_type


def is_bias_frame(header: dict) -> bool:
  if not header:
    return False
  if 'IMAGETYP' not in header:
    return False
  image_type = header['IMAGETYP'].lower()
  return 'bias' in image_type


def is_dark_frame(header: dict) -> bool:
  if not header:
    return False
  if 'IMAGETYP' not in header:
    return False
  image_type = header['IMAGETYP'].lower()
  return 'dark' in image_type


def is_flat_frame(header: dict) -> bool:
  if not header:
    return False
  if 'IMAGETYP' not in header:
    return False
  image_type = header['IMAGETYP'].lower()
  return 'flat' in image_type


def is_frame_type(header: dict, type: str) -> bool:
  if type == 'light':
    return is_light_frame(header)
  elif type == 'dark':
    return is_dark_frame(header)
  elif type == 'bias':
    return is_bias_frame(header)
  elif type == 'flat':
    return is_flat_frame(header)

  raise ValueError(f'Unknown image type {type}')


def is_master(header: dict) -> bool:
  if not header:
    return False
  if 'IMAGETYP' not in header:
    return False
  image_type = header['IMAGETYP'].lower()
  return 'master' in image_type
