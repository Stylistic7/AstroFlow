from context import context


# TODO: Respond to `reload_data` events.
def cached_stacks():
  if hasattr(cached_stacks, 'cached_value'):
    return cached_stacks.cached_value
  else:
    try:
      cached_stacks.cached_value = context.get('stacks', {})
      return cached_stacks.cached_value
    except:
      return None


# TODO: Respond to `reload_data` events.
def cached_calibration_stacks():
  if hasattr(cached_calibration_stacks, 'cached_value'):
    return cached_calibration_stacks.cached_value
  else:
    try:
      cached_calibration_stacks.cached_value = context.get(
          'calibration_stacks', {})
      return cached_calibration_stacks.cached_value
    except:
      return None


# TODO: Respond to `reload_data` events.
def cached_targets():
  if hasattr(cached_targets, 'cached_value'):
    return cached_targets.cached_value
  else:
    try:
      cached_targets.cached_value = context.get('targets', {})
      return cached_targets.cached_value
    except:
      return None


# TODO: Respond to `reload_data` events.
def cached_mosaics():
  if hasattr(cached_mosaics, 'cached_value'):
    return cached_mosaics.cached_value
  else:
    try:
      cached_mosaics.cached_value = context.get('mosaics', {})
      return cached_mosaics.cached_value
    except:
      return None
