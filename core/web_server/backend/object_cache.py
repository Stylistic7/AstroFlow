from context import context


# TODO: Respond to `reload_data` events.
def stacks():
  if hasattr(stacks, 'cached_value'):
    return stacks.cached_value
  else:
    try:
      stacks.cached_value = context.get('stacks')
      return stacks.cached_value
    except:
      return None


# TODO: Respond to `reload_data` events.
def targets():
  if hasattr(targets, 'cached_value'):
    return targets.cached_value
  else:
    try:
      targets.cached_value = context.get('targets')
      return targets.cached_value
    except:
      return None


# TODO: Respond to `reload_data` events.
def mosaics():
  if hasattr(mosaics, 'cached_value'):
    return mosaics.cached_value
  else:
    try:
      mosaics.cached_value = context.get('mosaics')
      return mosaics.cached_value
    except:
      return None
