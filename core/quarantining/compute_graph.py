import ray

from context import context


# TODO: These don't belong in the compute_graph.py file
@ray.remote
def quarantine_stacks():
  stacks = context.grouped_stacks
  if not stacks:
    return {}

  quarantined_stacks = context.quarantined_stacks or set()
  return {
      stack_id: stack
      for stack_id, stack in stacks.items()
      if stack_id not in quarantined_stacks
  }


@ray.remote
def quarantine_targets():
  targets = context.grouped_targets
  print(f'### quarantine_targets {len(targets)} grouped targets')
  if not targets:
    return {}

  quarantined_targets = context.quarantined_targets or set()
  return {
      target_id: target
      for target_id, target in targets.items()
      if target_id not in quarantined_targets
  }


@ray.remote
def quarantine_mosaics():
  mosaics = context.grouped_mosaics
  print(f'### quarantine_targets {len(mosaics)} grouped targets')
  if not mosaics:
    return {}

  quarantined_mosaics = context.quarantined_mosaics or set()
  return {
      mosaic_id: mosaic
      for mosaic_id, mosaic in mosaics.items()
      if mosaic_id not in quarantined_mosaics
  }
