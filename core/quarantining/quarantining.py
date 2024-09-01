import ray

from context import context


@ray.remote
def quarantine_stacks():
  # TODO: Walk through stacks, and remove quarantined stacks from `contents`
  # and `dependencies`. Remove quarantined and down-selected images from `frames`.
  context.stacks = {
      stack_id: stack
      for stack_id, stack in context.stacks.items()
      if stack_id not in (context.quarantine_stacks or set())
  }


@ray.remote
def quarantine_targets():
  context.targets = {
      target_id: target
      for target_id, target in context.targets.items()
      if target_id not in (context.quarantine_targets or set())
  }


@ray.remote
def quarantine_mosaics():
  context.mosaics = {
      mosaic_id: mosaic
      for mosaic_id, mosaic in context.mosaics.items()
      if mosaic_id not in (context.quarantine_mosaics or set())
  }
