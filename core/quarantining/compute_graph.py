import ray

from context import context


# TODO: These don't belong in the compute_graph.py file
@ray.remote
def delete_quarantined_stacks(ancestor_results):
  stacks = ancestor_results[0]
  quarantined_stacks = context.quarantined_stacks or set()
  return {
      stack_id: stack
      for stack_id, stack in stacks.items()
      if stack_id not in quarantined_stacks
  }


@ray.remote
def delete_quarantined_targets(ancestor_results):
  targets = ancestor_results[0]
  quarantined_targets = context.quarantined_targets or set()
  return {
      target_id: target
      for target_id, target in targets.items()
      if target_id not in quarantined_targets
  }


@ray.remote
def delete_quarantined_mosaics(ancestor_results):
  mosaics = ancestor_results[0]
  quarantined_mosaics = context.quarantined_mosaics or set()
  return {
      mosaic_id: mosaic
      for mosaic_id, mosaic in mosaics.items()
      if mosaic_id not in quarantined_mosaics
  }
