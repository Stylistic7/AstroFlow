# TODO: Make these "Stack graph query" functions universal.
def get_toplevel_stacks(stacks):
  return [s for s in stacks.values() if s.stable_hash]


def get_content_stacks_of_type(stacks, image_type):
  return [
      s for s in stacks.values()
      if s.image_type == image_type and not s.stable_hash
  ]
