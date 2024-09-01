from stacks import SimpleStack, CompositeStack


# TODO: Make these "Stack graph query" functions universal.
# TODO: Give these better names.
def get_toplevel_stacks(stacks):
  return [s for s in stacks.values() if type(s) is CompositeStack]


def get_content_stacks_of_type(stacks, image_type):
  return [
      s for s in stacks.values()
      if s.image_type == image_type and type(s) is SimpleStack
  ]
