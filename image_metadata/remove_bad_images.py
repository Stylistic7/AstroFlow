from context import context
from plate_solving import PlateSolution


# TODO: This is a bit weird. We want to keep unsolvable files in the cache,
# so we don't attempt to re-solve them every run. But we want to filter them
# out so downstream logic doesn't see them. We also want to ignore plate
# solutions for images that no longer exist, but ultimately we want to help
# the user find them.
#
# TODO: Report unsolvable images and missing images to the user. Include the
# cause of failure whenever possible, and provide a solution (e.g. increasing
# timeout on slow computers).
def remove_bad_images():
  image_hashes = context.image_hashes
  plate_solutions = context.plate_solutions
  quarantined_files = context.quarantined_files

  return {
      h: p
      for h, p in plate_solutions.items()
      if h in image_hashes and image_hashes[h] and
      h not in quarantined_files and type(p) is PlateSolution
  }
