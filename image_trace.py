from cache import cache

quarantined_files = cache.get('quarantined_files', [])
spatial_groups = cache.get('spatial_groups', [])
image_hashes = cache.get('image_hashes', {})
fits_headers = cache.get('fits_headers', {})
plate_solutions = cache.get('plate_solutions', {})
shopping_cart = cache.get('shopping_cart', [])
stacks = cache.get('stacks', {})
targets = cache.get('targets', {})
mosaics = cache.get('mosaics', {})

print(f'Cache contained {len(quarantined_files)} '
      f'quarantined files')
print(f'Cache contained {len(image_hashes)} image hashes')
print(f'Cache contained {len(fits_headers)} FITS headers')
print(f'Cache contained {len(plate_solutions)} plate solutions')
print(f'Cache contained {len(shopping_cart)} items in shopping cart')
print(f'Cache contained {len(spatial_groups)} spatial groups')
print(f'Cache contained {len(stacks)} stacks')
print(f'Cache contained {len(targets)} targets')
print(f'Cache contained {len(mosaics)} mosaics')


def trace_filename_fragment(filename_fragment):
  found = False
  for hash, filename in image_hashes.items():
    if filename_fragment in filename:
      found = True
      break

  if not found:
    print(f'Filename fragment "{filename_fragment}" is not in image_hashes')
    return

  print(f'Filename fragment {filename_fragment} has hash {hash}')

  trace_image_hash(hash)


def trace_image_hash(hash):
  if hash not in plate_solutions.keys():
    print(f'Image hash {hash} is not in plate_solutions')
  else:
    if plate_solutions[hash] is None:
      print(f'Image hash {hash} has None for plate solution')
    else:
      plate_solution = plate_solutions[hash]
      print(f'Image hash {hash} has '
            f'{plate_solution.num_stars} stars and center '
            f'{plate_solution.center}')

  found = False
  spatial_group_name = None
  for name, hashes in spatial_groups:
    if hash in hashes:
      spatial_group_name = name
      spatial_group_hashes = hashes
      found = True

  if not found:
    print(f'Image hash {hash} is not included in any spatial group')
    return

  print(f'Image hash {hash} was found in spatial group with '
        f'name {spatial_group_name} and {len(spatial_group_hashes)} members')

  if hash in stacks:
    print(f'Hash {hash} points to stack {stacks[hash]}')

  found = False
  for stack_id, stack in stacks.items():
    if hash in stack.frames:
      found = True
      break

  if not found:
    print(f"Image hash {hash} is not included in any stack")
    return

  print(f'Image hash {hash} is in stack {stack}')

  found = False
  for parent_stack in stacks.values():
    if stack.content_hash in parent_stack.dependencies.values():
      found = True
      break

  if not found:
    print(f"Stack {stack} is not included in any stack's dependencies")
  else:
    print(f'Stack {stack} is a dependency of stack {parent_stack}')

  found = False
  for containing_stack in stacks.values():
    if stack.content_hash in containing_stack.contents:
      found = True
      break

  if not found:
    print(f"Stack {stack} is not included in any stack's contents")
  else:
    print(f'Stack {stack} is contained by stack {containing_stack}')

  found = False
  for target in targets.values():
    if containing_stack.content_hash in target.stack_ids:
      print(
          f'Target {target} contains stack {containing_stack} via content hash')
      found = True
    if containing_stack.stable_hash in target.stack_ids:
      print(
          f'Target {target} contains stack {containing_stack} via geometric hash'
      )
      found = True

  if not found:
    print(f"Stack {containing_stack} was not contained in any target.")


# filename_fragments = [
#     'Light_M16_300.0s_Bin1_6200MM_G_20240523-022520_0002',
# ]
filename_fragments = [
    'Light_M16_300.0s_Bin1_6200MM_B_20240523-030301_0003',
]
for filename_fragment in filename_fragments:
  trace_filename_fragment(filename_fragment)

# hashes = ['5dfa470f72c02bd9']
# for hash in hashes:
#   trace_image_hash(hash)
