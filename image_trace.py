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

  if found:
    print(f'Image hash {hash} was found in spatial group with '
          f'name {spatial_group_name} and {len(spatial_group_hashes)} members')
  else:
    print(f'Image hash {hash} is not included in any spatial group')

  if hash in stacks:
    print(f'Hash {hash} points to stack {stacks[hash]}')

  found = False
  for stack_id, stack in stacks.items():
    for run in stack.runs:
      if hash in run.frames:
        found = True
        break

  if not found:
    print(f"Image hash {hash} is not included in any stack's runs")
  else:
    print(f'Image hash {hash} is in a run of stack {stack}')

  found = False
  for stack_id, stack in stacks.items():
    for run in stack.runs:
      if hash in run.stack_dependencies:
        found = True
        break

  if not found:
    print(f"Image hash {hash} is not included in any stack's dependencies")
  else:
    print(f'Image hash {hash} is a dependency of stack {stack}')


# filename_fragments = [
#     'Light_NGC2244_120.0s_Bin1_6200MM_G_20240209-014820_0001.fit',
#     'Light_NGC2244_120.0s_Bin1_6200MM_G_20240209-015022_0002.fit',
#     'Light_NGC2244_120.0s_Bin1_6200MM_G_20240209-015224_0003.fit',
#     'Light_NGC2244_120.0s_Bin1_6200MM_G_20240209-015426_0004.fit',
#     'Light_NGC2244_120.0s_Bin1_6200MM_G_20240209-015628_0005.fit',
#     'Light_NGC2244_120.0s_Bin1_6200MM_G_20240209-015920_0006.fit',
#     'Light_NGC2244_120.0s_Bin1_6200MM_G_20240209-020122_0007.fit',
#     'Light_NGC2244_120.0s_Bin1_6200MM_G_20240209-020324_0008.fit',
#     'Light_NGC2244_120.0s_Bin1_6200MM_G_20240209-020526_0009.fit',
#     'Light_NGC2244_120.0s_Bin1_6200MM_G_20240209-020728_0010.fit',
# ]

# for filename_fragment in filename_fragments:
#   trace_filename_fragment(filename_fragment)

hashes = ['5dfa470f72c02bd9']

for hash in hashes:
  trace_image_hash(hash)
