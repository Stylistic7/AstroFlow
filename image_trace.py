from cache import cache

image_hashes = cache['image_hashes']
plate_solutions = cache['plate_solutions']
stacks = cache['stacks']


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

  if hash not in plate_solutions.keys():
    print(f'Filename fragment {filename_fragment} is not in plate_solutions')
    return

  if plate_solutions[hash] is None:
    print(f'Filename fragment {filename_fragment} has None for plate solution')
    return

  print(f'Filename fragment {filename_fragment} has '
        f'{plate_solutions[hash].num_stars} stars')

  found = False
  for stack_id, stack in stacks.items():
    for run in stack.runs:
      if hash in run.frames:
        found = True
        break

  if not found:
    print(
        f'Filename fragment "{filename_fragment} is not included in any stack')
    return

  print(f'Filename fragment {filename_fragment} is in stack {stack}')


filename_fragments = [
    "Light_M13_300.0s_Bin1_6200MM_L_20240331-230559_0001.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240331-231934_0002.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240331-232436_0003.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240331-232938_0004.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240331-233440_0005.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240331-234024_0006.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240331-234526_0007.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240331-235027_0008.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240331-235529_0009.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240401-000031_0010.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240401-000604_0011.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240401-001106_0012.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240401-002119_0013.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240401-002621_0014.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240401-003122_0015.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240401-003804_0016.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240401-004305_0017.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240401-004807_0018.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240401-005309_0019.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240401-005811_0020.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240401-010349_0021.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240401-010850_0022.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240401-011352_0023.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240401-011854_0024.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240401-012918_0025.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240401-013448_0026.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240401-013950_0027.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240401-014452_0028.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240401-014954_0029.fit",
    "Light_M13_300.0s_Bin1_6200MM_L_20240401-015456_0030.fit",
]

for filename_fragment in filename_fragments:
  trace_filename_fragment(filename_fragment)
