import astropy.io.fits as pyfits
from pathlib import Path

path = "/Users/nobody/Projects/Astro/Raw/2023-10-06"
filenames = list(str(f) for f in Path(path).rglob("*.fit"))

num_files = 0
for filename in filenames:
  print(f'filename {filename}')
  with pyfits.open(filename, mode='update') as fits:
    print(f"INSTRUME '{fits[0].header['INSTRUME']}'")
    if 'RIG' in fits[0].header:
      print(f"RIG '{fits[0].header['RIG']}'")

    if 'RIG' not in fits[0].header:
      if fits[0].header['INSTRUME'] == 'ZWO ASI2600MM Pro':
        fits[0].header['RIG'] = 'RC71'
        num_files += 1
      elif fits[0].header['INSTRUME'] == 'ZWO ASI6200MM Pro':
        fits[0].header['RIG'] = 'C11'
        num_files += 1

    # if fits[0].header['RIG'] == 'C11':
    #   fits[0].header['FILTER'] = 'H'
    #   num_files += 1

  # fits[0].header['INSTRUME'] = 'ZWO ASI2600MM Pro'
  # fits[0].header['IMAGETYP'] = 'Dark'
  # if 'FILTER' in fits[0].header:
  #   del fits[0].header['FILTER']

  # print(fits[0].header['IMAGETYP'])

  # if fits[0].header['IMAGETYP'] == 'Flat' and 'RIG' not in fits[0].header:
  #   print(f'Found file {filename} with no RIG header')
  #   fits[0].header['RIG'] = 'C11_F7'

  # if 'C11' in filename:
  #   fits[0].header['RIG'] == 'C11'
  # elif 'RC71' in filename:
  #   fits[0].header['RIG'] == 'RC71'

print(f'Modified {num_files} files')