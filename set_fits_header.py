import astropy.io.fits as pyfits
from pathlib import Path

path = "/Users/nobody/Projects/Astro/M51 Whirlpool"
filenames = list(str(f) for f in Path(path).rglob("*.fit"))

for filename in filenames:
  with pyfits.open(filename, mode='update') as fits:
    # fits[0].header['INSTRUME'] = 'ZWO ASI2600MM Pro'
    # fits[0].header['IMAGETYP'] = 'Dark'
    # if 'FILTER' in fits[0].header:
    #   del fits[0].header['FILTER']

    # print(fits[0].header['IMAGETYP'])

    if fits[0].header['IMAGETYP'] == 'Flat' and 'RIG' not in fits[0].header:
      print(f'Found file {filename} with no RIG header')
      fits[0].header['RIG'] = 'C11_F7'

      # if 'C11' in filename:
      #   fits[0].header['RIG'] == 'C11'
      # elif 'RC71' in filename:
      #   fits[0].header['RIG'] == 'RC71'
